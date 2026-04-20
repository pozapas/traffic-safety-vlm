"""
video_engine.py
─────────────────────────────────────────────────────────────────
Video ingestion layer:
  1. Use yt-dlp to extract the best direct stream URL
  2. Use OpenCV to open the stream and sample frames
  3. Encode frames as base64 PNG for VLM submission

Works with both live YouTube streams and VODs.
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Generator, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Stream URL extraction
# ──────────────────────────────────────────────

_YTDLP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Format selectors tried in order — most lenient last
_FORMAT_FALLBACKS = [
    "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]",
    "best[height<=720]",
    "best",
]


def get_stream_url(
    yt_url: str,
    format_selector: str = None,
    timeout: int = 45,
) -> tuple[Optional[str], Optional[str]]:
    """
    Extract the direct stream URL using yt-dlp with browser cookies,
    geo-bypass, and format fallbacks to handle YouTube bot-detection.
    Returns (stream_url, error_message).
    """
    selectors = [format_selector] if format_selector else _FORMAT_FALLBACKS

    base_flags = [
        "--no-warnings", "--quiet", "-g", "--no-playlist",
        "--geo-bypass",
        "--user-agent", _YTDLP_USER_AGENT,
        "--add-header", "Accept-Language:en-US,en;q=0.9",
        "--extractor-retries", "3",
        "--socket-timeout", "15",
    ]

    # Pick first available browser for cookie extraction
    cookie_flags: list[str] = ["--cookies-from-browser", "chrome"]

    last_err = "Unknown error"
    for fmt in selectors:
        cmd = ["yt-dlp"] + base_flags + cookie_flags + ["-f", fmt, yt_url]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                url = result.stdout.strip().split("\n")[0].strip()
                if url:
                    return url, None
                last_err = "yt-dlp returned an empty URL"
            else:
                last_err = result.stderr.strip() or "yt-dlp non-zero exit"
                # Retry without cookies — some videos reject cookie-auth
                cmd2 = ["yt-dlp"] + base_flags + ["-f", fmt, yt_url]
                r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=timeout)
                if r2.returncode == 0:
                    url = r2.stdout.strip().split("\n")[0].strip()
                    if url:
                        return url, None
                last_err = r2.stderr.strip() or last_err
        except subprocess.TimeoutExpired:
            last_err = f"yt-dlp timed out after {timeout}s"
        except FileNotFoundError:
            return None, "yt-dlp not found — run: pip install yt-dlp"
        except Exception as e:
            last_err = str(e)

    return None, last_err



def get_video_metadata(yt_url: str) -> dict:
    """Fetch video title and duration metadata via yt-dlp."""
    cmd = [
        "yt-dlp", "--no-warnings", "--quiet",
        "--skip-download",
        "--print", "%(title)s|||%(duration_string)s|||%(view_count)s|||%(is_live)s",
        "--no-playlist",
        yt_url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            parts = result.stdout.strip().split("|||")
            if len(parts) >= 4:
                return {
                    "title": parts[0],
                    "duration": parts[1],
                    "views": parts[2],
                    "is_live": parts[3].lower() == "true",
                }
    except Exception:
        pass
    return {"title": yt_url, "duration": "unknown", "views": "N/A", "is_live": False}


# ──────────────────────────────────────────────
# Frame sampler
# ──────────────────────────────────────────────

class FrameSample:
    """Container for a single extracted frame."""

    def __init__(
        self,
        frame_index: int,
        timestamp_s: float,
        frame_bgr: np.ndarray,
    ):
        self.frame_index = frame_index
        self.timestamp_s = timestamp_s
        self.frame_bgr = frame_bgr
        self._b64: Optional[str] = None

    @property
    def base64_png(self) -> str:
        if self._b64 is None:
            rgb = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            # Resize to max 1280px wide to control token costs
            max_w = 1280
            if pil.width > max_w:
                ratio = max_w / pil.width
                pil = pil.resize(
                    (max_w, int(pil.height * ratio)),
                    Image.LANCZOS,
                )
            buf = io.BytesIO()
            pil.save(buf, format="PNG", optimize=True)
            self._b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return self._b64

    @property
    def pil_image(self) -> Image.Image:
        rgb = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @property
    def thumbnail(self) -> Image.Image:
        """Small thumbnail for UI preview."""
        img = self.pil_image
        img.thumbnail((320, 180), Image.LANCZOS)
        return img


class FrameSampler:
    """
    Opens a video stream (via URL) and yields FrameSample objects
    at a configurable interval.
    """

    def __init__(
        self,
        stream_url: str,
        interval_s: int = 10,
        max_frames: int = 6,
        skip_seconds: int = 0,
    ):
        self.stream_url = stream_url
        self.interval_s = interval_s
        self.max_frames = max_frames
        self.skip_seconds = skip_seconds

    def sample(
        self,
        progress_callback=None,
    ) -> list[FrameSample]:
        """
        Sample frames from the stream and return a list of FrameSample objects.

        Args:
            progress_callback: optional callable(current_frame, total_frames)
        """
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            raise IOError(f"Cannot open stream: {self.stream_url[:80]}…")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames_vid = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        is_live = total_frames_vid <= 0  # live streams report -1 or 0

        frames: list[FrameSample] = []
        frame_idx = 0

        # Skip initial seconds for live streams (buffer settling)
        if self.skip_seconds > 0 and not is_live:
            skip_frame = int(self.skip_seconds * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frame)

        interval_frames = max(1, int(fps * self.interval_s))
        read_attempts = 0
        max_read_attempts = self.max_frames * interval_frames * 3

        last_captured_pos = -interval_frames  # force capture of first frame

        while len(frames) < self.max_frames:
            if read_attempts > max_read_attempts:
                logger.warning("Max read attempts reached; stopping sampler.")
                break

            ret, frame = cap.read()
            read_attempts += 1

            if not ret:
                if is_live:
                    time.sleep(0.5)
                    continue
                else:
                    break

            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if current_pos - last_captured_pos >= interval_frames:
                timestamp_s = current_pos / fps
                sample = FrameSample(
                    frame_index=len(frames),
                    timestamp_s=round(timestamp_s, 2),
                    frame_bgr=frame.copy(),
                )
                frames.append(sample)
                last_captured_pos = current_pos
                frame_idx += 1

                if progress_callback:
                    progress_callback(len(frames), self.max_frames)

                # For VODs: jump ahead to next interval
                if not is_live:
                    next_pos = current_pos + interval_frames - 1
                    if next_pos < total_frames_vid:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, next_pos)

        cap.release()
        return frames


def _sample_frames_with_ffmpeg(
    stream_url: str,
    interval_s: int,
    max_frames: int,
    timeout: int = 120,
) -> tuple[list[FrameSample], Optional[str]]:
    """
    Fallback extractor for hosted environments where OpenCV cannot open
    signed YouTube HLS/DASH manifests directly.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return [], "ffmpeg is not installed; add it as a system dependency for deployment."

    clip_duration_s = max(interval_s * max_frames + interval_s, 12)

    try:
        with tempfile.TemporaryDirectory(prefix="vlm_ffmpeg_") as tmp_dir:
            output_pattern = str(Path(tmp_dir) / "frame_%03d.png")
            cmd = [
                ffmpeg_bin,
                "-y",
                "-nostdin",
                "-loglevel",
                "error",
                "-rw_timeout",
                "15000000",
                "-i",
                stream_url,
                "-t",
                str(clip_duration_s),
                "-an",
                "-vf",
                f"fps=1/{max(1, interval_s)}",
                "-frames:v",
                str(max_frames),
                output_pattern,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode != 0:
                err = result.stderr.strip() or "ffmpeg failed to extract frames"
                return [], err

            frames: list[FrameSample] = []
            for idx, frame_path in enumerate(sorted(Path(tmp_dir).glob("frame_*.png"))):
                frame_bgr = cv2.imread(str(frame_path))
                if frame_bgr is None:
                    continue
                frames.append(
                    FrameSample(
                        frame_index=len(frames),
                        timestamp_s=round(idx * interval_s, 2),
                        frame_bgr=frame_bgr,
                    )
                )

            if not frames:
                return [], "ffmpeg completed, but no frames were extracted."

            return frames, None
    except subprocess.TimeoutExpired:
        return [], f"ffmpeg timed out after {timeout}s"
    except Exception as e:
        return [], str(e)


def sample_frames_from_url(
    yt_url: str,
    interval_s: int = 10,
    max_frames: int = 6,
    progress_callback=None,
) -> tuple[list[FrameSample], Optional[str]]:
    """
    High-level convenience function: yt-dlp → FrameSampler → frames.

    Returns:
        (frames_list, error_message)
    """
    stream_url, err = get_stream_url(yt_url)
    if err:
        return [], err

    try:
        sampler = FrameSampler(
            stream_url=stream_url,
            interval_s=interval_s,
            max_frames=max_frames,
        )
        frames = sampler.sample(progress_callback=progress_callback)
        if not frames:
            ffmpeg_frames, ffmpeg_err = _sample_frames_with_ffmpeg(
                stream_url=stream_url,
                interval_s=interval_s,
                max_frames=max_frames,
            )
            if ffmpeg_frames:
                logger.info("OpenCV returned no frames; recovered with ffmpeg fallback.")
                return ffmpeg_frames, None
            return [], ffmpeg_err or "No frames could be extracted from the stream."
        return frames, None
    except Exception as e:
        logger.warning("OpenCV stream sampling failed, trying ffmpeg fallback: %s", e)
        ffmpeg_frames, ffmpeg_err = _sample_frames_with_ffmpeg(
            stream_url=stream_url,
            interval_s=interval_s,
            max_frames=max_frames,
        )
        if ffmpeg_frames:
            return ffmpeg_frames, None

        if ffmpeg_err:
            return [], f"{e} | ffmpeg fallback failed: {ffmpeg_err}"
        return [], str(e)
