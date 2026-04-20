"""
cv_analysis.py
─────────────────────────────────────────────────────────────────
Lightweight Computer Vision analysis on already-extracted frames.
Runs entirely on CPU with OpenCV + NumPy — zero API calls,
negligible runtime (~0.05 s per frame pair).

Extracts:
  • Dense optical flow (Farneback) → motion magnitude, direction histogram
  • Spatial activity heatmap (accumulated frame delta)
  • Edge density (Canny) → infrastructure complexity
  • Per-frame motion energy timeline
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
import base64
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────

@dataclass
class CVAnalysisResult:
    """Computer-vision metrics derived from the sampled frame set."""

    video_label: str = ""
    frame_count: int = 0

    # ── Optical flow ──────────────────────────────────────────────
    mean_motion_magnitude: float = 0.0
    max_motion_magnitude:  float = 0.0
    p85_motion_magnitude:  float = 0.0
    motion_direction_hist: list[float] = field(default_factory=lambda: [0.0]*8)
    frame_motion_energy:   list[float] = field(default_factory=list)

    # ── Spatial activity ──────────────────────────────────────────
    activity_heatmap_b64:  Optional[str] = None
    heatmap_shape:         tuple[int,int] = (0, 0)
    _activity_array:       Optional[np.ndarray] = field(default=None, repr=False)

    # ── Edge density ──────────────────────────────────────────────
    mean_edge_density:        float = 0.0
    edge_density_per_frame:   list[float] = field(default_factory=list)

    # ── Scene statistics ─────────────────────────────────────────
    mean_brightness: float = 0.0
    brightness_std:  float = 0.0
    contrast_rms:    float = 0.0

    # ── Rendered visualizations (base64 PNG) ─────────────────────
    # HSV optical flow overlay on first frame
    flow_viz_b64:     list[str] = field(default_factory=list)  # one per frame pair
    # Neon edge overlay on each frame
    edge_viz_b64:     list[str] = field(default_factory=list)  # one per frame
    # Frame-difference motion detection
    diff_viz_b64:     list[str] = field(default_factory=list)  # one per frame pair
    # Activity heatmap alpha-blended onto first frame
    heatmap_blend_b64: Optional[str] = None
    # First raw frame (for reference display)
    first_frame_b64:   Optional[str] = None


# ─────────────────────────────────────────────────────────────────
# Core analysis functions
# ─────────────────────────────────────────────────────────────────

def _to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def _compute_optical_flow(gray1: np.ndarray, gray2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense Farneback optical flow — returns (magnitude, angle_deg)."""
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    return mag, ang


def _direction_histogram(magnitudes: np.ndarray, angles: np.ndarray, bins: int = 8) -> list[float]:
    """
    8-bin direction histogram weighted by motion magnitude.
    Bins: N(0), NE(45), E(90), SE(135), S(180), SW(225), W(270), NW(315)
    """
    hist = np.zeros(bins, dtype=np.float64)
    bin_width = 360.0 / bins
    idx = ((angles % 360) / bin_width).astype(int) % bins
    np.add.at(hist, idx, magnitudes)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist.tolist()


def _activity_heatmap(grays: list[np.ndarray], target_h: int = 120, target_w: int = 213) -> np.ndarray:
    """
    Accumulated absolute frame-delta heatmap, resized to (H, W).
    Returns float32 array normalised 0–1.
    """
    if len(grays) < 2:
        return np.zeros((target_h, target_w), dtype=np.float32)

    h, w = grays[0].shape
    accum = np.zeros((h, w), dtype=np.float64)
    for g1, g2 in zip(grays[:-1], grays[1:]):
        diff = cv2.absdiff(g1, g2).astype(np.float64)
        accum += diff

    # Apply Gaussian blur to smooth
    accum = cv2.GaussianBlur(accum.astype(np.float32), (21, 21), 0)
    # Normalise
    max_val = accum.max()
    if max_val > 0:
        accum = accum / max_val
    # Resize
    resized = cv2.resize(accum, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def _heatmap_to_b64_png(heatmap: np.ndarray) -> str:
    """Convert float32 HxW heatmap to a RGBA PNG base64 for display."""
    uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
    # Apply a 'inferno'-style colormap via OpenCV
    colored = cv2.applyColorMap(uint8, cv2.COLORMAP_INFERNO)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(colored_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels that are Canny edges (0–1)."""
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    return float(np.count_nonzero(edges)) / gray.size


def _scene_stats(gray: np.ndarray) -> tuple[float, float, float]:
    """Returns (mean_brightness, std_brightness, rms_contrast)."""
    f = gray.astype(np.float32)
    mean = float(f.mean())
    std  = float(f.std())
    rms  = float(np.sqrt(np.mean((f - mean) ** 2)))
    return mean, std, rms


def _ndarray_to_b64(arr_rgb: np.ndarray, max_w: int = 640) -> str:
    """Resize and encode an RGB uint8 array as base64 PNG."""
    h, w = arr_rgb.shape[:2]
    if w > max_w:
        scale = max_w / w
        arr_rgb = cv2.resize(arr_rgb, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(arr_rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def render_flow_hsv(frame_bgr: np.ndarray, mag: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """HSV optical flow visualization — hue=direction, saturation=magnitude."""
    h, w = mag.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)   # hue = direction (0-180)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hsv[..., 1] = mag_norm                       # saturation = magnitude
    hsv[..., 2] = np.where(mag_norm > 8, 255, 60).astype(np.uint8)  # dim low-motion
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    # Alpha-blend onto original frame
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    blend = cv2.addWeighted(frame_rgb, 0.35, flow_rgb, 0.65, 0)
    return blend


def render_edge_overlay(frame_bgr: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Neon cyan Canny edges on a darkened version of the frame."""
    edges = cv2.Canny(gray, 40, 130)
    # Dilate edges slightly for visibility
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, kernel, iterations=1)
    dark = (cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) * 0.28).astype(np.uint8)
    overlay = dark.copy()
    overlay[edges > 0] = [99, 179, 237]   # steel blue (#63b3ed)
    return overlay


def render_frame_diff(frame1_bgr: np.ndarray, frame2_bgr: np.ndarray) -> np.ndarray:
    """Colorized motion-detection frame difference (INFERNO hotspots on dark frame)."""
    diff = cv2.absdiff(frame1_bgr, frame2_bgr)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Amplify + smooth
    gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    _, thresh = cv2.threshold(gray_diff, 15, 255, cv2.THRESH_TOZERO)
    amplified = cv2.convertScaleAbs(thresh, alpha=4)
    colored = cv2.applyColorMap(amplified, cv2.COLORMAP_INFERNO)
    dark = (frame1_bgr.astype(np.float32) * 0.25).astype(np.uint8)
    blend = cv2.addWeighted(dark, 0.55, colored, 0.45, 0)
    return cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)


def render_heatmap_blend(frame_bgr: np.ndarray, heatmap_norm: np.ndarray) -> np.ndarray:
    """Activity heatmap (INFERNO) alpha-blended onto frame."""
    h, w = frame_bgr.shape[:2]
    hm = cv2.resize(heatmap_norm, (w, h))
    uint8 = (hm * 255).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(uint8, cv2.COLORMAP_INFERNO)
    # Only apply heatmap where activity is significant (> 10%)
    mask = (hm > 0.10).astype(np.float32)
    alpha = (hm * mask * 0.7).clip(0, 0.7)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = (frame_rgb.astype(np.float32) * (1 - alpha[..., None]) +
               colored_rgb.astype(np.float32) * alpha[..., None]).clip(0, 255).astype(np.uint8)
    return blended

# ─────────────────────────────────────────────────────────────────
# High-level entry point
# ─────────────────────────────────────────────────────────────────

def analyze_frames_cv(frames, video_label: str = "") -> CVAnalysisResult:
    """
    Run all CV analyses on a list of FrameSample objects.
    Expects frames with .frame_bgr attribute (np.ndarray BGR).

    Designed to run in < 0.1 s for up to 6 frames on modern hardware.
    """
    result = CVAnalysisResult(
        video_label=video_label,
        frame_count=len(frames),
    )
    if not frames:
        return result

    grays = [_to_gray(f.frame_bgr) for f in frames]

    # ── Optical flow ──────────────────────────────────────────────
    all_magnitudes: list[np.ndarray] = []
    all_angles:     list[np.ndarray] = []
    dir_accum = np.zeros(8, dtype=np.float64)
    energy_per_pair: list[float] = []

    for g1, g2 in zip(grays[:-1], grays[1:]):
        mag, ang = _compute_optical_flow(g1, g2)
        all_magnitudes.append(mag)
        all_angles.append(ang)
        energy_per_pair.append(float(mag.mean()))
        hist = _direction_histogram(mag, ang)
        dir_accum += np.array(hist)

    if all_magnitudes:
        stacked = np.concatenate([m.ravel() for m in all_magnitudes])
        result.mean_motion_magnitude = float(stacked.mean())
        result.max_motion_magnitude  = float(stacked.max())
        result.p85_motion_magnitude  = float(np.percentile(stacked, 85))
        total = dir_accum.sum()
        result.motion_direction_hist = (dir_accum / total).tolist() if total > 0 else [0.0]*8
        result.frame_motion_energy   = energy_per_pair

    # ── Spatial activity heatmap ──────────────────────────────────
    hmap = _activity_heatmap(grays)
    result._activity_array      = hmap
    result.heatmap_shape        = (hmap.shape[0], hmap.shape[1])
    result.activity_heatmap_b64 = _heatmap_to_b64_png(hmap)

    # ── Edge density ──────────────────────────────────────────────
    densities = [_edge_density(g) for g in grays]
    result.mean_edge_density      = float(np.mean(densities))
    result.edge_density_per_frame = densities

    # ── Scene statistics ──────────────────────────────────────────
    scene_stats = [_scene_stats(g) for g in grays]
    result.mean_brightness = float(np.mean([s[0] for s in scene_stats]))
    result.brightness_std  = float(np.mean([s[1] for s in scene_stats]))
    result.contrast_rms    = float(np.mean([s[2] for s in scene_stats]))

    # ── Rendered visualizations ───────────────────────────────────
    bgr_frames = [f.frame_bgr for f in frames]

    # First raw frame
    if bgr_frames:
        result.first_frame_b64 = _ndarray_to_b64(
            cv2.cvtColor(bgr_frames[0], cv2.COLOR_BGR2RGB))

    # HSV optical flow (one per pair)
    for bgr, mag, ang in zip(bgr_frames[:-1], all_magnitudes, all_angles):
        rendered = render_flow_hsv(bgr, mag, ang)
        result.flow_viz_b64.append(_ndarray_to_b64(rendered))

    # Neon edge overlay (one per frame)
    for bgr, gray in zip(bgr_frames, grays):
        rendered = render_edge_overlay(bgr, gray)
        result.edge_viz_b64.append(_ndarray_to_b64(rendered))

    # Frame difference (one per pair)
    for bgr1, bgr2 in zip(bgr_frames[:-1], bgr_frames[1:]):
        rendered = render_frame_diff(bgr1, bgr2)
        result.diff_viz_b64.append(_ndarray_to_b64(rendered))

    # Heatmap blended on first frame
    if bgr_frames and result._activity_array is not None:
        rendered = render_heatmap_blend(bgr_frames[0], result._activity_array)
        result.heatmap_blend_b64 = _ndarray_to_b64(rendered)

    return result

