"""
app.py — VLM Traffic Safety Analyzer
Research-grade Streamlit application for VLM-powered traffic safety analysis.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="VLM Traffic Safety Analyzer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load custom CSS ───────────────────────────────────────────────────────────
_CSS_PATH = Path(__file__).parent / "assets" / "style.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── Module imports ────────────────────────────────────────────────────────────
from modules.ollama_utils import check_ollama_running, list_ollama_models, is_vision_model
from modules.prompt_library import get_system_prompt, get_user_prompt, list_facility_types, FACILITY_TYPES
from modules.safety_schema import parse_vlm_response, SafetyAnalysisResult
from modules.video_engine import sample_frames_from_url, get_video_metadata
from modules.vlm_dispatcher import create_adapter
from modules.speed_limit_lookup import resolve_speed_limit
from modules.cv_analysis import analyze_frames_cv
import modules.report_builder as rb

# ── Default video feeds ───────────────────────────────────────────────────────
DEFAULT_FEEDS = [
    {"label": "Highway Merge (I-xx)",        "url": "https://www.youtube.com/watch?v=3nx8AA35sos", "facility": "highway_merge"},
    {"label": "Exit Ramp",                    "url": "https://www.youtube.com/watch?v=nTDu6wSasGk", "facility": "exit_ramp"},
    {"label": "Barrier + Crash Cushion",      "url": "https://www.youtube.com/watch?v=SETJ79HmwI0", "facility": "barrier_cushion"},
    {"label": "Lane Direction Control",       "url": "https://www.youtube.com/watch?v=DnUFAShZKus", "facility": "lane_direction"},
    {"label": "Roundabout",                   "url": "https://www.youtube.com/watch?v=AShvF9ILGkc", "facility": "roundabout"},
]

# ── Session state initialization ──────────────────────────────────────────────
def _new_feed_id() -> str:
    return f"feed_{uuid.uuid4().hex[:8]}"


def _normalize_feeds(feeds: list[dict]) -> list[dict]:
    normalized = []
    seen_ids = set()
    for feed in feeds:
        feed_id = feed.get("id") or _new_feed_id()
        while feed_id in seen_ids:
            feed_id = _new_feed_id()
        seen_ids.add(feed_id)
        normalized.append(
            {
                "id": feed_id,
                "label": feed["label"],
                "url": feed["url"],
                "facility": feed["facility"],
            }
        )
    return normalized


def _clean_gemini_key(value: str | None) -> str:
    if not value:
        return ""
    cleaned = value.strip()
    return "" if cleaned == "your_gemini_api_key_here" else cleaned


def _init_state():
    defaults = {
        "session_id":       str(uuid.uuid4())[:8],
        "results":          [],          # list[SafetyAnalysisResult]
        "feeds":            _normalize_feeds([dict(f) for f in DEFAULT_FEEDS]),
        "stream_log":       "",
        "running":          False,
        "gemini_key":       _clean_gemini_key(os.getenv("GEMINI_API_KEY", "")),
        "openrouter_key":   os.getenv("OPENROUTER_API_KEY", ""),
        "ollama_host":      os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "provider":         "gemini",
        "gemini_model":     "gemini-flash-latest",
        "openrouter_model": "google/gemma-4-26b-a4b-it:free",
        "ollama_model":     "",
        "frame_interval":   10,
        "max_frames":       5,
        "posted_speed":     55.0,
        "extra_context":    "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
st.session_state["feeds"] = _normalize_feeds(st.session_state["feeds"])

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
      <div style="font-size:2.2rem;">🚦</div>
      <div style="font-family:'Outfit',sans-serif; font-size:1.1rem; font-weight:700;
                  background:linear-gradient(135deg,#00d4ff,#00b4a0);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        VLM Traffic Safety
      </div>
      <div style="font-size:0.72rem; color:#4a6080; letter-spacing:0.08em;">
        RESEARCH ANALYZER v1.0
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Provider ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Model Provider</div>', unsafe_allow_html=True)

    provider = st.selectbox(
        "Provider",
        options=["gemini", "openrouter", "ollama"],
        format_func=lambda x: {"gemini": "Google Gemini", "openrouter": "OpenRouter", "ollama": "Ollama (Local)"}[x],
        index=["gemini", "openrouter", "ollama"].index(st.session_state["provider"]),
        label_visibility="collapsed",
        key="provider_select",
    )
    st.session_state["provider"] = provider

    # ── API Keys & Model ──────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">API Keys & Model</div>', unsafe_allow_html=True)

    if provider == "gemini":
        st.session_state["gemini_key"] = st.text_input(
            "Gemini API Key", value=st.session_state["gemini_key"],
            type="password", placeholder="AIza...", key="gk_input"
        )
        st.session_state["gemini_model"] = st.text_input(
            "Model", value=st.session_state["gemini_model"],
            placeholder="gemini-flash-latest", key="gm_input"
        )
        st.caption("ℹ️ Models: `gemini-flash-latest`, `gemma-4-26b-a4b-it`")

    elif provider == "openrouter":
        st.session_state["openrouter_key"] = st.text_input(
            "OpenRouter API Key", value=st.session_state["openrouter_key"],
            type="password", placeholder="sk-or-...", key="ork_input"
        )
        st.session_state["openrouter_model"] = st.text_input(
            "Model", value=st.session_state["openrouter_model"],
            placeholder="google/gemini-flash-1.5", key="orm_input"
        )
        st.caption("ℹ️ Any vision model from openrouter.ai/models")

    elif provider == "ollama":
        ollama_host = st.text_input(
            "Ollama Host", value=st.session_state["ollama_host"],
            placeholder="http://localhost:11434", key="oh_input"
        )
        st.session_state["ollama_host"] = ollama_host

        ollama_running = check_ollama_running(ollama_host)
        if ollama_running:
            st.success("✅ Ollama is running", icon="🟢")
            local_models = list_ollama_models(ollama_host)
            if local_models:
                vision_models = [m for m in local_models if is_vision_model(m)]
                model_options = vision_models if vision_models else local_models
                default_idx = 0
                if st.session_state["ollama_model"] in model_options:
                    default_idx = model_options.index(st.session_state["ollama_model"])
                chosen = st.selectbox("Local Models", model_options, index=default_idx, key="om_select")
                st.session_state["ollama_model"] = chosen
            else:
                st.warning("No models found. Pull one first.")
        else:
            st.error("❌ Ollama not reachable", icon="🔴")
            st.caption(f"Expected at: {ollama_host}")

        custom_model = st.text_input(
            "Or type any model name", value="", placeholder="gemma4:31b-cloud", key="om_custom"
        )
        if custom_model.strip():
            st.session_state["ollama_model"] = custom_model.strip()

    # ── Sampling Parameters ───────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Sampling Parameters</div>', unsafe_allow_html=True)

    st.session_state["frame_interval"] = st.slider(
        "Frame Interval (s)", min_value=5, max_value=60,
        value=st.session_state["frame_interval"], step=5,
        help="Time between sampled frames. 10 s = 6 frames/min.",
    )
    st.session_state["max_frames"] = st.slider(
        "Max Frames per Video", min_value=1, max_value=12,
        value=st.session_state["max_frames"], step=1,
        help="Fewer frames = faster + cheaper. More = richer analysis.",
    )

    # ── Context ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Additional Context</div>', unsafe_allow_html=True)
    st.session_state["extra_context"] = st.text_area(
        "Extra context for VLM", value=st.session_state["extra_context"],
        height=80, placeholder="e.g. 'Night-time, wet road, construction zone nearby'",
        label_visibility="collapsed",
    )

    # ── Speed override ────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Speed Limit Override</div>', unsafe_allow_html=True)
    st.caption("Leave 0 to auto-detect from video signs or OpenStreetMap.")
    override_val = st.number_input(
        "Manual speed limit (mph)", min_value=0.0, max_value=85.0,
        value=float(st.session_state.get("posted_speed") or 0.0), step=5.0,
        label_visibility="collapsed",
    )
    st.session_state["posted_speed"] = float(override_val) if (override_val or 0) > 0 else None

    st.divider()
    st.caption(f"Session: `{st.session_state['session_id']}`")
    if st.button("🗑️ Clear All Results", use_container_width=True):
        st.session_state["results"] = []
        st.session_state["stream_log"] = ""
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 0.5rem 0 1.5rem 0;">
  <h1>VLM Traffic Safety Analyzer</h1>
  <p style="color:#8fa3bf; font-size:0.95rem; margin-top:-0.5rem;">
    Vision-Language Model · FHWA / HSM / HCM / MUTCD · Research Platform
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_analyze, tab_dashboard, tab_report = st.tabs([
    "🎬  Analyzer",
    "📊  Dashboard",
    "📄  Report & Export",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

with tab_analyze:
    # ── Video Feed Management ─────────────────────────────────────────────────
    st.markdown("### Video Feeds")

    facility_options = list_facility_types()
    fac_keys   = [k for k, _ in facility_options]
    fac_labels = [v for _, v in facility_options]

    # Display and manage feeds
    feeds = st.session_state["feeds"]
    cols_header = st.columns([3, 4, 2, 0.8])
    cols_header[0].markdown("**Label**")
    cols_header[1].markdown("**YouTube URL**")
    cols_header[2].markdown("**Facility Type**")
    cols_header[3].markdown("**Del**")

    feeds_to_keep = []
    delete_clicked = False
    for i, feed in enumerate(feeds):
        feed_id = feed["id"]
        c1, c2, c3, c4 = st.columns([3, 4, 2, 0.8])
        new_label = c1.text_input(f"Label_{i}", value=feed["label"],
                                   label_visibility="collapsed", key=f"fl_{feed_id}")
        new_url   = c2.text_input(f"URL_{i}", value=feed["url"],
                                   label_visibility="collapsed", key=f"fu_{feed_id}")
        fac_idx = fac_keys.index(feed["facility"]) if feed["facility"] in fac_keys else 0
        new_fac = c3.selectbox(f"Fac_{i}", options=fac_keys,
                                format_func=lambda k: FACILITY_TYPES.get(k, k),
                                index=fac_idx, label_visibility="collapsed", key=f"ff_{feed_id}")
        delete = c4.button("✕", key=f"del_{i}", help="Remove this feed")
        if not delete:
            feeds_to_keep.append({"id": feed_id, "label": new_label, "url": new_url, "facility": new_fac})
        else:
            delete_clicked = True

    if feeds_to_keep != feeds:
        st.session_state["feeds"] = feeds_to_keep
        if delete_clicked:
            st.rerun()

    # Add new feed
    with st.expander("➕ Add New Feed"):
        nc1, nc2, nc3, nc4 = st.columns([3, 4, 2, 1])
        new_label_in = nc1.text_input("Label", placeholder="My Camera", key="new_label")
        new_url_in   = nc2.text_input("URL", placeholder="https://youtube.com/watch?v=...", key="new_url")
        new_fac_in   = nc3.selectbox("Facility", fac_keys, format_func=lambda k: FACILITY_TYPES.get(k, k), key="new_fac")
        if nc4.button("Add", use_container_width=True):
            if new_url_in.strip():
                st.session_state["feeds"].append({
                    "id": _new_feed_id(),
                    "label": new_label_in or new_url_in[:30],
                    "url": new_url_in.strip(),
                    "facility": new_fac_in,
                })
                st.rerun()

    st.divider()

    # ── Run Analysis ──────────────────────────────────────────────────────────
    st.markdown("### Run Analysis")

    # Feed selector
    feed_labels = [f["label"] for f in st.session_state["feeds"]]
    selected_feeds_labels = st.multiselect(
        "Select feeds to analyze",
        options=feed_labels,
        default=feed_labels[:2] if len(feed_labels) >= 2 else feed_labels,
    )
    selected_feeds = [f for f in st.session_state["feeds"] if f["label"] in selected_feeds_labels]

    run_col, status_col = st.columns([2, 3])
    run_btn = run_col.button(
        "▶ Run VLM Analysis",
        disabled=st.session_state["running"] or not selected_feeds,
        use_container_width=True,
        type="primary",
    )

    if run_btn and selected_feeds:
        # Validate provider config
        provider = st.session_state["provider"]
        api_key = (st.session_state["gemini_key"] if provider == "gemini"
                   else st.session_state["openrouter_key"] if provider == "openrouter"
                   else "")
        model = (st.session_state["gemini_model"] if provider == "gemini"
                 else st.session_state["openrouter_model"] if provider == "openrouter"
                 else st.session_state["ollama_model"])

        if provider in ("gemini", "openrouter") and not api_key:
            st.error(f"⚠️ Please enter your {provider.title()} API key in the sidebar.")
        elif not model:
            st.error("⚠️ Please specify a model name.")
        else:
            st.session_state["running"] = True
            st.session_state["stream_log"] = ""

            # Progress area
            progress_bar = st.progress(0, text="Starting…")
            stream_box   = st.empty()
            frame_gallery = st.empty()

            try:
                adapter = create_adapter(
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    ollama_host=st.session_state["ollama_host"],
                )

                total_feeds = len(selected_feeds)
                for feed_idx, feed in enumerate(selected_feeds):
                    progress_bar.progress(
                        feed_idx / total_feeds,
                        text=f"📡 Extracting frames: {feed['label']}…"
                    )

                    # Extract frames
                    frames, err = sample_frames_from_url(
                        yt_url=feed["url"],
                        interval_s=st.session_state["frame_interval"],
                        max_frames=st.session_state["max_frames"],
                    )

                    if err:
                        st.warning(f"⚠️ {feed['label']}: {err}")
                        continue

                    if not frames:
                        st.warning(f"⚠️ {feed['label']}: No frames extracted.")
                        continue

                    # Show frame thumbnails
                    with frame_gallery.container():
                        st.markdown(f"**Frames from: {feed['label']}**")
                        thumb_cols = st.columns(min(len(frames), 6))
                        for fi, (fc, frame) in enumerate(zip(thumb_cols, frames)):
                            fc.image(frame.thumbnail, caption=f"t={frame.timestamp_s:.0f}s", use_container_width=True)

                    # ── CV analysis (fast, no API call) ──────────────────────
                    cv_result = analyze_frames_cv(frames, video_label=feed["label"])

                    # Build prompts
                    sys_prompt  = get_system_prompt(feed["facility"])
                    user_prompt = get_user_prompt(
                        facility_type=feed["facility"],
                        num_frames=len(frames),
                        frame_interval_s=st.session_state["frame_interval"],
                        video_label=feed["label"],
                        extra_context=st.session_state["extra_context"],
                    )

                    # Stream VLM response
                    progress_bar.progress(
                        (feed_idx + 0.5) / total_feeds,
                        text=f"🤖 Analyzing with {model}…"
                    )
                    _buf = [""]   # mutable container — works without nonlocal

                    def _stream_cb(chunk: str, _b=_buf):
                        _b[0] += chunk
                        stream_box.markdown(
                            f"<div class='stream-box'>{_b[0][-3000:]}</div>",
                            unsafe_allow_html=True,
                        )

                    raw_response = adapter.analyze_frames(
                        frames=frames,
                        system_prompt=sys_prompt,
                        user_prompt=user_prompt,
                        stream_callback=_stream_cb,
                    )

                    # Parse result
                    result = parse_vlm_response(
                        raw=raw_response,
                        session_id=st.session_state["session_id"],
                        video_label=feed["label"],
                        video_url=feed["url"],
                        frame_timestamp_s=frames[0].timestamp_s,
                        facility_type=feed["facility"],
                        model_used=model,
                        provider=provider,
                    )
                    result.cv_result = cv_result   # attach CV analysis
                    st.session_state["results"].append(result)

                    # ── Speed limit resolution (OSM fallback) ────────────────
                    if result.posted_speed_limit_mph is None:
                        manual_override = st.session_state.get("posted_speed")  # None or float
                        resolved, source = resolve_speed_limit(
                            vlm_detected=result.posted_speed_limit_mph,
                            location_hint=result.detected_location,
                            manual_override=manual_override,
                        )
                        result.posted_speed_limit_mph = resolved
                        result.speed_limit_source = source

                    # Show speed limit badge
                    if result.posted_speed_limit_mph:
                        src_emoji = {"vlm_sign": "🪧", "osm": "🗺️", "manual": "✏️"}.get(
                            result.speed_limit_source, "❓")
                        st.info(
                            f"{src_emoji} Speed limit detected: **{result.posted_speed_limit_mph:.0f} mph** "
                            f"(source: `{result.speed_limit_source}`)"
                            + (f" · location: *{result.detected_location}*" if result.detected_location else ""),
                            icon=None,
                        )

                    if not result.parse_success:
                        st.warning(f"⚠️ JSON parse issue for {feed['label']}. Raw text saved.")

                progress_bar.progress(1.0, text="✅ Analysis complete!")
                time.sleep(0.5)
                progress_bar.empty()
                st.success(f"✅ Analyzed {len(selected_feeds)} feed(s). See Dashboard & Report tabs.")

            except Exception as e:
                st.error(f"❌ Error: {e}")
                logger.exception("Analysis error")
            finally:
                st.session_state["running"] = False

    # ── Live Status Summary ───────────────────────────────────────────────────
    results = st.session_state["results"]
    if results:
        st.divider()
        stats = rb.compute_summary_stats(results)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Analyses Run",    stats.get("total_analyses", 0))
        m2.metric("Avg Safety Score", f"{stats.get('avg_safety_score', 0):.1f}")
        m3.metric("Total Conflicts",  stats.get("total_conflicts", 0))
        m4.metric("Avg Speed",        f"{stats.get('avg_speed_mph', 0):.1f} mph")
        m5.metric("85th %ile Speed",  f"{stats.get('p85_speed_mph', 0):.1f} mph")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

with tab_dashboard:
    results = st.session_state["results"]

    if not results:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#4a5568;">
            <div style="font-size:3rem;margin-bottom:1rem;">📡</div>
            <h3 style="color:#63b3ed;font-weight:600;">No Analysis Data Yet</h3>
            <p style="color:#718096;max-width:420px;margin:0 auto;">
                Run an analysis in the <strong>Analyzer</strong> tab to populate
                the research dashboard with CV and safety metrics.
            </p>
        </div>""", unsafe_allow_html=True)
    else:
        stats = rb.compute_summary_stats(results)

        # ── KPI Banner ────────────────────────────────────────────────────────
        st.markdown("""
        <div style="background:linear-gradient(135deg,#0b1628 0%,#0f1e35 100%);
                    border:1px solid rgba(99,179,237,0.15);border-radius:12px;
                    padding:1.4rem 1.8rem;margin-bottom:1.8rem;">
            <p style="color:#4a5568;font-size:0.75rem;letter-spacing:0.12em;
                      text-transform:uppercase;margin:0 0 0.8rem 0;">
                ◈ Research Session Overview
            </p>
        </div>""", unsafe_allow_html=True)

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        avg_s = stats["avg_safety_score"]
        score_color = "#68d391" if avg_s >= 80 else "#f6ad55" if avg_s >= 60 else "#fc8181"
        k1.metric("Frames Analyzed",   stats["total_analyses"])
        k2.metric("Videos",            stats["videos_analyzed"])
        k3.metric("Safety Score (μ)",  f"{avg_s:.1f}",
                  help="Mean composite safety score (0–100)")
        k4.metric("Safety Score (min)", f"{stats['min_safety_score']:.1f}",
                  delta=f"{stats['min_safety_score']-avg_s:.1f}", delta_color="inverse")
        k5.metric("Total Conflicts",   stats["total_conflicts"])
        k6.metric("Conflict Rate",     f"{stats['avg_conflict_rate']:.2f}",
                  help="Conflicts per 100 observed vehicles")

        # ── Section: Computer Vision Intelligence ────────────────────────────
        st.markdown("""
        <div style="margin:2.2rem 0 1rem 0;border-left:3px solid #b794f4;padding-left:1rem;">
            <h3 style="color:#b794f4;margin:0;font-size:1.1rem;letter-spacing:0.04em;">
                ◈ Computer Vision Intelligence
            </h3>
            <p style="color:#4a5568;font-size:0.82rem;margin:0.2rem 0 0 0;">
                Optical flow · Spatial activity · Edge density — derived from raw frames, zero API cost
            </p>
        </div>""", unsafe_allow_html=True)

        cv_col1, cv_col2 = st.columns([1, 1], gap="large")
        with cv_col1:
            st.plotly_chart(rb.chart_motion_polar(results),
                            use_container_width=True, config={"displayModeBar": False})
        with cv_col2:
            st.plotly_chart(rb.chart_motion_energy(results),
                            use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        cv_col3, cv_col4 = st.columns([1, 1], gap="large")
        with cv_col3:
            st.plotly_chart(rb.chart_activity_heatmap(results),
                            use_container_width=True, config={"displayModeBar": False})
        with cv_col4:
            st.plotly_chart(rb.chart_edge_density(results),
                            use_container_width=True, config={"displayModeBar": False})

        # ── Section: Safety Analytics ─────────────────────────────────────────
        st.markdown("""
        <div style="margin:2.4rem 0 1rem 0;border-left:3px solid #63b3ed;padding-left:1rem;">
            <h3 style="color:#63b3ed;margin:0;font-size:1.1rem;letter-spacing:0.04em;">
                ◈ Safety Analytics
            </h3>
            <p style="color:#4a5568;font-size:0.82rem;margin:0.2rem 0 0 0;">
                VLM-extracted surrogate safety measures · HSM / FHWA / MUTCD framework
            </p>
        </div>""", unsafe_allow_html=True)

        sa_col1, sa_col2 = st.columns([3, 2], gap="large")
        with sa_col1:
            st.plotly_chart(rb.chart_safety_score_timeline(results),
                            use_container_width=True, config={"displayModeBar": True})
        with sa_col2:
            st.plotly_chart(rb.chart_conflict_distribution(results),
                            use_container_width=True, config={"displayModeBar": True})

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        sa_col3, sa_col4 = st.columns([1, 1], gap="large")
        with sa_col3:
            st.plotly_chart(rb.chart_speed_histogram(results),
                            use_container_width=True, config={"displayModeBar": True})
        with sa_col4:
            st.plotly_chart(rb.chart_surrogate_scatter(results),
                            use_container_width=True, config={"displayModeBar": True})

        # ── Section: CV Image Intelligence ───────────────────────────────────
        st.markdown("""
        <div style="margin:2.4rem 0 1rem 0;border-left:3px solid #4fd1c5;padding-left:1rem;">
            <h3 style="color:#4fd1c5;margin:0;font-size:1.1rem;letter-spacing:0.04em;">
                ◈ CV Image Intelligence
            </h3>
            <p style="color:#4a5568;font-size:0.82rem;margin:0.2rem 0 0 0;">
                Processed frame visualizations · Optical flow · Edge detection · Motion detection · Heatmap overlay
            </p>
        </div>""", unsafe_allow_html=True)

        # Facility selector — collect results that have cv_result
        cv_results_map = {r.video_label: r.cv_result
                          for r in results if getattr(r, "cv_result", None) is not None}
        if not cv_results_map:
            st.info("Run an analysis to generate computer vision image outputs.", icon="🔬")
        else:
            selected_label = st.selectbox(
                "Select facility / video",
                options=list(cv_results_map.keys()),
                key="cv_gallery_select",
                label_visibility="collapsed",
            )
            cv = cv_results_map[selected_label]

            # Four processed-image panels
            img_col1, img_col2 = st.columns(2, gap="medium")

            with img_col1:
                st.markdown("""<p style="color:#b794f4;font-size:0.82rem;font-weight:600;
                    letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.3rem;">
                    🌈 Optical Flow (HSV) — Frame 1→2</p>""", unsafe_allow_html=True)
                if cv.flow_viz_b64:
                    st.image(f"data:image/png;base64,{cv.flow_viz_b64[0]}",
                             use_container_width=True,
                             caption="Hue = direction · Saturation = speed · Farneback dense flow")
                else:
                    st.caption("_Not enough frames for optical flow_")

                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                st.markdown("""<p style="color:#f6ad55;font-size:0.82rem;font-weight:600;
                    letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.3rem;">
                    🔥 Motion Detection — Frame Δ</p>""", unsafe_allow_html=True)
                if cv.diff_viz_b64:
                    st.image(f"data:image/png;base64,{cv.diff_viz_b64[0]}",
                             use_container_width=True,
                             caption="INFERNO hotspots = inter-frame change · thresholded & amplified")
                else:
                    st.caption("_Not enough frames for diff_")

            with img_col2:
                st.markdown("""<p style="color:#63b3ed;font-size:0.82rem;font-weight:600;
                    letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.3rem;">
                    🔵 Edge Detection — Infrastructure Lines</p>""", unsafe_allow_html=True)
                if cv.edge_viz_b64:
                    st.image(f"data:image/png;base64,{cv.edge_viz_b64[0]}",
                             use_container_width=True,
                             caption="Canny edge detection — road markings, barriers, signs")
                else:
                    st.caption("_No edge data_")

                st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                st.markdown("""<p style="color:#fc8181;font-size:0.82rem;font-weight:600;
                    letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.3rem;">
                    🌡 Activity Heatmap Overlay</p>""", unsafe_allow_html=True)
                if cv.heatmap_blend_b64:
                    st.image(f"data:image/png;base64,{cv.heatmap_blend_b64}",
                             use_container_width=True,
                             caption="Accumulated motion density overlaid — red = high-activity zones")
                else:
                    st.caption("_No heatmap data_")

            # Extra flow frames (if more than 1 pair)
            if len(cv.flow_viz_b64) > 1:
                with st.expander(f"📽 All {len(cv.flow_viz_b64)} Optical Flow Frames", expanded=False):
                    flow_cols = st.columns(min(len(cv.flow_viz_b64), 4))
                    for i, (fc, b64) in enumerate(zip(flow_cols, cv.flow_viz_b64)):
                        fc.image(f"data:image/png;base64,{b64}",
                                 caption=f"Pair {i+1}→{i+2}", use_container_width=True)

        # ── Raw data expander ─────────────────────────────────────────────────
        with st.expander("📋 Raw Results Table", expanded=False):
            df = rb.results_to_df(results)
            st.dataframe(df, use_container_width=True, height=300)







# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: REPORT & EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_report:
    results = st.session_state["results"]

    if not results:
        st.info("📄 Run an analysis first to generate a report.", icon="ℹ️")
    else:
        # Export buttons row
        exp1, exp2, exp3, _ = st.columns([1.5, 1.5, 1.5, 3])

        csv_bytes = rb.export_csv(results)
        exp1.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name=f"traffic_safety_{st.session_state['session_id']}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        latex_str = rb.generate_latex_table(results)
        exp2.download_button(
            "⬇️ LaTeX Table",
            data=latex_str.encode("utf-8"),
            file_name=f"safety_table_{st.session_state['session_id']}.tex",
            mime="text/plain",
            use_container_width=True,
        )

        report_md = rb.generate_executive_report(results, st.session_state["session_id"])
        exp3.download_button(
            "⬇️ Markdown Report",
            data=report_md.encode("utf-8"),
            file_name=f"safety_report_{st.session_state['session_id']}.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.divider()

        # Executive report render
        st.markdown(report_md)

        st.divider()

        # LaTeX preview
        with st.expander("📐 LaTeX Table (copy for paper)"):
            st.code(latex_str, language="latex")

        # Per-result detail
        with st.expander("🔬 Per-Frame Detail"):
            for i, r in enumerate(results):
                st.markdown(f"**Frame {i+1}** — `{r.video_label}` | t={r.frame_timestamp_s:.0f}s | "
                            f"Score: **{r.safety_score:.1f}** | Conflicts: {r.conflict_count}")
                if r.narrative_summary:
                    st.markdown(f"> {r.narrative_summary}")
                if r.recommendations:
                    for rec in r.recommendations:
                        st.markdown(f"  - {rec}")
                if r.parse_warnings:
                    st.warning(f"Parse warnings: {'; '.join(r.parse_warnings)}")
                with st.expander(f"Raw VLM response — Frame {i+1}"):
                    st.code(r.raw_response[:4000], language="json")
                st.divider()
