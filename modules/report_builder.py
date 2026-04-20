"""
report_builder.py — Advanced interactive Plotly charts + export utilities
"""
from __future__ import annotations

import io
import json
import textwrap
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Shared dark theme ────────────────────────────────────────────────────────

THEME = dict(
    bg       = "#060c18",
    bg_card  = "#0b1628",
    bg_card2 = "#0f1e35",
    grid     = "rgba(99,179,237,0.07)",
    cyan     = "#63b3ed",   # steel blue — primary
    teal     = "#4fd1c5",   # teal — secondary
    amber    = "#f6ad55",   # warm amber — warning
    rose     = "#fc8181",   # coral rose — danger
    violet   = "#b794f4",   # soft violet — special
    green    = "#68d391",   # sage green — safe
    sky      = "#90cdf4",   # pale sky — accent
    text     = "#edf2f7",
    muted    = "#4a5568",
    font     = "Inter, Space Grotesk, sans-serif",
)

SEVERITY_COLORS = {1: THEME["teal"], 2: THEME["cyan"], 3: THEME["violet"],
                   4: THEME["amber"], 5: THEME["rose"]}

CONFLICT_PALETTE = [
    THEME["cyan"], THEME["teal"], THEME["violet"], THEME["amber"],
    THEME["rose"], THEME["green"], THEME["sky"], "#f687b3", "#76e4f7", "#9ae6b4",
]

def _base_layout(**kwargs) -> dict:
    layout = dict(
        paper_bgcolor=THEME["bg_card"],
        plot_bgcolor=THEME["bg_card"],
        font=dict(family=THEME["font"], color=THEME["text"], size=12),
        margin=dict(l=60, r=40, t=64, b=56),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=THEME["grid"],
            borderwidth=1,
            font=dict(size=11),
            itemsizing="constant",
        ),
        xaxis=dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"],
                   linecolor=THEME["grid"], tickfont=dict(size=10),
                   showspikes=True, spikecolor=THEME["muted"], spikethickness=1),
        yaxis=dict(gridcolor=THEME["grid"], zerolinecolor=THEME["grid"],
                   linecolor=THEME["grid"], tickfont=dict(size=10)),
        hoverlabel=dict(bgcolor=THEME["bg_card2"], bordercolor=THEME["cyan"],
                        font=dict(family=THEME["font"], size=12, color=THEME["text"])),
    )
    layout.update(kwargs)
    return layout


def _hex_to_rgba(hex_color: str, alpha: float = 0.35) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,alpha)' for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def results_to_df(results: list) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([r.to_flat_dict() for r in results])


# ── Chart 1: Safety Score Timeline ───────────────────────────────────────────

def chart_safety_score_timeline(results: list) -> go.Figure:
    df = results_to_df(results)
    if df.empty:
        return go.Figure().update_layout(**_base_layout(title="No data yet"))

    fig = go.Figure()

    for label, grp in df.groupby("video_label"):
        grp = grp.sort_values("frame_timestamp_s")
        # Smoothed line
        fig.add_trace(go.Scatter(
            x=grp["frame_timestamp_s"],
            y=grp["safety_score"],
            mode="lines+markers",
            name=label,
            line=dict(width=2.5, shape="spline", smoothing=1.2),
            marker=dict(size=7, symbol="circle",
                        line=dict(width=1.5, color=THEME["bg"])),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "t = %{x:.0f} s<br>"
                "Safety Score: <b>%{y:.1f}</b><extra></extra>"
            ),
        ))

    # Reference bands
    for y0, y1, color, label in [
        (80, 100, "rgba(52,211,153,0.07)", "Safe (80–100)"),
        (60, 80,  "rgba(245,158,11,0.07)", "Caution (60–80)"),
        (0,  60,  "rgba(251,113,133,0.07)", "Risk (<60)"),
    ]:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0,
                      annotation_text=label,
                      annotation_position="right",
                      annotation_font=dict(size=9, color=THEME["muted"]))

    fig.update_layout(**_base_layout(
        title=dict(text="📊 Safety Score Timeline", font=dict(size=16, color=THEME["cyan"])),
        xaxis_title="Frame Timestamp (s)",
        yaxis_title="Safety Score (0–100)",
        yaxis_range=[0, 105],
        hovermode="x unified",
    ))
    return fig


# ── Chart 2: Conflict Type Distribution ──────────────────────────────────────

def chart_conflict_distribution(results: list) -> go.Figure:
    from collections import Counter
    all_conflicts = []
    for r in results:
        for ev in r.conflict_events:
            all_conflicts.append({
                "type": ev.conflict_type.replace("_", " ").title(),
                "severity": ev.severity,
                "video": r.video_label,
            })

    if not all_conflicts:
        return go.Figure().update_layout(**_base_layout(title="No conflicts detected"))

    cdf = pd.DataFrame(all_conflicts)
    counts = cdf.groupby(["type", "severity"]).size().reset_index(name="count")
    types_sorted = counts.groupby("type")["count"].sum().sort_values(ascending=True).index.tolist()

    fig = go.Figure()
    for sev in sorted(counts["severity"].unique()):
        sub = counts[counts["severity"] == sev]
        color = SEVERITY_COLORS.get(sev, THEME["violet"])
        fig.add_trace(go.Bar(
            y=sub["type"],
            x=sub["count"],
            name=f"Severity {sev}",
            orientation="h",
            marker=dict(color=color, opacity=0.85,
                        line=dict(color=THEME["bg"], width=0.5)),
            hovertemplate="<b>%{y}</b><br>Count: %{x}<br>Severity: " + str(sev) + "<extra></extra>",
        ))

    fig.update_layout(**_base_layout(
        title=dict(text="⚠️ Conflict Type Distribution by Severity", font=dict(size=16, color=THEME["amber"])),
        barmode="stack",
        xaxis_title="Conflict Count",
        yaxis=dict(categoryorder="array", categoryarray=types_sorted,
                   gridcolor=THEME["grid"], tickfont=dict(size=10)),
        showlegend=True,
    ))
    return fig


# ── Chart 3: Speed Distribution Histogram ────────────────────────────────────

def chart_speed_histogram(results: list, posted_limit_mph: float = None) -> go.Figure:
    speeds = []
    for r in results:
        for s in r.estimated_speeds_mph:
            speeds.append({"speed": s, "video": r.video_label, "facility": r.facility_type})

    if not speeds:
        return go.Figure().update_layout(**_base_layout(title="No speed data"))

    sdf = pd.DataFrame(speeds)
    fig = go.Figure()

    for i, (label, grp) in enumerate(sdf.groupby("video")):
        color = CONFLICT_PALETTE[i % len(CONFLICT_PALETTE)]
        fig.add_trace(go.Histogram(
            x=grp["speed"], name=label, nbinsx=20,
            marker=dict(color=color, opacity=0.7, line=dict(color=THEME["bg"], width=0.5)),
            hovertemplate="Speed: %{x:.1f} mph<br>Count: %{y}<extra></extra>",
        ))

    # Collect per-result speed limits with source labels
    seen: dict[float, str] = {}
    for r in results:
        if r.posted_speed_limit_mph and r.posted_speed_limit_mph > 0:
            seen[r.posted_speed_limit_mph] = r.speed_limit_source
    if not seen and posted_limit_mph and posted_limit_mph > 0:
        seen[posted_limit_mph] = "manual"

    src_emoji = {"vlm_sign": "🪧", "osm": "🗺️", "manual": "✏️"}
    for lim, src in seen.items():
        fig.add_vline(
            x=lim, line_width=2, line_dash="dash", line_color=THEME["rose"],
            annotation_text=f"Posted {lim:.0f} mph ({src_emoji.get(src, src)})",
            annotation_font=dict(color=THEME["rose"], size=11),
            annotation_position="top right",
        )

    all_speeds = sdf["speed"].dropna()
    if len(all_speeds) > 0:
        p85 = np.percentile(all_speeds, 85)
        fig.add_vline(
            x=p85, line_width=1.5, line_dash="dot", line_color=THEME["amber"],
            annotation_text=f"85th: {p85:.1f} mph",
            annotation_font=dict(color=THEME["amber"], size=11),
            annotation_position="top left",
        )

    fig.update_layout(**_base_layout(
        title=dict(text="🚗 Vehicle Speed Distribution", font=dict(size=16, color=THEME["green"])),
        barmode="overlay",
        xaxis_title="Estimated Speed (mph)",
        yaxis_title="Frequency",
        xaxis=dict(gridcolor=THEME["grid"], tickfont=dict(size=10)),
    ))
    return fig


# ── Chart 4: Vehicle Count Time Series ───────────────────────────────────────

def chart_vehicle_timeline(results: list) -> go.Figure:
    if not results:
        return go.Figure().update_layout(**_base_layout(title="No data yet"))

    records = []
    for r in results:
        vc = r.vehicle_classes
        records.append({
            "t": r.frame_timestamp_s,
            "label": r.video_label,
            "Car": vc.car, "Truck": vc.truck,
            "Motorcycle": vc.motorcycle, "Bus": vc.bus,
            "Bicycle": vc.bicycle, "Pedestrian": vc.pedestrian,
        })
    df = pd.DataFrame(records).sort_values("t")

    classes = ["Car", "Truck", "Motorcycle", "Bus", "Bicycle", "Pedestrian"]
    colors  = [THEME["cyan"], THEME["amber"], THEME["violet"],
               THEME["rose"], THEME["green"], THEME["teal"]]

    fig = go.Figure()
    for cls, color in zip(classes, colors):
        if df[cls].sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=df["t"], y=df[cls],
            name=cls,
            mode="lines",
            stackgroup="one",
            fill="tonexty",
            line=dict(width=1.5, color=color),
            fillcolor=_hex_to_rgba(color, 0.35),
            hovertemplate=f"{cls}: %{{y}}<br>t=%{{x:.0f}}s<extra></extra>",
        ))

    fig.update_layout(**_base_layout(
        title=dict(text="🚦 Vehicle Class Composition Over Time", font=dict(size=16, color=THEME["violet"])),
        xaxis_title="Frame Timestamp (s)",
        yaxis_title="Vehicle Count",
        hovermode="x unified",
    ))
    return fig


# ── Chart 5: Severity Heatmap ─────────────────────────────────────────────────

def chart_severity_heatmap(results: list) -> go.Figure:
    if not results:
        return go.Figure().update_layout(**_base_layout(title="No data yet"))

    # Pivot: rows = video_label, cols = conflict_type, values = max severity
    records = []
    for r in results:
        for ev in r.conflict_events:
            records.append({
                "video": r.video_label,
                "type": ev.conflict_type.replace("_", " ").title(),
                "severity": ev.severity,
            })

    if not records:
        return go.Figure().update_layout(**_base_layout(title="No conflict data for heatmap"))

    df = pd.DataFrame(records)
    pivot = df.groupby(["video", "type"])["severity"].max().unstack(fill_value=0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0,  THEME["bg_card"]],
            [0.2,  "#0e4f6e"],
            [0.4,  THEME["teal"]],
            [0.6,  THEME["violet"]],
            [0.8,  THEME["amber"]],
            [1.0,  THEME["rose"]],
        ],
        zmin=0, zmax=5,
        text=pivot.values,
        texttemplate="%{text}",
        textfont=dict(size=12, color=THEME["text"]),
        hovertemplate="Video: %{y}<br>Conflict: %{x}<br>Max Severity: %{z}<extra></extra>",
        colorbar=dict(
            title=dict(text="Severity", font=dict(color=THEME["text"])),
            tickvals=[0,1,2,3,4,5],
            ticktext=["0","1","2","3","4","5"],
            tickfont=dict(color=THEME["text"]),
        ),
    ))

    fig.update_layout(**_base_layout(
        title=dict(text="🔥 Conflict Severity Heatmap", font=dict(size=16, color=THEME["rose"])),
        xaxis=dict(tickangle=-30, gridcolor=THEME["grid"]),
    ))
    return fig


# ── Chart 6: Infrastructure Compliance Radar ─────────────────────────────────

def chart_compliance_radar(results: list) -> go.Figure:
    from collections import defaultdict

    standards: dict[str, list[int]] = defaultdict(list)
    for r in results:
        for obs in r.compliance_observations:
            std = obs.standard[:30] if obs.standard else "Unknown"
            score = 1 if obs.compliant else (0 if obs.compliant is False else 0.5)
            standards[std].append(score)

    if not standards:
        # fallback placeholder radar
        cats = ["Signing", "Marking", "Geometry", "Lighting", "Sight Distance", "Barrier"]
        fig = go.Figure(go.Scatterpolar(
            r=[0]*len(cats), theta=cats, fill="toself",
            line=dict(color=THEME["cyan"]),
        ))
        fig.update_layout(**_base_layout(
            title=dict(text="🛡️ Infrastructure Compliance Radar", font=dict(size=16, color=THEME["teal"])),
            polar=dict(
                bgcolor=THEME["bg_card"],
                radialaxis=dict(visible=True, range=[0,1], gridcolor=THEME["grid"]),
                angularaxis=dict(gridcolor=THEME["grid"]),
            ),
        ))
        return fig

    cats = list(standards.keys())
    scores = [np.mean(v) * 100 for v in standards.values()]
    cats_closed = cats + [cats[0]]
    scores_closed = scores + [scores[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(0,212,255,0.12)",
        line=dict(color=THEME["cyan"], width=2),
        marker=dict(color=THEME["cyan"], size=7),
        hovertemplate="%{theta}<br>Compliance: %{r:.0f}%<extra></extra>",
        name="Compliance %",
    ))

    fig.update_layout(**_base_layout(
        title=dict(text="🛡️ Infrastructure Compliance Radar", font=dict(size=16, color=THEME["teal"])),
        polar=dict(
            bgcolor=THEME["bg_card"],
            radialaxis=dict(visible=True, range=[0,100], gridcolor=THEME["grid"],
                            tickfont=dict(size=9)),
            angularaxis=dict(gridcolor=THEME["grid"], tickfont=dict(size=10)),
        ),
        showlegend=False,
    ))
    return fig


# ── Chart 7: Surrogate Measures Scatter (TTC vs PET) ─────────────────────────

def chart_surrogate_scatter(results: list) -> go.Figure:
    records = []
    for r in results:
        sm = r.surrogate_measures
        if sm.ttc_estimate_s is not None or sm.pet_estimate_s is not None:
            records.append({
                "TTC (s)": sm.ttc_estimate_s,
                "PET (s)": sm.pet_estimate_s,
                "Safety Score": r.safety_score,
                "Video": r.video_label,
                "Facility": r.facility_type,
                "Conflicts": r.conflict_count,
            })

    if not records:
        return go.Figure().update_layout(**_base_layout(title="No surrogate measure data"))

    df = pd.DataFrame(records)

    fig = px.scatter(
        df,
        x="TTC (s)", y="PET (s)",
        color="Safety Score",
        size="Conflicts",
        size_max=28,
        symbol="Facility",
        hover_data=["Video", "Facility", "Conflicts", "Safety Score"],
        color_continuous_scale=[
            [0.0, THEME["rose"]],
            [0.5, THEME["amber"]],
            [1.0, THEME["cyan"]],
        ],
        range_color=[0, 100],
    )

    # FHWA SSAM thresholds
    fig.add_hline(y=1.0, line_dash="dash", line_color=THEME["rose"], line_width=1.5,
                  annotation_text="PET < 1.0 s (serious)", annotation_font=dict(size=9, color=THEME["rose"]))
    fig.add_vline(x=1.5, line_dash="dash", line_color=THEME["amber"], line_width=1.5,
                  annotation_text="TTC < 1.5 s (conflict)", annotation_font=dict(size=9, color=THEME["amber"]))

    fig.update_layout(**_base_layout(
        title=dict(text="🎯 Surrogate Safety Measures: TTC vs PET (FHWA SSAM)", font=dict(size=16, color=THEME["amber"])),
        xaxis_title="Time-to-Collision — TTC (s)",
        yaxis_title="Post-Encroachment Time — PET (s)",
    ))
    fig.update_traces(marker=dict(line=dict(color=THEME["bg"], width=0.8)))
    return fig


# ── CV Chart A: Motion Direction Polar (Wind-Rose) ───────────────────────────

def chart_motion_polar(results: list) -> go.Figure:
    """Optical flow direction histogram as a polar wind-rose per video."""
    DIR_LABELS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    fig = go.Figure()
    has_data = False
    for i, r in enumerate(results):
        cv = getattr(r, "cv_result", None)
        if cv is None or not any(cv.motion_direction_hist):
            continue
        hist = cv.motion_direction_hist + [cv.motion_direction_hist[0]]  # close loop
        labels = DIR_LABELS + [DIR_LABELS[0]]
        color = CONFLICT_PALETTE[i % len(CONFLICT_PALETTE)]
        fig.add_trace(go.Scatterpolar(
            r=hist, theta=labels,
            fill="toself", fillcolor=_hex_to_rgba(color, 0.18),
            line=dict(color=color, width=2),
            name=r.video_label,
            hovertemplate="Direction: %{theta}<br>Weight: %{r:.3f}<extra></extra>",
        ))
        has_data = True
    if not has_data:
        fig.add_annotation(text="No optical flow data", showarrow=False,
                           font=dict(color=THEME["muted"], size=14))
    fig.update_layout(**_base_layout(
        title=dict(text="🧭 Dominant Motion Directions (Optical Flow)", font=dict(size=15, color=THEME["sky"])),
        polar=dict(
            bgcolor=THEME["bg_card"],
            radialaxis=dict(visible=True, tickfont=dict(size=9, color=THEME["muted"]),
                            gridcolor=THEME["grid"], linecolor=THEME["grid"]),
            angularaxis=dict(gridcolor=THEME["grid"], tickfont=dict(size=11, color=THEME["text"])),
        ),
        showlegend=True,
    ))
    return fig


# ── CV Chart B: Spatial Activity Heatmap (Plotly) ────────────────────────────

def chart_activity_heatmap(results: list) -> go.Figure:
    """Accumulated frame-delta spatial heatmap — shows where motion occurs."""
    cv_results = [r.cv_result for r in results if getattr(r, "cv_result", None) is not None]
    if not cv_results:
        return go.Figure().update_layout(**_base_layout(title="No CV data"))

    # Average heatmaps across all results that share a video
    from collections import defaultdict
    label_maps: dict[str, list] = defaultdict(list)
    for r in results:
        cv = getattr(r, "cv_result", None)
        if cv and cv._activity_array is not None:
            label_maps[r.video_label].append(cv._activity_array)

    if not label_maps:
        return go.Figure().update_layout(**_base_layout(title="No heatmap data"))

    # Use first video's heatmap
    first_label = list(label_maps.keys())[0]
    maps = label_maps[first_label]
    avg_map = np.mean(np.stack(maps, axis=0), axis=0)

    fig = go.Figure(go.Heatmap(
        z=avg_map,
        colorscale=[
            [0.0, THEME["bg_card"]], [0.3, "#1a365d"],
            [0.6, THEME["violet"]], [0.85, THEME["amber"]], [1.0, THEME["rose"]],
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text="Activity", font=dict(color=THEME["muted"], size=10)),
            tickfont=dict(color=THEME["muted"], size=9),
        ),
        hovertemplate="Row %{y} · Col %{x}<br>Activity: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=f"🌡 Spatial Activity Heatmap — {first_label}", font=dict(size=15, color=THEME["rose"])),
        xaxis=dict(showticklabels=False, gridcolor=THEME["grid"]),
        yaxis=dict(showticklabels=False, autorange="reversed", gridcolor=THEME["grid"]),
    ))
    return fig


# ── CV Chart C: Per-Frame Motion Energy Timeline ──────────────────────────────

def chart_motion_energy(results: list) -> go.Figure:
    """Per-frame motion energy (mean optical flow magnitude) timeline."""
    fig = go.Figure()
    has_data = False
    for i, r in enumerate(results):
        cv = getattr(r, "cv_result", None)
        if cv is None or not cv.frame_motion_energy:
            continue
        color = CONFLICT_PALETTE[i % len(CONFLICT_PALETTE)]
        xs = list(range(1, len(cv.frame_motion_energy) + 1))
        fig.add_trace(go.Scatter(
            x=xs, y=cv.frame_motion_energy,
            name=r.video_label,
            mode="lines+markers",
            line=dict(width=2.5, color=color, shape="spline", smoothing=1.1),
            marker=dict(size=8, color=color, line=dict(width=1.5, color=THEME["bg_card"])),
            fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.08),
            hovertemplate="Frame pair %{x}<br>Motion energy: %{y:.2f} px<extra></extra>",
        ))
        has_data = True
    if not has_data:
        fig.add_annotation(text="No motion data", showarrow=False, font=dict(color=THEME["muted"], size=14))
    fig.update_layout(**_base_layout(
        title=dict(text="⚡ Inter-Frame Motion Energy", font=dict(size=15, color=THEME["amber"])),
        xaxis_title="Frame Pair Index",
        yaxis_title="Mean Flow Magnitude (px)",
        hovermode="x unified",
    ))
    return fig


# ── CV Chart D: Edge Density & Scene Complexity ───────────────────────────────

def chart_edge_density(results: list) -> go.Figure:
    """
    Grouped bar: mean edge density (infrastructure complexity) vs mean motion
    per video — dual-axis for direct comparison.
    """
    labels, edge_vals, motion_vals, brightness_vals = [], [], [], []
    for r in results:
        cv = getattr(r, "cv_result", None)
        if cv is None:
            continue
        if r.video_label not in labels:
            labels.append(r.video_label)
            edge_vals.append(round(cv.mean_edge_density * 100, 2))
            motion_vals.append(round(cv.mean_motion_magnitude, 2))
            brightness_vals.append(round(cv.mean_brightness, 1))

    if not labels:
        return go.Figure().update_layout(**_base_layout(title="No CV data"))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=labels, y=edge_vals, name="Edge Density (%)",
        marker=dict(color=THEME["teal"], opacity=0.85,
                    line=dict(color=THEME["bg_card"], width=1)),
        hovertemplate="%{x}<br>Edge density: %{y:.2f}%<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=labels, y=motion_vals, name="Motion Magnitude (px)",
        mode="markers+lines",
        marker=dict(size=12, color=THEME["amber"], symbol="diamond",
                    line=dict(width=2, color=THEME["bg_card"])),
        line=dict(width=2, color=THEME["amber"], dash="dot"),
        hovertemplate="%{x}<br>Motion: %{y:.2f} px<extra></extra>",
    ), secondary_y=True)
    fig.update_layout(**_base_layout(
        title=dict(text="🔬 Infrastructure Complexity vs Scene Motion", font=dict(size=15, color=THEME["teal"])),
        barmode="group",
    ))
    fig.update_yaxes(title_text="Edge Density (% pixels)", secondary_y=False,
                     gridcolor=THEME["grid"], tickfont=dict(size=10, color=THEME["teal"]))
    fig.update_yaxes(title_text="Mean Motion Magnitude (px)", secondary_y=True,
                     tickfont=dict(size=10, color=THEME["amber"]))
    return fig


# ── Summary statistics ────────────────────────────────────────────────────────

def compute_summary_stats(results: list) -> dict:
    if not results:
        return {}
    df = results_to_df(results)
    speeds = [s for r in results for s in r.estimated_speeds_mph]
    return {
        "total_analyses":     len(results),
        "avg_safety_score":   round(df["safety_score"].mean(), 1),
        "min_safety_score":   round(df["safety_score"].min(), 1),
        "total_conflicts":    int(df["conflict_count"].sum()),
        "avg_conflict_rate":  round(df["conflict_rate"].mean(), 2),
        "avg_vehicle_count":  round(df["vehicle_count"].mean(), 1),
        "avg_speed_mph":      round(np.mean(speeds), 1) if speeds else 0.0,
        "p85_speed_mph":      round(float(np.percentile(speeds, 85)), 1) if speeds else 0.0,
        "total_frames":       len(results),
        "videos_analyzed":    df["video_label"].nunique(),
    }


# ── CSV Export ────────────────────────────────────────────────────────────────

def export_csv(results: list) -> bytes:
    df = results_to_df(results)
    return df.to_csv(index=False).encode("utf-8")


# ── LaTeX Table ───────────────────────────────────────────────────────────────

def generate_latex_table(results: list) -> str:
    if not results:
        return "% No data"

    df = results_to_df(results)
    summary = (
        df.groupby("video_label")
          .agg(
              Frames=("result_id", "count"),
              Avg_Score=("safety_score", "mean"),
              Min_Score=("safety_score", "min"),
              Conflicts=("conflict_count", "sum"),
              Avg_Speed=("avg_speed_mph", "mean"),
              Avg_TTC=("ttc_estimate_s", "mean"),
          )
          .round(2)
          .reset_index()
    )

    rows = []
    for _, row in summary.iterrows():
        label = row["video_label"].replace("_", r"\_")
        ttc = f"{row['Avg_TTC']:.2f}" if not pd.isna(row["Avg_TTC"]) else "---"
        rows.append(
            f"    {label} & {int(row['Frames'])} & {row['Avg_Score']:.1f} & "
            f"{row['Min_Score']:.1f} & {int(row['Conflicts'])} & "
            f"{row['Avg_Speed']:.1f} & {ttc} \\\\"
        )

    rows_str = "\n".join(rows)
    return textwrap.dedent(rf"""
\begin{{table}}[htbp]
\centering
\caption{{Traffic Safety Analysis Summary by Video Feed — VLM-Based Surrogate Measures}}
\label{{tab:safety_summary}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
Video Feed & Frames & \multicolumn{{2}}{{c}}{{Safety Score}} & Conflicts & Avg Speed & Avg TTC \\
           &        & (mean) & (min)  & (count)   & (mph)     & (s)     \\
\midrule
{rows_str}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}
\small
\item \textit{{Safety Score}}: 0--100 composite index (deductions for conflicts, non-compliance); see §3.2.
\item \textit{{TTC}}: Time-to-Collision estimate from VLM spatial reasoning; cf.\ FHWA SSAM threshold $< 1.5$\,s.
\item \textit{{Speed}}: Estimated from lane-width reference ($3.65$\,m per standard lane, AASHTO Green Book).
\end{{tablenotes}}
\end{{table}}
""").strip()


# ── Executive Markdown Report ─────────────────────────────────────────────────

def generate_executive_report(results: list, session_id: str = "") -> str:
    if not results:
        return "_No analyses completed yet._"

    stats = compute_summary_stats(results)
    df = results_to_df(results)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Top recommendations across all results
    all_recs = []
    for r in results:
        all_recs.extend(r.recommendations)
    top_recs = list(dict.fromkeys(all_recs))[:8]

    # Manual references
    all_refs = []
    for r in results:
        all_refs.extend(r.manual_references)
    unique_refs = sorted(set(all_refs))[:12]

    recs_md = "\n".join(f"- {rec}" for rec in top_recs) if top_recs else "_No recommendations generated._"
    refs_md = "\n".join(f"- {ref}" for ref in unique_refs) if unique_refs else "_No references cited._"

    narratives = [r.narrative_summary for r in results if r.narrative_summary]
    narrative_block = "\n\n".join(
        f"> **Frame {r.frame_timestamp_s:.0f}s** ({r.video_label}): {r.narrative_summary}"
        for r in results[:5] if r.narrative_summary
    )

    return f"""# Traffic Safety Analysis Report
**Session:** `{session_id}` | **Generated:** {now}

---

## Executive Summary

| Metric | Value |
|---|---|
| Total Analyses | {stats.get('total_analyses', 0)} frames |
| Videos Analyzed | {stats.get('videos_analyzed', 0)} |
| Mean Safety Score | **{stats.get('avg_safety_score', 0):.1f} / 100** |
| Min Safety Score | {stats.get('min_safety_score', 0):.1f} |
| Total Conflicts Detected | {stats.get('total_conflicts', 0)} |
| Mean Conflict Rate | {stats.get('avg_conflict_rate', 0):.2f} per 100 vehicles |
| Mean Estimated Speed | {stats.get('avg_speed_mph', 0):.1f} mph |
| 85th Percentile Speed | {stats.get('p85_speed_mph', 0):.1f} mph |

---

## Methodology

Video frames were sampled from YouTube traffic camera streams using `yt-dlp` and `OpenCV`.
Each frame set was submitted to a Vision Language Model (VLM) with facility-type-specific
structured prompts aligned to:
- **HSM** (Highway Safety Manual, 1st Ed. + 2014 Supplement)
- **FHWA SSAM** (Surrogate Safety Assessment Model methodology)
- **HCM 7th Edition** (Highway Capacity Manual)
- **MUTCD 2023** (Manual on Uniform Traffic Control Devices)
- **AASHTO Green Book** (A Policy on Geometric Design, 2018)

Conflict severity follows a 1–5 ordinal scale (1 = minor, 5 = imminent crash).
Safety Score = 100 − Σ(severity-weighted deductions), bounded [0, 100].

---

## Frame-Level Observations

{narrative_block}

---

## Key Recommendations

{recs_md}

---

## Manual References Cited

{refs_md}

---

*This report was generated automatically by the VLM Traffic Safety Analyzer. All VLM-based
estimates carry inherent uncertainty and should be validated against field measurements
before use in engineering decisions.*
"""
