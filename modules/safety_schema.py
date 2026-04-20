"""
safety_schema.py
─────────────────────────────────────────────────────────────────
Defines the canonical SafetyAnalysisResult dataclass and the
LLM-response parser.  Every VLM adapter returns a raw string;
this module converts it to a structured object for storage and
visualisation.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# CV analysis result (imported lazily to avoid circular import)
try:
    from modules.cv_analysis import CVAnalysisResult as _CVResult
except ImportError:
    _CVResult = None  # type: ignore


# ──────────────────────────────────────────────
# Sub-structures
# ──────────────────────────────────────────────

@dataclass
class VehicleClasses:
    car: int = 0
    truck: int = 0
    motorcycle: int = 0
    bus: int = 0
    bicycle: int = 0
    pedestrian: int = 0

    @property
    def total(self) -> int:
        return self.car + self.truck + self.motorcycle + self.bus + self.bicycle + self.pedestrian


@dataclass
class ConflictEvent:
    conflict_type: str = "unknown"          # e.g. merge_conflict, rear_end_risk
    severity: int = 1                        # 1 (minor) – 5 (critical)
    involved_vehicles: int = 2
    description: str = ""
    manual_reference: str = ""              # e.g. "HSM Ch.18 §3.2"


@dataclass
class SurrogateMeasures:
    ttc_estimate_s: Optional[float] = None  # Time-to-Collision (seconds)
    pet_estimate_s: Optional[float] = None  # Post-Encroachment Time (seconds)
    gap_acceptance_s: Optional[float] = None
    deceleration_rate_ms2: Optional[float] = None
    notes: str = ""


@dataclass
class InfrastructureCompliance:
    standard: str = ""          # e.g. "MUTCD R4-7", "AASHTO Green Book §9"
    observation: str = ""
    compliant: Optional[bool] = None


# ──────────────────────────────────────────────
# Main result object
# ──────────────────────────────────────────────

@dataclass
class SafetyAnalysisResult:
    # Identifiers
    result_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str = ""
    video_label: str = ""
    video_url: str = ""
    frame_timestamp_s: float = 0.0
    wall_time: float = field(default_factory=time.time)

    # Facility metadata
    facility_type: str = "unknown"
    model_used: str = ""
    provider: str = ""

    # Speed limit detection (multi-source)
    posted_speed_limit_mph: Optional[float] = None
    speed_limit_source: str = "unknown"
    detected_location: Optional[str] = None

    # Computer Vision analysis (populated after frame extraction)
    cv_result: Optional[object] = None   # CVAnalysisResult

    # Traffic state
    vehicle_count: int = 0
    vehicle_classes: VehicleClasses = field(default_factory=VehicleClasses)
    estimated_speeds_mph: list[float] = field(default_factory=list)
    avg_speed_mph: Optional[float] = None
    speed_variance_mph: Optional[float] = None

    # Conflict data
    conflict_events: list[ConflictEvent] = field(default_factory=list)
    conflict_count: int = 0
    dominant_conflict_type: str = "none"

    # Surrogate safety measures (HSM / FHWA)
    surrogate_measures: SurrogateMeasures = field(default_factory=SurrogateMeasures)

    # Composite scores
    safety_score: float = 100.0        # 0 (worst) – 100 (best)
    severity_index: float = 0.0        # Σ(severity × count) / total_vehicles
    conflict_rate: float = 0.0         # conflicts per 100 vehicles (HSM §3)

    # Infrastructure
    compliance_observations: list[InfrastructureCompliance] = field(default_factory=list)
    infrastructure_condition: str = "good"  # good / fair / poor

    # Qualitative outputs
    narrative_summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    manual_references: list[str] = field(default_factory=list)

    # Raw LLM output (for reproducibility)
    raw_response: str = ""
    parse_success: bool = True
    parse_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_flat_dict(self) -> dict:
        """Flattened for pandas DataFrame export."""
        speeds = self.estimated_speeds_mph
        return {
            "result_id":            self.result_id,
            "session_id":           self.session_id,
            "video_label":          self.video_label,
            "facility_type":        self.facility_type,
            "model_used":           self.model_used,
            "provider":             self.provider,
            "frame_timestamp_s":    self.frame_timestamp_s,
            "posted_speed_limit":   self.posted_speed_limit_mph,
            "speed_limit_source":   self.speed_limit_source,
            "detected_location":    self.detected_location,
            "vehicle_count":        self.vehicle_count,
            "vc_car":               self.vehicle_classes.car,
            "vc_truck":             self.vehicle_classes.truck,
            "vc_motorcycle":        self.vehicle_classes.motorcycle,
            "vc_bus":               self.vehicle_classes.bus,
            "vc_bicycle":           self.vehicle_classes.bicycle,
            "vc_pedestrian":        self.vehicle_classes.pedestrian,
            "avg_speed_mph":        self.avg_speed_mph,
            "speed_variance_mph":   self.speed_variance_mph,
            "conflict_count":       self.conflict_count,
            "dominant_conflict":    self.dominant_conflict_type,
            "safety_score":         self.safety_score,
            "severity_index":       self.severity_index,
            "conflict_rate":        self.conflict_rate,
            "ttc_estimate_s":       self.surrogate_measures.ttc_estimate_s,
            "pet_estimate_s":       self.surrogate_measures.pet_estimate_s,
            "gap_acceptance_s":     self.surrogate_measures.gap_acceptance_s,
            "infrastructure":       self.infrastructure_condition,
            "parse_success":        self.parse_success,
        }


# ──────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────

def _extract_json_block(text: str) -> Optional[str]:
    """Try to extract the first ```json ... ``` or bare { ... } block."""
    # Fenced JSON block
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fenced generic block
    m = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Bare JSON object (greedy last)
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _safe_int(val, default=0) -> int:
    try:
        return int(val)
    except Exception:
        return default


def _safe_float(val, default=None):
    try:
        return float(val)
    except Exception:
        return default


def parse_vlm_response(
    raw: str,
    *,
    session_id: str = "",
    video_label: str = "",
    video_url: str = "",
    frame_timestamp_s: float = 0.0,
    facility_type: str = "unknown",
    model_used: str = "",
    provider: str = "",
) -> SafetyAnalysisResult:
    """
    Parse a VLM response string into a SafetyAnalysisResult.
    Strategy: attempt JSON extraction → field-by-field parsing → sensible defaults.
    """
    warnings: list[str] = []
    result = SafetyAnalysisResult(
        session_id=session_id,
        video_label=video_label,
        video_url=video_url,
        frame_timestamp_s=frame_timestamp_s,
        facility_type=facility_type,
        model_used=model_used,
        provider=provider,
        raw_response=raw,
    )

    json_str = _extract_json_block(raw)
    if json_str is None:
        warnings.append("No JSON block found in VLM response; using fallback extraction.")
        result.parse_success = False
        result.parse_warnings = warnings
        result.narrative_summary = raw[:2000]
        return result

    try:
        data: dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to clean common LLM JSON mistakes
        cleaned = re.sub(r",\s*([}\]])", r"\1", json_str)   # trailing commas
        cleaned = re.sub(r"//[^\n]*", "", cleaned)           # JS comments
        try:
            data = json.loads(cleaned)
        except Exception:
            warnings.append(f"JSON parse error: {e}. Fallback to narrative.")
            result.parse_success = False
            result.parse_warnings = warnings
            result.narrative_summary = raw[:2000]
            return result

    # ── Vehicle counts ──
    result.vehicle_count = _safe_int(data.get("vehicle_count", 0))

    # ── Speed limit & location (VLM-detected) ──
    result.detected_location = data.get("detected_location") or None
    vlm_speed = _safe_float(data.get("posted_speed_limit_mph"))
    if vlm_speed and vlm_speed > 0:
        result.posted_speed_limit_mph = vlm_speed
        result.speed_limit_source = "vlm_sign"

    vc_raw = data.get("vehicle_classes", {})
    if isinstance(vc_raw, dict):
        result.vehicle_classes = VehicleClasses(
            car=_safe_int(vc_raw.get("car", vc_raw.get("cars", 0))),
            truck=_safe_int(vc_raw.get("truck", vc_raw.get("trucks", 0))),
            motorcycle=_safe_int(vc_raw.get("motorcycle", vc_raw.get("motorcycles", 0))),
            bus=_safe_int(vc_raw.get("bus", vc_raw.get("buses", 0))),
            bicycle=_safe_int(vc_raw.get("bicycle", vc_raw.get("bicycles", 0))),
            pedestrian=_safe_int(vc_raw.get("pedestrian", vc_raw.get("pedestrians", 0))),
        )

    # ── Speed data ──
    speeds_raw = data.get("estimated_speeds", data.get("estimated_speeds_mph", []))
    if isinstance(speeds_raw, list):
        result.estimated_speeds_mph = [_safe_float(s) for s in speeds_raw if _safe_float(s) is not None]
    if result.estimated_speeds_mph:
        import numpy as np
        arr = result.estimated_speeds_mph
        result.avg_speed_mph = round(float(np.mean(arr)), 1)
        result.speed_variance_mph = round(float(np.var(arr)), 2)

    # ── Conflict events ──
    conflicts_raw = data.get("conflict_events", [])
    events: list[ConflictEvent] = []
    if isinstance(conflicts_raw, list):
        for c in conflicts_raw:
            if isinstance(c, dict):
                events.append(ConflictEvent(
                    conflict_type=str(c.get("type", c.get("conflict_type", "unknown"))),
                    severity=_safe_int(c.get("severity", 1)),
                    involved_vehicles=_safe_int(c.get("involved_vehicles", 2)),
                    description=str(c.get("description", "")),
                    manual_reference=str(c.get("manual_reference", "")),
                ))
    result.conflict_events = events
    result.conflict_count = len(events)
    if events:
        from collections import Counter
        type_counts = Counter(e.conflict_type for e in events)
        result.dominant_conflict_type = type_counts.most_common(1)[0][0]

    # ── Surrogate measures ──
    sm_raw = data.get("surrogate_measures", {})
    if isinstance(sm_raw, dict):
        result.surrogate_measures = SurrogateMeasures(
            ttc_estimate_s=_safe_float(sm_raw.get("ttc_estimate_s", sm_raw.get("ttc"))),
            pet_estimate_s=_safe_float(sm_raw.get("pet_estimate_s", sm_raw.get("pet"))),
            gap_acceptance_s=_safe_float(sm_raw.get("gap_acceptance_s", sm_raw.get("gap_acceptance"))),
            deceleration_rate_ms2=_safe_float(sm_raw.get("deceleration_rate_ms2")),
            notes=str(sm_raw.get("notes", "")),
        )

    # ── Composite scores ──
    result.safety_score = _safe_float(data.get("safety_score", 100.0)) or 100.0
    result.severity_index = _safe_float(data.get("severity_index", 0.0)) or 0.0
    if result.vehicle_count > 0:
        result.conflict_rate = round(result.conflict_count / result.vehicle_count * 100, 2)

    # ── Infrastructure compliance ──
    infra_raw = data.get("infrastructure_compliance", [])
    observations: list[InfrastructureCompliance] = []
    if isinstance(infra_raw, list):
        for obs in infra_raw:
            if isinstance(obs, dict):
                comp = obs.get("compliant")
                observations.append(InfrastructureCompliance(
                    standard=str(obs.get("standard", "")),
                    observation=str(obs.get("observation", "")),
                    compliant=bool(comp) if comp is not None else None,
                ))
    result.compliance_observations = observations
    result.infrastructure_condition = str(data.get("infrastructure_condition", "good"))

    # ── Qualitative ──
    result.narrative_summary = str(data.get("narrative_summary", data.get("summary", "")))
    recs = data.get("recommendations", [])
    result.recommendations = [str(r) for r in recs] if isinstance(recs, list) else []
    refs = data.get("manual_references", [])
    result.manual_references = [str(r) for r in refs] if isinstance(refs, list) else []

    result.parse_success = True
    result.parse_warnings = warnings
    return result
