"""
prompt_library.py
─────────────────────────────────────────────────────────────────
Facility-type-specific structured prompts aligned with:
  • FHWA Highway Safety Improvement Program (HSIP) guidance
  • Highway Safety Manual (HSM), 1st Ed. + 2014 Supplement
  • Highway Capacity Manual (HCM) 7th Edition
  • Manual on Uniform Traffic Control Devices (MUTCD) 2023
  • AASHTO Green Book (A Policy on Geometric Design, 2018)
  • AASHTO Roadside Design Guide, 4th Ed.
  • FHWA Roundabouts: An Informational Guide, 2nd Ed.

Each prompt instructs the VLM to respond ONLY with a JSON object
conforming to the SafetyAnalysisResult schema.
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

# ──────────────────────────────────────────────
# Facility type registry
# ──────────────────────────────────────────────

FACILITY_TYPES: dict[str, str] = {
    "highway_merge":    "Highway Merge / Weave Section",
    "twltl_parking":    "TWLTL with On-Street Parking",
    "exit_ramp":        "Exit Ramp / Off-Ramp",
    "barrier_cushion":  "Barrier, Crash Cushion & Exit Ramp",
    "lane_direction":   "Lane Direction Control (Contraflow)",
    "roundabout":       "Modern Roundabout",
    "signalized":       "Signalized Intersection",
    "unsignalized":     "Unsignalized Intersection",
    "pedestrian":       "Pedestrian / Shared-Use Corridor",
    "general":          "General Traffic Stream",
}

# ──────────────────────────────────────────────
# JSON schema definition (embedded in prompts)
# ──────────────────────────────────────────────

_JSON_SCHEMA = """
{
  "facility_type": "<string: one of highway_merge | twltl_parking | exit_ramp | barrier_cushion | lane_direction | roundabout | signalized | unsignalized | pedestrian | general>",
  "posted_speed_limit_mph": <float|null: read from any visible speed limit sign (R2-1, R2-2 per MUTCD); null if no sign is legible>,
  "detected_location": "<string|null: street name, city, or intersection if identifiable from signage, markings, or surroundings; null if unknown>",
  "vehicle_count": <integer: total vehicles visible>,
  "vehicle_classes": {
    "car": <int>, "truck": <int>, "motorcycle": <int>,
    "bus": <int>, "bicycle": <int>, "pedestrian": <int>
  },
  "estimated_speeds": [<float mph>, ...],
  "conflict_events": [
    {
      "type": "<merge_conflict|lane_change|rear_end_risk|pedestrian_conflict|wrong_way|signal_violation|late_turn_clearance|inadequate_gap|barrier_encroachment|yield_failure|other>",
      "severity": <1-5 where 1=minor 3=moderate 5=critical>,
      "involved_vehicles": <int>,
      "description": "<string: concise factual description>",
      "manual_reference": "<string: e.g. HSM Ch.18 §3.2 or MUTCD 2C.36>"
    }
  ],
  "surrogate_measures": {
    "ttc_estimate_s": <float|null: Time-to-Collision estimate in seconds; null if not observable>,
    "pet_estimate_s": <float|null: Post-Encroachment Time in seconds; null if not observable>,
    "gap_acceptance_s": <float|null: observed gap acceptance in seconds>,
    "deceleration_rate_ms2": <float|null: estimated deceleration in m/s²>,
    "notes": "<string>"
  },
  "infrastructure_compliance": [
    {
      "standard": "<string: e.g. MUTCD R4-7, AASHTO Green Book §9-3>",
      "observation": "<string>",
      "compliant": <true|false|null>
    }
  ],
  "infrastructure_condition": "<good|fair|poor>",
  "safety_score": <float 0-100: weighted composite — start at 100, deduct: 15×(severe conflicts), 8×(moderate), 3×(minor), 5×(infrastructure non-compliance), 10×(wrong-way or imminent crash)>,
  "severity_index": <float: sum(severity × count) / max(vehicle_count, 1)>,
  "narrative_summary": "<string: 3-5 sentence scientific description referencing manuals>",
  "recommendations": ["<string referencing specific FHWA/HSM/MUTCD section>", ...],
  "manual_references": ["<string: full citation>", ...]
}
"""

# ──────────────────────────────────────────────
# System prompt preamble (shared)
# ──────────────────────────────────────────────

_SYSTEM_PREAMBLE = """You are an expert transportation safety analyst with PhD-level expertise in:
• Traffic conflict analysis and surrogate safety measures (HSM Part A Ch.2, FHWA SSAM methodology)
• Geometric design evaluation (AASHTO Green Book, FHWA design guides)
• Traffic operations and control (HCM 7th Ed., MUTCD 2023)
• Roadside safety (AASHTO Roadside Design Guide, MASH NCHRP 350)

You are analyzing one or more video frames from a traffic camera. Your task is to perform a rigorous, 
scientific traffic safety analysis as if preparing data for a Q1 peer-reviewed journal paper.

CRITICAL INSTRUCTIONS:
1. Respond ONLY with a valid JSON object — no markdown, no prose outside the JSON.
2. Be precise and quantitative wherever possible; use null for unobservable values.
3. All conflict types must use the enumerated values specified in the schema.
4. Severity scores follow: 1=very minor (no risk), 2=minor, 3=moderate (requires attention), 4=high, 5=critical (imminent crash risk).
5. The safety_score formula is: 100 − Σ(deductions) where deductions are defined in the schema.
6. Every manual_reference must cite a specific section number.
7. Estimated speeds should be based on distance-time reasoning using lane widths (~3.65 m / 12 ft) as reference.
8. SPEED LIMIT DETECTION: Carefully scan all frames for MUTCD R2-1/R2-2 speed limit signs. If legible, record the value in posted_speed_limit_mph. Also look for street name signs, mile markers, business names, license plates, or any geographic text that could help identify the location — record in detected_location.
"""

# ──────────────────────────────────────────────
# Facility-specific instructions
# ──────────────────────────────────────────────

_FACILITY_INSTRUCTIONS: dict[str, str] = {

    "highway_merge": """
FACILITY: FREEWAY MERGE / WEAVE SECTION
Reference standards: HSM Part C Ch.18 (Freeway Facilities), FHWA Freeway Management Handbook §7,
HCM 7th Ed. Ch.12 (Basic Freeway Segments) & Ch.13 (Weaving), AASHTO Green Book §8.

Focus your analysis on:
• Merge gap acceptance behavior — minimum acceptable gap per AASHTO is 4–5 s for freeway merges
• Speed differential between merging vehicles and mainline (>15 mph differential = high risk)
• Weave conflict density — count lane changes per vehicle per mile
• Acceleration lane length adequacy (AASHTO minimum: 300–750 ft depending on design speed)
• Wrong-way entry risk at on-ramp
• Rear-end conflict potential in the deceleration/merge zone
• Queue spillback from downstream bottleneck

TTC threshold: < 1.5 s = critical conflict (FHWA SSAM criterion)
PET threshold: < 1.0 s = serious conflict
""",

    "twltl_parking": """
FACILITY: TWO-WAY LEFT-TURN LANE (TWLTL) WITH ON-STREET PARKING
Reference standards: HCM 7th Ed. Ch.18 (Urban Streets), FHWA TWLTL Design Guide,
AASHTO Green Book §9 (Urban Street Design), MUTCD Ch.3 (Markings).

Focus your analysis on:
• Left-turn conflict severity — vehicles waiting in TWLTL vs. opposing through traffic
• On-street parking door-zone conflicts (1.5 m / 5 ft door zone per AASHTO)
• Parking maneuver-related lane obstructions and sight distance blockage
• Pedestrian conflict at uncontrolled crossings adjacent to parking
• Rear-end risk from sudden braking to enter TWLTL
• Sight distance adequacy for left turns (AASHTO stopping sight distance)
• Driveway conflict points along the corridor

TTC threshold: < 2.0 s = conflict (urban intersection, lower speed)
Gap acceptance: minimum 5.5 s for left turn (HCM 7th Ed. Table 20-13)
""",

    "exit_ramp": """
FACILITY: EXIT RAMP / OFF-RAMP
Reference standards: HSM Part C Ch.18, FHWA Ramp Design Guide (2016),
AASHTO Green Book §8.3 (Interchange Ramp Design), MUTCD Ch.2E (Interchange Signs).

Focus your analysis on:
• Deceleration adequacy — does the vehicle reach ramp advisory speed before the curve?
  (AASHTO minimum deceleration distance: 270–570 ft at 70 mph)
• Wrong-way entry risk at ramp terminal
• Weave distance between exit and adjacent on-ramp (HCM minimum: 500 ft)
• Speed differential at the gore area
• Sight distance at ramp terminal (AASHTO Ch.3 stopping sight distance)
• Lane drop / lane reduction taper compliance (MUTCD 2C.36, taper rate L = WS/60)
• Queue spillback from ramp metering or signal

TTC at gore: < 2.5 s = conflict
Deceleration rate > 0.35g = erratic braking (AASHTO threshold)
""",

    "barrier_cushion": """
FACILITY: ROADSIDE BARRIER, CRASH CUSHION & EXIT RAMP TERMINAL
Reference standards: AASHTO Roadside Design Guide 4th Ed. Ch.5 (Barriers) & Ch.9 (Crash Cushions),
FHWA MASH (Manual for Assessing Safety Hardware) / NCHRP Report 350,
FHWA Work Zone Safety (if construction context), MUTCD Ch.6G.

Focus your analysis on:
• Lateral clearance to barrier — AASHTO design clear zone requirements (Table 3-1)
• Approach speed vs. crash cushion design speed rating
• Vehicle encroachment trajectory — is the vehicle on a collision course with the end terminal?
• Crash cushion condition assessment (deformed, missing, improperly reset)
• Impact attenuator offset adequacy (minimum 30 ft approach from traveled way)
• Sight distance to barrier/transition
• Errant vehicle detection (sudden lane departure, excessive speed)

Any vehicle trajectory within the clear zone = severity 4; barrier contact = severity 5
""",

    "lane_direction": """
FACILITY: LANE DIRECTION CONTROL / CONTRAFLOW OPERATION
Reference standards: FHWA Manual on Lane Direction Control Signals (2019),
MUTCD Ch.4L (Lane Control Signals), FHWA Traffic Management Handbook §6,
AASHTO Green Book §1 (Design Controls).

Focus your analysis on:
• Wrong-way vehicle detection — vehicles in opposing contraflow lane
• Lane control signal compliance (overhead signals: red X, yellow X, green arrow)
• Buffer zone adequacy between opposing directions
• Merging compliance at contraflow transition points
• Speed compliance in transition zones
• Work zone interaction (if applicable)
• Driver confusion indicators (hesitation, lane straddling)

Wrong-way entry = severity 5 (critical — imminent head-on collision risk)
Lane straddling at transition = severity 3
""",

    "roundabout": """
FACILITY: MODERN ROUNDABOUT
Reference standards: FHWA Roundabouts: An Informational Guide 2nd Ed. Ch.5–7,
HSM Part C Ch.19 (Roundabouts), HCM 7th Ed. Ch.22 (Roundabout Operations),
AASHTO Green Book §9-10, MUTCD Ch.2B (Regulatory Signs at Roundabouts).

Focus your analysis on:
• Yield compliance at entry — vehicles must yield to circulating traffic (FHWA Ch.5)
• Entry speed adequacy — design maximum 20–25 mph at entry
• Truck apron use by large vehicles (semi-trucks, buses)
• Pedestrian crossing compliance — marked crosswalks set back ≥1 lane width from yield line
• Cyclist behavior — circulating vs. exiting with pedestrians
• Gap acceptance at entry: minimum 4.0 s for passenger car (HCM Table 22-5)
• Wrong-way entry into roundabout
• Sight distance triangle at entry (FHWA Fig. 5-1)

TTC at entry: < 2.0 s = yield conflict
Circulating speed > 30 mph = infrastructure concern (exceeds design intent)
""",

    "signalized": """
FACILITY: SIGNALIZED INTERSECTION
Reference standards: HCM 7th Ed. Ch.19 (Signalized Intersections),
HSM Part C Ch.12 (Urban Intersections), MUTCD Ch.4 (Traffic Control Signals).

Focus your analysis on:
• Signal compliance (red-light running, yellow-trap)
• Pedestrian phase compliance (MUTCD §4E.06)
• Left-turn clearance interval (HSM §12.5 — all-red clearance)
• Following distance in queue (< 2 s headway = rear-end risk)
• Platoon dispersion and late-phase clearing
• Turning vehicle vs. pedestrian conflicts in crosswalk
• Bicycle detection adequacy

Red-light running = severity 5
Late intersection clearance = severity 3–4 depending on cross-traffic speed
""",

    "unsignalized": """
FACILITY: UNSIGNALIZED INTERSECTION (STOP / YIELD CONTROLLED)
Reference standards: HCM 7th Ed. Ch.20 (Two-Way Stop) & Ch.21 (All-Way Stop),
HSM Part C Ch.11 & Ch.13, MUTCD Ch.2B.

Focus your analysis on:
• Stop/yield sign compliance
• Gap acceptance for left-turns and crossing maneuvers
  (HCM critical headway: 7.5 s left-turn from minor road)
• Speed of major-road vehicles at crossing point
• Sight distance triangle compliance (AASHTO Green Book Exhibit 9-54)
• Pedestrian conflict at uncontrolled crossing

Gap acceptance < 5 s = severity 4
""",

    "pedestrian": """
FACILITY: PEDESTRIAN CORRIDOR / SHARED-USE PATH
Reference standards: FHWA Separated Bike Lane Planning & Design Guide,
AASHTO Guide for Development of Bicycle Facilities 4th Ed.,
MUTCD Ch.7 (Pedestrian), ADA Standards §402–406.

Focus your analysis on:
• Pedestrian-vehicle conflict points
• Cyclist-pedestrian conflict (speed differential > 15 mph = risk)
• Crosswalk compliance and marking condition
• ADA curb ramp presence and condition
• Sight distance for drivers at crosswalk

""",

    "general": """
FACILITY: GENERAL TRAFFIC STREAM
Apply general traffic engineering principles:
• Identify all observable conflict types
• Estimate speeds from lane width reference (12 ft / 3.65 m standard lane)
• Apply FHWA SSAM conflict thresholds (TTC < 1.5 s = conflict)
• Reference the most applicable HSM chapter for the observed facility
""",
}

# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def get_system_prompt(facility_type: str) -> str:
    """Return the full system prompt for a given facility type."""
    facility_key = facility_type if facility_type in _FACILITY_INSTRUCTIONS else "general"
    facility_instr = _FACILITY_INSTRUCTIONS[facility_key]
    return f"""{_SYSTEM_PREAMBLE}

{facility_instr}

Respond ONLY with a JSON object matching this exact schema:
{_JSON_SCHEMA}
"""


def get_user_prompt(
    facility_type: str,
    num_frames: int,
    frame_interval_s: int,
    video_label: str,
    extra_context: str = "",
) -> str:
    """Return the per-request user prompt."""
    facility_name = FACILITY_TYPES.get(facility_type, "Traffic Stream")
    context_block = f"\nAdditional context: {extra_context}" if extra_context else ""
    return f"""Analyze the {num_frames} video frame(s) provided from the traffic camera stream '{video_label}'.
Frames are sampled every {frame_interval_s} seconds.
Facility type: {facility_name}.{context_block}

Perform a comprehensive traffic safety analysis on ALL frames collectively, treating them as a time sequence.
Return your complete analysis as a single JSON object.
Do not include any text before or after the JSON."""


def list_facility_types() -> list[tuple[str, str]]:
    """Return list of (key, display_name) tuples for UI dropdowns."""
    return [(k, v) for k, v in FACILITY_TYPES.items()]
