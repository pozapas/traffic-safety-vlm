"""
speed_limit_lookup.py
─────────────────────────────────────────────────────────────────
Two-stage speed limit detection:
  Stage 1 — VLM reads posted signs from video frames (handled in
             prompt_library.py schema; result stored in SafetyAnalysisResult)
  Stage 2 — OpenStreetMap Overpass API fallback:
             location string → Nominatim geocode → Overpass maxspeed query

All network calls have short timeouts and return None on failure so
they never block the main analysis pipeline.
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_OVERPASS_URL  = "https://overpass-api.de/api/interpreter"
_HEADERS = {"User-Agent": "VLM-Traffic-Safety-Analyzer/1.0 (research)"}


# ── Stage 2a: Geocode a location string ──────────────────────────────────────

def geocode_location(location_hint: str) -> Optional[tuple[float, float]]:
    """
    Convert a free-text location (e.g. 'Hopkins Street San Marcos TX')
    to (lat, lon) via OSM Nominatim.

    Returns None on failure.
    """
    if not location_hint or len(location_hint.strip()) < 4:
        return None
    try:
        resp = requests.get(
            _NOMINATIM_URL,
            params={"q": location_hint, "format": "json", "limit": 1},
            headers=_HEADERS,
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logger.debug(f"Nominatim geocode failed for '{location_hint}': {e}")
    return None


# ── Stage 2b: Query OSM Overpass for speed limit ──────────────────────────────

def query_osm_speed_limit(lat: float, lon: float, radius_m: int = 50) -> Optional[float]:
    """
    Query OpenStreetMap Overpass API for the maxspeed tag on the nearest
    road within `radius_m` metres of (lat, lon).

    Returns speed limit in mph (converts km/h if needed), or None.
    """
    query = f"""
    [out:json][timeout:8];
    way(around:{radius_m},{lat},{lon})["highway"]["maxspeed"];
    out tags 1;
    """
    try:
        resp = requests.post(
            _OVERPASS_URL,
            data={"data": query},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        if not elements:
            # Widen search radius once
            query2 = f"""
            [out:json][timeout:8];
            way(around:150,{lat},{lon})["highway"]["maxspeed"];
            out tags 1;
            """
            resp2 = requests.post(_OVERPASS_URL, data={"data": query2},
                                  headers=_HEADERS, timeout=10)
            elements = resp2.json().get("elements", [])

        for el in elements:
            raw = el.get("tags", {}).get("maxspeed", "")
            mph = _parse_maxspeed(raw)
            if mph is not None:
                return mph
    except Exception as e:
        logger.debug(f"Overpass query failed at ({lat},{lon}): {e}")
    return None


def _parse_maxspeed(raw: str) -> Optional[float]:
    """
    Parse OSM maxspeed tag values like:
      '55 mph', '55', '88 km/h', 'US:urban' (→ 25 mph default), etc.
    Returns speed in mph or None.
    """
    if not raw:
        return None

    raw = raw.strip().lower()

    # US jurisdiction defaults
    _US_DEFAULTS = {
        "us:urban": 25.0, "us:rural": 55.0, "us:motorway": 65.0,
        "us:living_street": 15.0,
    }
    if raw in _US_DEFAULTS:
        return _US_DEFAULTS[raw]

    # Explicit mph
    m = re.match(r"(\d+(?:\.\d+)?)\s*mph", raw)
    if m:
        return float(m.group(1))

    # Explicit km/h
    m = re.match(r"(\d+(?:\.\d+)?)\s*(?:km/h|kph|kmh)?$", raw)
    if m:
        kmh = float(m.group(1))
        # Heuristic: if the number looks like mph already (≤ 85), keep it
        # OSM in the US often stores bare numbers as mph
        if kmh <= 85:
            return kmh          # likely already mph
        return round(kmh * 0.621371, 1)   # convert km/h → mph

    return None


# ── High-level convenience function ──────────────────────────────────────────

def resolve_speed_limit(
    vlm_detected: Optional[float],
    location_hint: Optional[str],
    manual_override: Optional[float] = None,
) -> tuple[Optional[float], str]:
    """
    Return (speed_limit_mph, source_label) using priority:
      1. Manual override (sidebar)
      2. VLM-detected from video signs
      3. OpenStreetMap lookup via location hint
      4. None  (unknown)

    source_label is one of:
      'manual', 'vlm_sign', 'osm', 'unknown'
    """
    if manual_override is not None and manual_override > 0:
        return manual_override, "manual"

    if vlm_detected is not None and vlm_detected > 0:
        return vlm_detected, "vlm_sign"

    if location_hint:
        coords = geocode_location(location_hint)
        if coords:
            lat, lon = coords
            osm_limit = query_osm_speed_limit(lat, lon)
            if osm_limit is not None:
                return osm_limit, "osm"

    return None, "unknown"
