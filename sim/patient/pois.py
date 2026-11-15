"""POI layer: one internal schema for OSM (stand-in) and Google Maps Places (sponsor, final).

Internal schema (parquet):
    poi_id str, source str ('osm'|'places'), name str, cls str ('clinic'|'pharmacy'|'grocery'),
    raw_category str, lat f64, lon f64, hours_default bool,
    open_wd1000 bool, open_wd1800 bool     # open on a weekday (Tuesday) at 10:00 / 18:00
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("clinic", "pharmacy", "grocery")
DEPARTS = {"weekday_1000": 10.0, "weekday_1800": 18.0}
DAY = "tuesday"

# ---------------------------------------------------------------- category mapping (Places)
# Reviewed lists: keep exact lower-cased Google category strings. Anything not listed is dropped.
# Exclusions matter more than inclusions (vet clinics, beauty "clinics", cannabis "pharmacies").
PLACES_CATEGORY_MAP: dict[str, str] = {
    # clinic: places an adult can get primary / urgent / community care. Reviewed against the full file's
    # category counts (2026-11-14). Deliberately OUT: "doctor" (11.6k individual listings, mostly specialists
    # and duplicates of their clinic), "pediatrician" (children only), "emergency care physician" (ER staff
    # listings duplicating hospitals), mental-health / dental / PT / specialty clinics.
    "medical clinic": "clinic", "clinic": "clinic", "health clinic": "clinic", "walk-in clinic": "clinic",
    "community health centre": "clinic", "community health center": "clinic", "health center": "clinic",
    "urgent care center": "clinic", "urgent care facility": "clinic", "emergency clinic": "clinic",
    "family practice physician": "clinic", "general practitioner": "clinic", "internist": "clinic",
    "medical center": "clinic", "medical group": "clinic", "primary care": "clinic",
    "hospital": "clinic", "general hospital": "clinic", "women's health clinic": "clinic",
    # pharmacy
    "pharmacy": "pharmacy", "drug store": "pharmacy", "drugstore": "pharmacy", "chemist": "pharmacy",
    "compounding pharmacy": "pharmacy",
    # grocery: full-line food stores (not liquor / convenience / food bank / delivery)
    "grocery store": "grocery", "supermarket": "grocery", "asian grocery store": "grocery",
    "mexican grocery store": "grocery", "indian grocery store": "grocery", "korean grocery store": "grocery",
    "japanese grocery store": "grocery", "chinese supermarket": "grocery", "filipino grocery store": "grocery",
    "latin american grocery store": "grocery", "vietnamese grocery store": "grocery", "gourmet grocery store": "grocery",
    "produce market": "grocery", "health food store": "grocery", "organic food store": "grocery",
    "fruit and vegetable store": "grocery", "warehouse store": "grocery", "grocery": "grocery",
}
PLACES_EXCLUDE = re.compile(
    r"veterin|animal|pet |dental|dentist|orthodont|cosmetic|beauty|laser|spa|cannabis|marijuana|dispensary|"
    r"chiropract|acupunct|massage|tattoo|botox|weight loss|abortion|plastic surg|liquor|convenience",
    re.I,
)

# OSM tag -> class (stand-in)
OSM_TAGS = {
    ("amenity", "clinic"): "clinic", ("amenity", "doctors"): "clinic", ("amenity", "hospital"): "clinic",
    ("healthcare", "clinic"): "clinic", ("healthcare", "doctor"): "clinic", ("healthcare", "hospital"): "clinic",
    ("amenity", "pharmacy"): "pharmacy", ("healthcare", "pharmacy"): "pharmacy",
    ("shop", "supermarket"): "grocery", ("shop", "grocery"): "grocery", ("shop", "greengrocer"): "grocery",
}


# ---------------------------------------------------------------- opening hours
_DAYS = ["mo", "tu", "we", "th", "fr", "sa", "su"]


def _hm(s: str) -> float | None:
    """'9', '9:30', '09:30', '9 AM', '9:30 pm', 'noon' -> hours float."""
    s = s.strip().lower().replace(".", "")
    if s in ("noon",):
        return 12.0
    if s in ("midnight",):
        return 24.0
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|a|p)?", s)
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap:
        if ap.startswith("p") and h != 12:
            h += 12
        if ap.startswith("a") and h == 12:
            h = 0
    return h + mi / 60.0


def _ranges(text: str) -> list[tuple[float, float]] | None:
    """Parse '9 AM–5 PM, 6–9 PM' / '09:00-17:00,18:00-20:00' / 'Open 24 hours' / 'Closed'."""
    t = text.strip().lower().replace(" ", " ").replace(" ", " ")
    if not t:
        return None
    if "24 hours" in t or t in ("24/7", "00:00-24:00", "open 24 hours"):
        return [(0.0, 24.0)]
    if t.startswith("closed") or t == "off":
        return []
    out = []
    for part in re.split(r"[,;]", t):
        part = part.strip()
        if not part:
            continue
        m = re.split(r"\s*(?:–|—|-|to)\s*", part, maxsplit=1)
        if len(m) != 2:
            return None
        a_txt, b_txt = m
        # Google omits AM/PM on the first time when it equals the second: '6–9 PM'
        b_ap = re.search(r"(am|pm)$", b_txt.replace(".", ""))
        if b_ap and not re.search(r"(am|pm)$", a_txt.replace(".", "")):
            a_try = _hm(a_txt + " " + b_ap.group(1))
            b_try = _hm(b_txt)
            if a_try is not None and b_try is not None and a_try > b_try:  # '11–2 PM' -> 11 AM
                a_try = _hm(a_txt + " am")
        else:
            a_try, b_try = _hm(a_txt), _hm(b_txt)
        if a_try is None or b_try is None:
            return None
        if b_try <= a_try:  # past midnight
            b_try += 24.0
        out.append((a_try, b_try))
    return out


def _open_at(ranges: list[tuple[float, float]], hour: float) -> bool:
    return any(a <= hour < b for a, b in ranges)


def parse_places_hours(raw) -> list[tuple[float, float]] | None:
    """Tuesday ranges from a Places `open_hours` cell, or None if missing/unparseable.

    Handles JSON dicts ({"Tuesday": "9 AM–5 PM"} or {"Tuesday": ["9AM-5PM"]}), JSON lists of
    "Tuesday: ..." strings, and flat "Monday: ...; Tuesday: ..." / "Tuesday 9AM-5PM | ..." strings.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "[]", "{}"):
        return None
    obj = None
    if s[0] in "[{":
        try:
            obj = json.loads(s)
        except Exception:
            try:
                obj = json.loads(s.replace("'", '"'))
            except Exception:
                obj = None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).strip().lower().startswith("tu"):
                if v is None:
                    return None
                v = ", ".join(map(str, v)) if isinstance(v, list) else str(v)
                return _ranges(v.replace("\n", ", "))
        return None
    if isinstance(obj, list):
        s = " | ".join(map(str, obj))
    m = re.search(r"tuesday\s*[:\-]?\s*([^|;\n\]]+?)(?=\s*(?:\||;|\n|wednesday|thursday|friday|saturday|sunday|monday|$))", s, re.I)
    if not m:
        return None
    return _ranges(m.group(1).strip(" ,'\""))


def parse_osm_hours(raw) -> list[tuple[float, float]] | None:
    """Tuesday ranges from a simple OSM opening_hours value, None if unsupported."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s == "24/7":
        return [(0.0, 24.0)]
    result = None
    for rule in s.split(";"):
        rule = rule.strip()
        if not rule:
            continue
        if re.search(r"\b(PH|SH|week|[A-Z][a-z]{2} \d|sunrise|sunset)\b", rule):
            if result is None:
                continue
        m = re.fullmatch(r"((?:[A-Za-z]{2}(?:-[A-Za-z]{2})?)(?:,(?:[A-Za-z]{2}(?:-[A-Za-z]{2})?))*)?\s*(.*)", rule)
        if not m:
            return None
        days_txt, times = m.group(1), m.group(2).strip()
        days = set(_DAYS)
        if days_txt:
            days = set()
            for d in days_txt.lower().split(","):
                if "-" in d:
                    a, b = d.split("-")
                    if a not in _DAYS or b not in _DAYS:
                        return None
                    i, j = _DAYS.index(a), _DAYS.index(b)
                    days.update(_DAYS[i: j + 1] if i <= j else _DAYS[i:] + _DAYS[: j + 1])
                elif d in _DAYS:
                    days.add(d)
                else:
                    return None
        if "tu" not in days:
            continue
        if times in ("off", "closed"):
            result = []
            continue
        r = _ranges(times)
        if r is None:
            return None
        result = r  # later rules override earlier ones (OSM semantics)
    return result


HOSPITAL = re.compile(r"hospital|emergency", re.I)  # hospitals default to 24h (ER), not clinic hours


def add_open_flags(df: pd.DataFrame, ranges: pd.Series, default_hours: dict) -> pd.DataFrame:
    df = df.copy()
    df["hours_default"] = ranges.isna().to_numpy()
    hosp = df["raw_category"].astype(str).str.contains(HOSPITAL).to_numpy()
    for dep, hour in DEPARTS.items():
        col = "open_" + dep.replace("weekday_", "wd")
        vals = []
        for i, (cls, rng) in enumerate(zip(df["cls"], ranges)):
            if rng is None or (isinstance(rng, float) and np.isnan(rng)):
                a, b = (0, 24) if hosp[i] else default_hours[cls]
                vals.append(a <= hour < b)
            else:
                vals.append(_open_at(rng, hour))
        df[col] = np.array(vals, dtype=bool)
    return df


# ---------------------------------------------------------------- loaders
def map_places_category(category, all_categories=None) -> str | None:
    """Map on the PRIMARY category only: secondary categories are noisy (a psychiatrist listing
    carries 'Mental Hospital', a liquor store 'Grocery store')."""
    c = str(category or "").strip().lower()
    if not c or PLACES_EXCLUDE.search(c):
        return None
    return PLACES_CATEGORY_MAP.get(c)


def load_places(csv_path: Path, default_hours: dict, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """bbox = (min_lon, min_lat, max_lon, max_lat); county clip happens in the caller (region polygon)."""
    use = ["place_id", "name", "category", "all_categories", "lat", "lon", "open_hours",
           "permanently_closed", "temporarily_closed"]
    df = pd.read_csv(csv_path, usecols=lambda c: c in use, low_memory=False)
    df = df[df["lon"].between(bbox[0], bbox[2]) & df["lat"].between(bbox[1], bbox[3])]
    for c in ("permanently_closed", "temporarily_closed"):
        if c in df:
            df = df[~df[c].astype(str).str.lower().isin(["true", "1", "yes"])]
    df["cls"] = [map_places_category(a, b) for a, b in zip(df["category"], df.get("all_categories"))]
    df = df[df["cls"].notna()].copy()
    out = pd.DataFrame({
        "poi_id": "places_" + df["place_id"].astype(str),
        "source": "places",
        "name": df["name"].astype(str),
        "cls": df["cls"],
        "raw_category": df["category"].astype(str),
        "lat": df["lat"].astype(float),
        "lon": df["lon"].astype(float),
    })
    ranges = pd.Series([parse_places_hours(x) for x in df["open_hours"]], dtype=object)
    return add_open_flags(out.reset_index(drop=True), ranges, default_hours)


def load_pois(cfg: dict) -> pd.DataFrame:
    from .config import p

    src = cfg["pois"]["source"]
    places_pq, osm_pq = p(cfg, "places_pois"), p(cfg, "osm_pois")
    if src == "places" or (src == "auto" and places_pq.exists()):
        df = pd.read_parquet(places_pq)
    elif src in ("osm", "auto") and osm_pq.exists():
        df = pd.read_parquet(osm_pq)
    else:
        raise FileNotFoundError(f"no POI parquet: run etl/sim/extract_osm.py or etl/sim/places_pois.py ({places_pq}, {osm_pq})")
    return df
