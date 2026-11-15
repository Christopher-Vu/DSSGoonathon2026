"""Patient sim unit tests. Fast tests use synthetic inputs; `test_fixture_run_*` runs the full pipeline with
the euclid backend when the processed PUMS/blocks/POI inputs exist (skipped otherwise)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import pytest

from sim.patient import access, pums
from sim.patient.config import load_config, p
from sim.patient.fixtures import make_fixture
from sim.patient.pois import CLASSES, add_open_flags, parse_osm_hours, parse_places_hours
from sim.patient.run import CONTRACT_COLS, attach_times, expand

SF_BBOX = (-122.52, 37.70, -122.35, 37.84)

CONTRACT_TYPES = {
    "household_id": "str", "scenario": "str", "age_band": "str", "income_band": "str", "lang": "str",
    "has_car": "bool", "disability": "bool", "dest_puma": "str", "dest_lat": "float64", "dest_lon": "float64",
    "mode": "str", "depart": "str",
    **{f"t_{c}_{s}": "float32" for c in CLASSES for s in ("before", "after")},
    **{f"acc30_{c}_{s}": "bool" for c in CLASSES for s in ("before", "after")},
}


def check_contract(res: pd.DataFrame):
    assert list(res.columns[: len(CONTRACT_COLS)]) == CONTRACT_COLS
    for c, t in CONTRACT_TYPES.items():
        if t == "str":
            assert res[c].map(type).eq(str).all(), c
        else:
            assert str(res[c].dtype) == t, (c, res[c].dtype)
    assert set(res["mode"]) <= {"transit_walk", "car"}
    assert set(res["depart"]) <= {"weekday_1000", "weekday_1800"}
    assert res["dest_puma"].str.fullmatch(r"06\d{5}").all()
    assert res["household_id"].is_unique or res.groupby("scenario")["household_id"].apply(lambda s: s.is_unique).all()
    for c in CLASSES:
        for s in ("before", "after"):
            t = res[f"t_{c}_{s}"].to_numpy()
            assert not np.isnan(t).any(), f"NaN in t_{c}_{s}: unreachable must be inf"
            assert ((t <= 120) | np.isinf(t)).all()
            assert (res[f"acc30_{c}_{s}"].to_numpy() == (t <= 30)).all()


# ---------------------------------------------------------------- hours parsing
@pytest.mark.parametrize("raw,at10,at18", [
    ('{"Tuesday": "9 AM–5 PM"}', True, False),
    ('{"Tuesday": ["8AM-10PM"]}', True, True),
    ("Monday: Closed | Tuesday: 6–9 PM | Wednesday: 9 AM–5 PM", False, True),
    ('{"Tuesday": "Open 24 hours"}', True, True),
    ('{"Tuesday": "Closed"}', False, False),
])
def test_places_hours(raw, at10, at18):
    r = parse_places_hours(raw)
    assert r is not None
    assert any(a <= 10 < b for a, b in r) == at10
    assert any(a <= 18 < b for a, b in r) == at18


def test_places_hours_missing():
    assert parse_places_hours(None) is None
    assert parse_places_hours(float("nan")) is None
    assert parse_places_hours("") is None


@pytest.mark.parametrize("raw,at10,at18", [
    ("Mo-Fr 09:00-17:00", True, False),
    ("Mo-Su 08:00-22:00", True, True),
    ("24/7", True, True),
    ("Mo-Fr 09:00-17:00; Tu 12:00-20:00", False, True),
    ("Sa-Su 10:00-16:00", False, False),
])
def test_osm_hours(raw, at10, at18):
    r = parse_osm_hours(raw)
    r = r or []
    assert any(a <= 10 < b for a, b in r) == at10
    assert any(a <= 18 < b for a, b in r) == at18


def test_default_hours_and_hospital_24h():
    df = pd.DataFrame({"cls": ["clinic", "clinic", "pharmacy"], "raw_category": ["amenity=doctors", "amenity=hospital", "pharmacy"]})
    out = add_open_flags(df, pd.Series([None, None, None], dtype=object),
                         {"clinic": [9, 17], "pharmacy": [9, 21], "grocery": [8, 22]})
    assert out["hours_default"].all()
    assert out["open_wd1000"].tolist() == [True, True, True]
    assert out["open_wd1800"].tolist() == [False, True, True]


# ---------------------------------------------------------------- routing semantics
def _toy_pois():
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame({"poi_id": [f"p{i}" for i in range(n)], "cls": np.tile(CLASSES, n // 3),
                       "raw_category": "x", "lat": rng.uniform(37.6, 37.9, n), "lon": rng.uniform(-122.5, -122.1, n)})
    return add_open_flags(df, pd.Series([None] * n, dtype=object),
                          {"clinic": [9, 17], "pharmacy": [9, 21], "grocery": [8, 22]})


def _toy_sim(n=40, seed=2):
    rng = np.random.default_rng(seed)
    sim = pd.DataFrame({
        "origin_lat": rng.uniform(*SF_BBOX[1::2], n), "origin_lon": rng.uniform(*SF_BBOX[0::2], n),
        "dest_lat": rng.uniform(37.3, 38.3, n), "dest_lon": rng.uniform(-122.6, -121.5, n),
        "has_car": rng.random(n) < 0.5, "depart": rng.choice(["weekday_1000", "weekday_1800"], n),
    })
    sim["cell_before"] = [h3.latlng_to_cell(a, b, 9) for a, b in zip(sim.origin_lat, sim.origin_lon)]
    sim["cell_after"] = [h3.latlng_to_cell(a, b, 9) for a, b in zip(sim.dest_lat, sim.dest_lon)]
    return sim


def _rc():
    return load_config("configs/sim.yaml")["routing"] | {"backend": "euclid"}


def _needs(sim):
    rows = []
    for side in ("cell_before", "cell_after"):
        rows.append(sim[[side, "depart"]].rename(columns={side: "cell"}).assign(mode="transit_walk"))
        rows.append(sim[[side, "depart"]].rename(columns={side: "cell"}).assign(mode="car"))
    return pd.concat(rows).drop_duplicates()


def test_car_never_slower_than_transit():
    sim, pois = _toy_sim(), _toy_pois()
    tab = access.access_table(_needs(sim), pois, _rc())
    with_car = attach_times(sim.assign(has_car=True), tab, 120.0)
    no_car = attach_times(sim.assign(has_car=False), tab, 120.0)
    for c in CLASSES:
        for s in ("before", "after"):
            assert (with_car[f"t_{c}_{s}"] <= no_car[f"t_{c}_{s}"]).all()


def test_before_computed_from_sf_origin():
    sim, pois = _toy_sim(), _toy_pois()
    tab = access.access_table(_needs(sim), pois, _rc())
    out = attach_times(sim, tab, 120.0)
    # before cells are the SF origins, after cells are destinations
    assert (out["cell_before"] == [h3.latlng_to_cell(a, b, 9) for a, b in zip(sim.origin_lat, sim.origin_lon)]).all()
    lat = np.array([h3.cell_to_latlng(c)[0] for c in out["cell_before"]])
    lon = np.array([h3.cell_to_latlng(c)[1] for c in out["cell_before"]])
    assert ((lat > SF_BBOX[1] - 0.01) & (lat < SF_BBOX[3] + 0.01) & (lon > SF_BBOX[0] - 0.01) & (lon < SF_BBOX[2] + 0.01)).all()
    # recompute one household's before time directly from its origin cell
    i = 0
    row = tab[(tab.cell == out.cell_before[i]) & (tab.depart == out.depart[i]) & (tab["mode"] == "transit_walk")]
    exp = float(row["t_clinic"].iloc[0])
    if not out.has_car[i]:
        assert out["t_clinic_before"][i] == pytest.approx(exp if exp <= 120 else np.inf)


def test_unreachable_is_inf_not_nan():
    sim = _toy_sim()
    far = pd.DataFrame({"poi_id": ["far"], "cls": ["clinic"], "raw_category": "x", "lat": [34.0], "lon": [-118.2]})
    far = add_open_flags(far, pd.Series([None], dtype=object), {"clinic": [0, 24]})
    tab = access.access_table(_needs(sim), far, _rc())
    out = attach_times(sim, tab, 120.0)
    for c in CLASSES:
        for s in ("before", "after"):
            assert np.isinf(out[f"t_{c}_{s}"]).all()
            assert not out[f"acc30_{c}_{s}"].any()


def test_expand_makes_synthetic_ids():
    fx = make_fixture(100, 0)
    out = expand(fx, 300, np.random.default_rng(0))
    for _, df in out.groupby("scenario"):
        assert len(df) == 300
        assert df["household_id"].str.startswith("syn_").all()
        assert df["origin_household_id"].isin(fx["household_id"]).all()
    assert out.groupby("scenario")["weight"].sum().round(2).to_dict() == fx.groupby("scenario")["weight"].sum().round(2).to_dict()


def test_migration_smoothing_and_support():
    cfg = load_config("configs/sim.yaml")
    mv = pd.DataFrame({"year": [2022] * 4, "puma": ["00101", "00101", "00102", "07507"],
                       "w": [10.0, 10.0, 5.0, 50.0], "income_band": ["<35k", "<35k", "150k+", "<35k"],
                       "in_sf": [False, False, False, True]})
    m = pums.fit_migration(mv, {"00101", "00102", "00103", "07507"}, cfg, bands=["<35k", "150k+", "35-75k"])
    assert "07507" not in m.pumas  # SF excluded
    for pr in m.probs.values():
        assert pr.sum() == pytest.approx(1.0) and (pr > 0).all()
    assert m.probs["<35k"][m.pumas.index("00101")] > m.probs["150k+"][m.pumas.index("00101")]
    assert m.stats["share_within_sf"] == pytest.approx(50 / 75)


# ---------------------------------------------------------------- full pipeline on the fixture
def _inputs_ready(cfg):
    inter = p(cfg, "interim")
    return all((inter / f).exists() for f in ("pums_households.parquet", "pums_persons.parquet", "blocks.parquet")) \
        and p(cfg, "osm_pois").exists() and p(cfg, "puma_shapes").exists()


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory):
    from sim.patient.run import run

    cfg = load_config("configs/sim_fixture.yaml")
    if not _inputs_ready(cfg):
        pytest.skip("processed sim inputs missing (run etl/sim first)")
    cfg["routing"]["backend"] = "euclid"
    cfg["min_households_per_scenario"] = 300
    d = tmp_path_factory.mktemp("fx")
    hh = d / "hh.parquet"
    make_fixture(200, 0).to_parquet(hh)
    cfg["paths"]["interim"] = str(p(cfg, "interim"))
    return run(hh, cfg)


def test_fixture_run_contract(fixture_run):
    res, summary = fixture_run
    check_contract(res)
    assert res.groupby("scenario").size().min() >= 300
    json.dumps(summary, default=str)
    assert set(summary["scenarios"]) == set(res["scenario"])


def test_fixture_run_before_in_sf(fixture_run):
    res, _ = fixture_run
    assert res["origin_puma"].str[2:].isin(pums.SF_PUMAS).all()
    assert not res["dest_puma"].str[2:].isin(pums.SF_PUMAS).any()
