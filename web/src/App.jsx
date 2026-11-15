import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { FlyToInterpolator } from "@deck.gl/core";
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import { ArcLayer, ScatterplotLayer } from "@deck.gl/layers";
import { Map } from "react-map-gl/maplibre";
import { cellToLatLng } from "h3-js";
import "maplibre-gl/dist/maplibre-gl.css";
import { Bars, Fan, Hist, LineChart, fmt, useCountUp } from "./charts.jsx";

const BASEMAP = "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json";
const SF = { longitude: -122.44, latitude: 37.758 };
const C = { model: [57, 135, 229], city: [217, 89, 38], event: [230, 103, 103], ink: [240, 239, 236] };
const RAMP = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]
  .map((h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)));
const ramp = (t) => RAMP[Math.max(0, Math.min(RAMP.length - 1, Math.round(t * (RAMP.length - 1))))];

const VIEWS = {
  hero: { ...SF, zoom: 11.7, pitch: 58, bearing: -18 },
  wm: { ...SF, zoom: 11.9, pitch: 0, bearing: 0 },
  frozen: { longitude: -122.425, latitude: 37.765, zoom: 12.4, pitch: 52, bearing: -24 },
  budget: { ...SF, zoom: 11.9, pitch: 45, bearing: 12 },
  futures: { ...SF, zoom: 12, pitch: 30, bearing: 0 },
  sim: { longitude: -122.15, latitude: 37.75, zoom: 8.9, pitch: 42, bearing: 8 },
  honesty: { ...SF, zoom: 11.6, pitch: 0, bearing: 0 },
};
// on wide screens the story panel covers the left ~40%: shift the camera so the city sits in the open area
const WIDE = new Set(["wm", "sim", "honesty"]);
const framed = (v, id) => {
  if (typeof window === "undefined" || window.innerWidth <= 720) return v;
  const k = WIDE.has(id) && window.innerWidth > 1000 ? 0.34 : 0.2;
  return { ...v, longitude: v.longitude - (k * window.innerWidth * 360) / (512 * 2 ** v.zoom) };
};
const SECTIONS = ["hero", "wm", "frozen", "budget", "futures", "sim", "honesty"];
const FILES = ["headline", "sweep", "fairness", "futures", "risk_map", "wm", "sim"];

function useData() {
  const [d, setD] = useState({});
  useEffect(() => {
    FILES.forEach((f) =>
      fetch(`/data/${f}.json`).then((r) => (r.ok ? r.json() : null)).then((j) => j && setD((o) => ({ ...o, [f]: j }))).catch(() => {}));
  }, []);
  return d;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export default function App() {
  const d = useData();
  const [active, setActive] = useState("hero");
  const [view, setView] = useState(framed(VIEWS.hero));
  const [mult, setMult] = useState(1);
  const [year, setYear] = useState(2018);
  const [playing, setPlaying] = useState(false);
  const [wmSel, setWmSel] = useState(null);
  const [wm16, setWm16] = useState(false);
  const [futSel, setFutSel] = useState(null);
  const [scen, setScen] = useState("realized_2019_2024");
  const [cls, setCls] = useState("clinic");
  const [caption, setCaption] = useState(null);
  const [tick, setTick] = useState(0);
  const [hover, setHover] = useState(null);
  const presenting = useRef(false);
  const refs = useRef({});

  // ---------------------------------------------------------------- scroll -> scene
  useEffect(() => {
    const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setActive(e.target.dataset.id)),
      { rootMargin: "-45% 0px -45% 0px" });
    Object.values(refs.current).forEach((el) => el && io.observe(el));
    return () => io.disconnect();
  }, [d.headline]);
  useEffect(() => {
    setView((v) => ({ ...framed(VIEWS[active], active), transitionDuration: 2200, transitionInterpolator: new FlyToInterpolator({ speed: 1.4 }) }));
  }, [active]);
  // hero: slow orbit
  useEffect(() => {
    if (active !== "hero") return;
    let raf, t0;
    const spin = (t) => {
      t0 = t0 ?? t;
      if (t - t0 > 2300) setView((v) => ({ ...v, bearing: (v.bearing ?? 0) + 0.04, transitionDuration: 0 }));
      raf = requestAnimationFrame(spin);
    };
    raf = requestAnimationFrame(spin);
    return () => cancelAnimationFrame(raf);
  }, [active]);
  // pulse clock
  useEffect(() => {
    let raf;
    const f = () => { setTick((t) => (t + 1) % 100000); raf = requestAnimationFrame(f); };
    if (active === "frozen" || active === "sim") raf = requestAnimationFrame(f);
    return () => cancelAnimationFrame(raf);
  }, [active]);
  // time-lapse
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setYear((y) => { if (y >= 2024) { setPlaying(false); return 2024; } return y + 1; }), 1300);
    return () => clearInterval(id);
  }, [playing]);

  // ---------------------------------------------------------------- derived data
  const risk = useMemo(() => {
    if (!d.risk_map) return null;
    const hs = d.risk_map.hexes.map((x) => ({ ...x, d: x.risk / x.n }));
    const vals = hs.map((x) => x.risk).sort((a, b) => a - b);
    const q = (v) => { let lo = 0, hi = vals.length; while (lo < hi) { const m = (lo + hi) >> 1; vals[m] < v ? (lo = m + 1) : (hi = m); } return lo / vals.length; };
    hs.forEach((x) => (x.q = q(x.risk)));
    const ev = {};
    d.risk_map.events.forEach(([h, y, kind, n]) => {
      if (kind === "other") return;
      (ev[y] = ev[y] || {})[h] = (ev[y][h] || 0) + n;
    });
    const pts = {};
    Object.entries(ev).forEach(([y, m]) => (pts[y] = Object.entries(m).map(([h, n]) => ({ h, n, p: cellToLatLng(h) }))));
    return { hs, pts };
  }, [d.risk_map]);

  const sweepRow = useMemo(() => d.sweep && d.sweep.rows.reduce((b, r) => (Math.abs(r.mult - mult) < Math.abs(b.mult - mult) ? r : b)), [d.sweep, mult]);

  const wmCells = useMemo(() => {
    if (!d.wm) return null;
    const cells = d.wm.cells;
    let sim = null;
    if (wmSel) {
      const s = cells.find((c) => c.h === wmSel);
      if (s) sim = cells.map((c) => c.v.reduce((a, x, i) => a + x * s.v[i], 0) / (127 * 127));
    }
    return { cells, sim };
  }, [d.wm, wmSel]);

  const futHex = useMemo(() => {
    if (!d.futures) return null;
    const hs = Object.entries(d.futures.hex).map(([h, f]) => ({ h, med: f.q["50"][f.q["50"].length - 1], f }));
    const mx = Math.max(...hs.map((x) => x.med)) || 1;
    return { hs, mx };
  }, [d.futures]);

  // ---------------------------------------------------------------- layers per scene
  const layers = useMemo(() => {
    const L = [];
    const pulse = (tick % 90) / 90;
    if ((active === "hero" || active === "frozen" || active === "honesty") && risk) {
      L.push(new H3HexagonLayer({
        id: "risk", data: risk.hs, getHexagon: (x) => x.h, extruded: active !== "honesty", elevationScale: 1,
        getElevation: (x) => 30 + x.q * x.q * 1400, coverage: 0.92,
        getFillColor: (x) => [...ramp(x.q), active === "honesty" ? 90 : 230], pickable: true,
        material: { ambient: 0.5, diffuse: 0.6, shininess: 20 },
        transitions: { getElevation: 900 }, updateTriggers: { getFillColor: active, getElevation: active },
        onHover: (i) => setHover(i.object ? { x: i.x, y: i.y, lines: [`Expected households displaced / yr: ${fmt(i.object.risk, 2)}`, `${i.object.n} rent-controlled buildings`] } : null),
      }));
    }
    if (active === "frozen" && risk) {
      for (let y = 2019; y <= year; y++) {
        const cur = y === year;
        L.push(new ScatterplotLayer({
          id: `ev${y}`, data: risk.pts[y] || [], getPosition: (x) => [x.p[1], x.p[0], cur ? 1500 : 1200],
          getRadius: (x) => Math.sqrt(x.n) * (cur ? 40 + 50 * pulse : 34), radiusUnits: "meters",
          getFillColor: [...C.event, cur ? Math.round(220 * (1 - pulse * 0.6)) : 70], stroked: false,
          updateTriggers: { getRadius: [pulse, year], getFillColor: [pulse, year] }, parameters: { depthTest: false },
        }));
      }
    }
    if (active === "wm" && wmCells) {
      const { cells, sim } = wmCells;
      L.push(new H3HexagonLayer({
        id: "wm", data: cells, getHexagon: (c) => c.h, extruded: !!sim, coverage: 0.94, pickable: true,
        getFillColor: (c, { index }) => {
          const rgb = (wm16 && c.rgb16) || c.rgb;
          if (!sim) return [...rgb, 235];
          const s = Math.max(0, sim[index]);
          return [...rgb, c.h === wmSel ? 255 : Math.round(30 + 225 * Math.pow(s, 3))];
        },
        getElevation: (c, { index }) => (sim ? Math.pow(Math.max(0, sim[index]), 4) * 900 : 0),
        updateTriggers: { getFillColor: [wmSel, wm16], getElevation: wmSel },
        transitions: { getFillColor: 700, getElevation: 700 },
        onClick: (i) => i.object && setWmSel(i.object.h === wmSel ? null : i.object.h),
        onHover: (i) => setHover(i.object ? { x: i.x, y: i.y, lines: [wmSel ? "Click another hexagon, or the same one to clear" : "Click: find hexagons the model thinks are similar"] } : null),
      }));
    }
    if (active === "budget" && d.sweep && sweepRow) {
      const pk = d.sweep.picks[String(sweepRow.mult)] || { model: {}, city_policy: {} };
      const base = Object.entries(d.sweep.hex8).map(([h]) => ({ h, m: pk.model[h] || 0, c: pk.city_policy[h] || 0 }));
      L.push(new H3HexagonLayer({
        id: "elig", data: base, getHexagon: (x) => x.h, extruded: false, filled: true, stroked: true, coverage: 0.96,
        getFillColor: [255, 255, 255, 10], getLineColor: [255, 255, 255, 40], lineWidthMinPixels: 1,
      }));
      L.push(new H3HexagonLayer({
        id: "city", data: base.filter((x) => x.c), getHexagon: (x) => x.h, extruded: true, coverage: 0.42,
        getElevation: (x) => 120 + x.c * 260, getFillColor: [...C.city, 235], transitions: { getElevation: 600 },
        updateTriggers: { getElevation: sweepRow.mult }, pickable: true,
        onHover: (i) => setHover(i.object ? { x: i.x, y: i.y, lines: [`City-policy picks here: ${i.object.c}`, `Model picks here: ${i.object.m}`] } : null),
      }));
      L.push(new H3HexagonLayer({
        id: "model", data: base.filter((x) => x.m), getHexagon: (x) => x.h, extruded: true, coverage: 0.78,
        getElevation: (x) => 120 + x.m * 260, getFillColor: [...C.model, 220], transitions: { getElevation: 600 },
        updateTriggers: { getElevation: sweepRow.mult }, pickable: true,
        onHover: (i) => setHover(i.object ? { x: i.x, y: i.y, lines: [`Model picks here: ${i.object.m}`, `City-policy picks here: ${i.object.c}`] } : null),
      }));
    }
    if (active === "futures" && futHex) {
      L.push(new H3HexagonLayer({
        id: "fut", data: futHex.hs, getHexagon: (x) => x.h, extruded: true, coverage: 0.9, pickable: true,
        getElevation: (x) => (x.h === futSel ? 1400 : 40 + (x.med / futHex.mx) * 700),
        getFillColor: (x) => (x.h === futSel ? [...C.ink, 255] : [...ramp(Math.sqrt(x.med / futHex.mx)), 220]),
        updateTriggers: { getElevation: futSel, getFillColor: futSel }, transitions: { getElevation: 500 },
        onClick: (i) => i.object && setFutSel(i.object.h),
        onHover: (i) => setHover(i.object ? { x: i.x, y: i.y, lines: [`Median forecast 2019–24: ${fmt(i.object.med, 1)} households`, "Click for its 1,000 futures"] } : null),
      }));
    }
    if (active === "sim" && d.sim) {
      const sc = d.sim.scenarios[scen];
      if (sc) {
        const mx = Math.max(...sc.flows.map((f) => f[3]));
        L.push(new ArcLayer({
          id: "arcs", data: sc.flows, getSourcePosition: (f) => [f[1][1], f[1][0]], getTargetPosition: (f) => [f[2][1], f[2][0]],
          getSourceColor: [...C.model, 200], getTargetColor: [...C.city, 230], getWidth: (f) => 0.6 + 5 * Math.sqrt(f[3] / mx),
          getHeight: 0.6, greatCircle: false, pickable: true,
          onHover: (i) => setHover(i.object ? { x: i.x, y: i.y, lines: [`${i.object[3]} simulated households`] } : null),
        }));
        L.push(new ScatterplotLayer({
          id: "dest", data: sc.flows, getPosition: (f) => [f[2][1], f[2][0]], getRadius: (f) => 300 + 2200 * Math.sqrt(f[3] / mx) * (0.6 + 0.4 * pulse),
          getFillColor: [...C.city, 60], updateTriggers: { getRadius: pulse },
        }));
      }
    }
    return L;
  }, [active, risk, year, tick, wmCells, wm16, wmSel, d.sweep, sweepRow, futHex, futSel, d.sim, scen]);

  // ---------------------------------------------------------------- presentation mode (press P)
  const go = (id) => refs.current[id]?.scrollIntoView({ behavior: "smooth", block: id === "hero" ? "end" : "start" });
  const present = useCallback(async () => {
    if (presenting.current) return;
    presenting.current = true;
    const cap = (t) => setCaption(t);
    const tour = [
      ["hero", 7000, null, "A world model of San Francisco, frozen in 2018."],
      ["wm", 4500, null, "It learned the city's neighborhoods without ever seeing a map."],
      ["wm", 5000, () => setWmSel(pickMission(d.wm)), "Pick one block. It finds every block like it."],
      ["frozen", 3000, () => { setYear(2018); }, "From 2018 data alone, it forecasts where tenants will be displaced."],
      ["frozen", 9000, () => setPlaying(true), "Then 2019–2024 happened."],
      ["budget", 3500, () => setMult(1), "Same budget as the city's Small Sites purchases."],
      ["budget", 8000, () => animate(setMult, 1, 5, 7000), "More households protected at every budget."],
      ["futures", 7000, () => setFutSel(topHex(futHex)), "Every forecast is a distribution: 1,000 sampled futures."],
      ["sim", 9000, null, "Displacement has a cost: where people go, and how far care becomes."],
      ["honesty", 6000, null, "Hexagons only. No building list. Fairness checked."],
    ];
    for (const [id, ms, act, text] of tour) {
      if (!presenting.current) break;
      go(id);
      cap(text);
      await sleep(900);
      act && act();
      await sleep(ms);
    }
    cap(null);
    presenting.current = false;
  }, [d.wm, futHex]);
  useEffect(() => {
    const k = (e) => {
      if (e.key === "p" || e.key === "P") present();
      if (e.key === "Escape") { presenting.current = false; setCaption(null); }
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [present]);

  // URL state
  useEffect(() => {
    const q = new URLSearchParams(location.search);
    if (q.get("b")) setMult(+q.get("b"));
    const v = q.get("view");
    if (v && SECTIONS.includes(v)) setTimeout(() => go(v), 600);
  }, []);
  useEffect(() => {
    const q = new URLSearchParams({ view: active, ...(active === "budget" ? { b: mult } : {}) });
    history.replaceState(null, "", `?${q}`);
  }, [active, mult]);

  const h = d.headline;
  if (!h) return <div className="loading">Loading the city…</div>;
  const reg = (id) => (el) => (refs.current[id] = el);

  return (
    <>
      <div className="map-layer">
        <DeckGL viewState={view} onViewStateChange={({ viewState }) => setView(viewState)}
          controller={{ scrollZoom: false, dragRotate: true }} layers={layers}
          getCursor={({ isHovering }) => (isHovering ? "pointer" : "grab")}>
          <Map mapStyle={BASEMAP} reuseMaps attributionControl={{ compact: true }} />
        </DeckGL>
      </div>
      <div className="vignette" />
      <nav className="nav">{SECTIONS.map((s) => <a key={s} href={`#${s}`} className={s === active ? "on" : ""} title={s}
        onClick={(e) => { e.preventDefault(); go(s); }} />)}</nav>
      {caption && <div className="caption">{caption}</div>}
      {hover && <div className="tip" style={{ left: hover.x + 14, top: hover.y + 14 }}>{hover.lines.map((l, i) => <div key={i}>{l}</div>)}</div>}

      <main className="story">
        <Hero refEl={reg("hero")} h={h} sim={d.sim} wm={d.wm} />
        <WM refEl={reg("wm")} wm={d.wm} sel={wmSel} setSel={setWmSel} wm16={wm16} setWm16={setWm16} />
        <Frozen refEl={reg("frozen")} h={h} rm={d.risk_map} year={year} setYear={setYear} playing={playing} setPlaying={setPlaying} />
        <Budget refEl={reg("budget")} sw={d.sweep} row={sweepRow} mult={mult} setMult={setMult} h={h} />
        <Futures refEl={reg("futures")} fu={d.futures} sel={futSel} setSel={setFutSel} futHex={futHex} />
        <Sim refEl={reg("sim")} sim={d.sim} scen={scen} setScen={setScen} cls={cls} setCls={setCls} />
        <Honesty refEl={reg("honesty")} fair={d.fairness} h={h} sw={d.sweep} />
      </main>
    </>
  );
}

function animate(set, a, b, ms) {
  const t0 = performance.now();
  const f = (t) => {
    const k = Math.min(1, (t - t0) / ms);
    set(+(a + (b - a) * k).toFixed(1));
    if (k < 1) requestAnimationFrame(f);
  };
  requestAnimationFrame(f);
}
function pickMission(wm) {
  if (!wm) return null;
  const target = [37.7599, -122.4148];
  let best = null, bd = 1e9;
  wm.cells.forEach((c) => { const [la, lo] = cellToLatLng(c.h); const dd = (la - target[0]) ** 2 + (lo - target[1]) ** 2; if (dd < bd) { bd = dd; best = c.h; } });
  return best;
}
function topHex(fh) {
  return fh ? fh.hs.reduce((b, x) => (x.med > b.med ? x : b)).h : null;
}

// ================================================================== sections
function Step({ id, refEl, children, cls = "" }) {
  return (
    <section className={`step ${cls}`} data-id={id} ref={refEl} id={id}>
      <div className={`panel${WIDE.has(id) ? " wide" : ""}`}>{children}</div>
    </section>
  );
}

function Hero({ refEl, h, sim, wm }) {
  const x = useCountUp(h.vs_city, 1800);
  const sh = sim?.scenarios?.realized_2019_2024?.delta?.clinic;
  const lift = wm?.probes_final?.mean_lift ?? meanLift(wm?.probes);
  return (
    <Step id="hero" refEl={refEl} cls="hero">
      <p className="kicker">SF World Model · 8th Datathon for Social Good</p>
      <h1>Keeping San Francisco tenants housed.</h1>
      <div className="big">{fmt(x, 2)}<span className="x">×</span></div>
      <p style={{ fontSize: 18, color: "var(--ink)", maxWidth: 560 }}>
        more households protected from displacement than the city's own acquisition strategy, with the same budget,
        forecasting from 2018 and scored on what actually happened in 2019–2024.
      </p>
      <div className="stats">
        <div className="stat"><b>{fmt(h.vs_random, 1)}×</b><span>vs. random buildings at the same budget</span></div>
        {lift != null && <div className="stat"><b>+{fmt(lift, 2)}</b><span>world-model lift over hand-made features (probe R²/PR-AUC)</span></div>}
        {h.metrics?.capture_top10pct && <div className="stat"><b>{fmt(h.metrics.capture_top10pct.model * 100, 0)}%</b><span>of 2019–24 displaced households were in the model's top 10% of buildings</span></div>}
        {sh && <div className="stat"><b>{sh.median_delta >= 0 ? "+" : ""}{fmt(sh.median_delta, 1)} min</b><span>median change in travel to the nearest clinic after displacement; {fmt(sh.pct_lose_30, 0)}% lose 30-minute access</span></div>}
      </div>
      <p className="scroll-hint">Scroll to explore · press <b>P</b> for the guided tour</p>
    </Step>
  );
}
function meanLift(probes) {
  if (!probes) return null;
  const l = probes.filter((p) => p.features === "emb" && p.lift != null).map((p) => p.lift);
  return l.length ? l.reduce((a, b) => a + b, 0) / l.length : null;
}

function WM({ refEl, wm, sel, setSel, wm16, setWm16 }) {
  if (!wm) return <Step id="wm" refEl={refEl}><h2>The world model</h2><p>Loading…</p></Step>;
  const pf = wm.probes_final;
  const rows = (pf?.targets || probeRows(wm.probes)).map((r) => ({ label: r.target.replace(/_/g, " "), value: r.lift, lo: r.lift_lo, hi: r.lift_hi, color: "var(--model)" }));
  const ls = wm.lambda_sweep;
  return (
    <Step id="wm" refEl={refEl}>
      <p className="kicker">1 · The world model</p>
      <h2>It learned the city without a map</h2>
      <div className="cols"><div>
      <p>A 5M-parameter transformer read every sale, permit, eviction notice and buyout in San Francisco up to 2018,
        as a sequence per hexagon, and learned to fill in what was hidden and predict next year. Each hexagon's color is
        its learned embedding. No neighborhood boundaries were given; they show up anyway.</p>
      <div className="row" style={{ marginBottom: 10 }}>
        <button className="pill" aria-pressed={!wm16} onClick={() => setWm16(false)}>2018 model</button>
        <button className="pill" aria-pressed={wm16} onClick={() => setWm16(true)}>2016 model</button>
        {sel && <button className="pill" onClick={() => setSel(null)}>Clear similarity</button>}
      </div>
      <p className="small">Click any hexagon: everything the model thinks is alike lights up and rises.</p>
      </div><div>
      <p style={{ marginBottom: 4, color: "var(--ink)" }}>The embeddings predict real outcomes better than hand-made features</p>
      <Bars rows={rows} vfmt={(v) => (v >= 0 ? "+" : "") + fmt(v, 2)} />
      <p className="small">Lift in R² (PR-AUC for the binary target) from a linear probe on the embeddings over the same probe on hand-made features, with 95% CI{pf ? `, on held-out neighborhoods (locked test, run once). Mean lift +${fmt(pf.mean_lift, 2)}` : ", validation split"}.</p>
      {ls && <>
        <p style={{ marginTop: 14, marginBottom: 4, color: "var(--ink)" }}>Adding next-year prediction to training helps</p>
        <LineChart xs={ls.map((r) => r.lam)} xfmt={(x) => `${x}`} xlabel="λ" yfmt={(y) => fmt(y, 2)} h={150}
          series={[{ key: "l", label: "probe lift", color: "var(--model)", values: ls.map((r) => r.mean_lift_emb) }]} cursor={wm.lam} />
        <p className="small">λ weights the next-year loss. λ = {wm.lam} was chosen on the 2016 model only, before any test data was read.</p>
      </>}
    </div></div>
    </Step>
  );
}
function probeRows(probes) {
  return (probes || []).filter((p) => p.features === "emb" && p.lift != null)
    .map((p) => ({ target: p.target, lift: p.lift, lift_lo: p.lift_lo, lift_hi: p.lift_hi }));
}

function Frozen({ refEl, h, rm, year, setYear, playing, setPlaying }) {
  const m = h.metrics;
  return (
    <Step id="frozen" refEl={refEl}>
      <p className="kicker">2 · Frozen in 2018</p>
      <h2>Where the model expected displacement</h2>
      <p>Height and brightness: the model's expected households displaced per year, from data through 2018 only.
        Then press play: red rings are the Ellis Act, owner move-in and buyout cases that actually landed, 2019–2024.</p>
      <div className="row">
        <button className="pill" onClick={() => { if (year >= 2024) setYear(2018); setPlaying(!playing); }}>{playing ? "Pause" : "▶ Play 2019–2024"}</button>
        <input type="range" min={2018} max={2024} step={1} value={year} onChange={(e) => { setPlaying(false); setYear(+e.target.value); }} style={{ flex: 1 }} />
        <b style={{ fontVariantNumeric: "tabular-nums", width: 44 }}>{year === 2018 ? "2018" : year}</b>
      </div>
      <div className="legend"><span><i style={{ background: "var(--model)" }} />model risk (2018)</span><span><i style={{ background: "var(--event)", borderRadius: 5 }} />actual cases</span></div>
      {rm && <p><b style={{ color: "var(--ink)" }}>{fmt(rm.top_decile_share * 100, 0)}%</b> of 2019–24 Ellis, owner move-in and buyout cases landed in the model's top 10% of hexagons.</p>}
      {m && <>
        <table style={{ marginTop: 8 }}>
          <thead><tr><th>Test, 2019–24</th><th>World model</th><th>City's selection pattern</th><th>Base rate</th></tr></thead>
          <tbody>
            <tr><td>PR-AUC: buyout at the building</td><td>{fmt(m.building_buyout.model.pr_auc, 3)}</td><td>{fmt(m.building_buyout.city_policy.pr_auc, 3)}</td><td>{fmt(m.building_buyout.prevalence, 3)}</td></tr>
            <tr><td>PR-AUC: any displacement</td><td>{fmt(m.any_displacement.model.pr_auc, 3)}</td><td>{fmt(m.any_displacement.city_policy.pr_auc, 3)}</td><td>{fmt(m.any_displacement.prevalence, 3)}</td></tr>
            <tr><td>Precision@100: any displacement</td><td>{fmt(m.any_displacement.model["p@100"], 2)}</td><td>{fmt(m.any_displacement.city_policy["p@100"], 2)}</td><td>{fmt(m.any_displacement.prevalence, 2)}</td></tr>
            <tr><td>Households in top 10% of buildings</td><td>{fmt(m.capture_top10pct.model * 100, 0)}%</td><td>{fmt(m.capture_top10pct.city_policy * 100, 0)}%</td><td>10%</td></tr>
          </tbody>
        </table>
        <p className="small">All {h.universe_n.toLocaleString()} 5–25-unit rent-controlled buildings, ranked from 2018 data. "Any displacement" = a buyout at the building or an Ellis/owner move-in notice on its block-face, 2019–24. Scored once, by a frozen model.</p>
      </>}
    </Step>
  );
}

function Budget({ refEl, sw, row, mult, setMult, h }) {
  const m = useCountUp(row?.model), c = useCountUp(row?.city_policy), r = useCountUp(row?.random);
  if (!sw || !row) return <Step id="budget" refEl={refEl}><h2>The budget</h2></Step>;
  const rows = sw.rows.filter((x) => x.mult <= 5);
  const B = (row.budget_usd / 1e6);
  return (
    <Step id="budget" refEl={refEl}>
      <p className="kicker">3 · Same money, more people</p>
      <h2>Drag the budget</h2>
      <p>At the end of 2018, with the money the city actually spent on Small Sites acquisitions in 2019–23 (1× = ${fmt(h.budget_1x_usd / 1e6, 1)}M
        in this building universe), which buildings would you buy? Scored by households displaced from them in 2019–24.</p>
      <div className="row"><input type="range" min={0.5} max={5} step={0.5} value={mult} onChange={(e) => setMult(+e.target.value)} />
      </div>
      <div className="row" style={{ justifyContent: "space-between" }}><b>{fmt(row.mult, 1)}× budget</b><span className="small">${fmt(B, 0)}M · model buys {row.model_n} buildings</span></div>
      <div className="counters">
        <div className="counter" style={{ "--c": "var(--model)" }}><b>{fmt(m, 1)}</b><span>World model</span><i>CI {fmt(row.model_ci[0], 0)}–{fmt(row.model_ci[1], 0)}</i></div>
        <div className="counter" style={{ "--c": "var(--city)" }}><b>{fmt(c, 1)}</b><span>City's selection pattern</span><i>CI {fmt(row.city_policy_ci[0], 0)}–{fmt(row.city_policy_ci[1], 0)}</i></div>
        <div className="counter" style={{ "--c": "var(--random)" }}><b>{fmt(r, 1)}</b><span>Random</span><i>p5–p95 {fmt(row.random_p5, 0)}–{fmt(row.random_p95, 0)}</i></div>
      </div>
      <p className="small">Households protected (2019–24). CIs: block-face cluster bootstrap with the picks fixed. 1–5× is the pre-registered range; the full sweep to 20× is in the repo.</p>
      <LineChart xs={rows.map((x) => x.mult)} xfmt={(x) => `${x}×`} cursor={row.mult} h={180}
        series={[
          { key: "m", label: "World model", color: "var(--model)", values: rows.map((x) => x.model), band: rows.map((x) => x.model_ci) },
          { key: "c", label: "City pattern", color: "var(--city)", values: rows.map((x) => x.city_policy) },
          { key: "r", label: "Random", color: "var(--random)", values: rows.map((x) => x.random), dash: "4 4" },
        ]} />
      <div className="legend"><span><i style={{ background: "var(--model)" }} />model picks</span><span><i style={{ background: "var(--city)" }} />city-pattern picks</span><span className="small">count per hexagon (≈0.7 km²), never buildings</span></div>
    </Step>
  );
}

function Futures({ refEl, fu, sel, setSel, futHex }) {
  const [prog, setProg] = useState(1);
  useEffect(() => { setProg(0); const t0 = performance.now(); let raf; const f = (t) => { const k = Math.min(1, (t - t0) / 1600); setProg(k); if (k < 1) raf = requestAnimationFrame(f); }; raf = requestAnimationFrame(f); return () => cancelAnimationFrame(raf); }, [sel]);
  if (!fu) return <Step id="futures" refEl={refEl}><h2>Futures</h2></Step>;
  const f = sel ? fu.hex[sel] : fu.city;
  return (
    <Step id="futures" refEl={refEl}>
      <p className="kicker">4 · 1,000 futures</p>
      <h2>{sel ? "One neighborhood's futures" : "The whole city's futures"}</h2>
      <p>The model doesn't give one number; it gives a distribution. From its 2018 forecast we sample 1,000 possible 2019–24 histories
        (cumulative households displaced), then overlay what actually happened in red.</p>
      <Fan fan={f} years={fu.years} progress={prog} />
      <div className="row">{sel && <button className="pill" onClick={() => setSel(null)}>Back to whole city</button>}
        <span className="small">Click a hexagon on the map to see its fan.</span></div>
      <p className="small" style={{ marginTop: 8 }}>Neighborhood by neighborhood, the actual 2019–24 total fell inside the model's 80% band in {fmt(fu.coverage_80 * 100, 0)}% of {fu.n_hex} hexagons
        (the band's width was set on the 2016 model, before any test data was read).</p>
      {!sel && <p className="small"><b style={{ color: "var(--ink)" }}>Where it missed:</b> citywide, the model expected a steady rate. Actual displacement fell every year,
        from {fmt(f.realized[0], 0)} households in 2019 to {fmt(f.realized[5] - f.realized[4], 0)} in 2024, through the COVID eviction moratoria and a slowdown in buyouts, which no model frozen in 2018 could see.
        It got the <i>where</i> right more than the <i>how much</i>, which is what a purchase decision needs.</p>}
    </Step>
  );
}

const SCEN_LABEL = { realized_2019_2024: "What happened" };
function scenLabel(s) {
  if (SCEN_LABEL[s]) return SCEN_LABEL[s];
  const m = s.match(/unprotected__(.*)_(\d+)x/);
  if (!m) return s;
  return `${m[1] === "ours" ? "If the model's picks were protected" : m[1] === "city_policy" ? "If the city pattern's picks were protected" : m[1]} (${m[2]}×)`;
}

function Sim({ refEl, sim, scen, setScen, cls, setCls }) {
  if (!sim) return <Step id="sim" refEl={refEl}><p className="kicker">5 · What eviction costs</p><h2>Patient simulation</h2><p>Results loading…</p></Step>;
  const sc = sim.scenarios.realized_2019_2024 || Object.values(sim.scenarios)[0];
  const dd = sc.delta[cls];
  const real = sim.scenarios.realized_2019_2024;
  return (
    <Step id="sim" refEl={refEl}>
      <p className="kicker">5 · What eviction costs</p>
      <h2>Displacement moves people away from care</h2>
      <div className="cols"><div>
      <p>{real.n.toLocaleString()} simulated households, built from real Census microdata and matched to real 2019–24 displacement events.
        Each moves where similar renters actually moved (Census migration flows), then we route by transit (or car, if they have one)
        to the nearest open clinic, pharmacy and grocery, before and after.</p>
      <div className="row" style={{ marginBottom: 6 }}>
        {["clinic", "pharmacy", "grocery"].map((c) => <button key={c} className="pill" aria-pressed={cls === c} onClick={() => setCls(c)}>{c}</button>)}
      </div>
      <div className="counters" style={{ gridTemplateColumns: "1fr 1fr" }}>
        <div className="counter" style={{ "--c": "var(--city)" }}><b>{dd.median_delta >= 0 ? "+" : ""}{fmt(dd.median_delta, 1)} min</b><span>median change to nearest {cls}</span></div>
        <div className="counter" style={{ "--c": "var(--event)" }}><b>{fmt(dd.pct_lose_30, 1)}%</b><span>lose 30-minute access</span></div>
      </div>
      <Hist counts={dd.hist} edges={dd.edges} />
      </div><div>
      <table>
        <thead><tr><th>Group</th><th>median Δ</th><th>% lose 30-min</th></tr></thead>
        <tbody>
          {["has_car", "age_band", "lang"].flatMap((g) => Object.entries(dd.by[g] || {}).filter(([, v]) => v.n >= 100).map(([k, v]) => (
            <tr key={g + k}><td>{g === "has_car" ? (k === "True" ? "has a car" : "no car") : g === "lang" ? `speaks ${k}` : `age ${k}`}</td><td>{fmt(v.median_delta, 1)}</td><td>{fmt(v.pct_lose_30, 1)}%</td></tr>)))}
        </tbody>
      </table>
      {sim.personas?.length > 0 && <>
        <p style={{ color: "var(--ink)", marginBottom: 6 }}>Synthetic households (Census microdata, not real people)</p>
        <div className="grid2">{sim.personas.slice(0, 2).map((p, i) => (
          <div className="persona" key={i}><b>Age {p.age_band}, {p.has_car ? "has a car" : "no car"}</b><br />{p.lang} speaker · income {p.income_band}
            <br />clinic: {p.clinic_before} → <b>{p.clinic_after} min</b></div>))}</div>
      </>}
      </div></div>
      <p className="small" style={{ marginTop: 8 }}>Covers all 2019–24 displacement events citywide, not only buildings in the budget universe. Within-SF moves are excluded, so this covers households that left the city. Migration model held-out JSD {fmt(sim.summary?.migration_validation?.jsd_model, 2)} (population-proportional baseline {fmt(sim.summary?.migration_validation?.jsd_population_prop, 2)}).</p>
    </Step>
  );
}

function Honesty({ refEl, fair, h, sw }) {
  const at = (m) => sw?.rows.find((r) => r.mult === m);
  const sig = sw ? sw.rows.filter((r) => r.mult >= 1 && r.mult <= 5 && Number.isInteger(r.mult) && r.diff_ci[0] > 0).map((r) => `${r.mult}×`) : [];
  const [dim, setDim] = useState("income_tercile");
  const f1 = fair?.model_1x?.[dim], c1 = fair?.city_policy_1x?.[dim];
  const f5 = fair?.model_5x?.[dim];
  const keys = f1 ? Object.keys({ ...f1, ...(c1 || {}) }).filter((k) => k !== "NA" && k !== "nan").sort(sortKey) : [];
  return (
    <Step id="honesty" refEl={refEl}>
      <p className="kicker">6 · Honesty</p>
      <h2>Who it protects, and what it doesn't claim</h2>
      <div className="cols"><div>
      <div className="row" style={{ marginBottom: 8 }}>
        {[["income_tercile", "tract income"], ["nonwhite_tercile", "tract % nonwhite"], ["size_band", "building size"], ["supervisor_district", "district"]].map(([k, l]) =>
          <button key={k} className="pill" aria-pressed={dim === k} onClick={() => setDim(k)}>{l}</button>)}
      </div>
      {f1 && <table>
        <thead><tr><th>{dim.replace(/_/g, " ")}</th><th>share of displacement</th><th>model protected 1×</th><th>5×</th><th>city pattern 1×</th></tr></thead>
        <tbody>{keys.map((k) => (
          <tr key={k}><td>{k}</td><td>{pct(f1[k]?.share_displacement)}</td><td>{pct(f1[k]?.share_protected)}</td><td>{pct(f5?.[k]?.share_protected)}</td><td>{pct(c1?.[k]?.share_protected)}</td></tr>))}
          {sw && <tr><td><i>households protected</i></td><td></td><td>{fmt(at(1)?.model, 1)}</td><td>{fmt(at(5)?.model, 1)}</td><td>{fmt(at(1)?.city_policy, 1)}</td></tr>}</tbody>
      </table>}
      <p className="small">Share of all households protected that fall in each group, next to that group's share of actual displacement.</p>
      </div><div>
      <p><b style={{ color: "var(--ink)" }}>Ethics.</b> A building-level risk list is also a shopping list for speculators.
        Everything public here is aggregated to hexagons; building-level output is for program staff and land trusts.</p>
      <p><b style={{ color: "var(--ink)" }}>Leakage locks.</b> The world model behind every number here was trained only on data through 2018; tests enforce no year after the cutoff and no sponsor data in the backtest. The test split was scored by a frozen, committed model.</p>
      <p><b style={{ color: "var(--ink)" }}>Caveats.</b> A purchase is assumed to prevent all later Ellis, owner move-in and buyout displacement in the building (the same upper bound for every strategy).
        Notices are reported per block-face and shared across its buildings by unit count. Reported notices miss informal displacement.
        The city buys for more than prevention (often buildings already in trouble), so its pattern is not a prevention-only strategy.
        Over a single six-year window, the bootstrap interval on the model-minus-city difference excludes zero at {sig.length ? sig.join(", ") : "none of the budgets"} and includes it elsewhere in 1–5×.</p>
      </div></div>
      <p className="small">Model: two-head LightGBM on out-of-time world-model parcel features, blended with a sale-recency prior. Commit {h.commit}. Code: github.com/Christopher-Vu/DSSGoonathon2026</p>
    </Step>
  );
}
const pct = (x) => (x == null ? "–" : `${fmt(x * 100, 0)}%`);
const ORDER = { low: 0, mid: 1, high: 2 };
const sortKey = (a, b) => (a in ORDER && b in ORDER ? ORDER[a] - ORDER[b] : isNaN(a) || isNaN(b) ? String(a).localeCompare(String(b)) : a - b);
