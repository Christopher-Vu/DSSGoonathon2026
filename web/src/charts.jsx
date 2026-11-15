import { useEffect, useRef, useState } from "react";

const W = 416;
export const fmt = (x, d = 1) => (x == null || !isFinite(x) ? "–" : Number(x).toFixed(d));

function ticks(lo, hi, n = 4) {
  const span = hi - lo || 1;
  const step = Math.pow(10, Math.floor(Math.log10(span / n)));
  const s = [1, 2, 5, 10].map((k) => k * step).find((k) => span / k <= n) || step * 10;
  const out = [];
  for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-9; v += s) out.push(+v.toFixed(10));
  return out;
}

function Tip({ tip }) {
  if (!tip) return null;
  return (
    <div className="tip" style={{ left: tip.x + 14, top: tip.y + 14 }}>
      {tip.lines.map((l, i) => (
        <div key={i}>{l}</div>
      ))}
    </div>
  );
}

/** series: [{key, label, color, values:[y], band?:[[lo,hi]], dash?}] over shared xs. cursor: x value to mark. */
export function LineChart({ xs, series, h = 190, cursor, xfmt = (x) => x, yfmt = (y) => fmt(y, 0), xlabel, logx }) {
  const [hov, setHov] = useState(null);
  const m = { l: 34, r: 64, t: 10, b: 26 };
  const fx = logx ? Math.log : (x) => x;
  const x0 = fx(xs[0]), x1 = fx(xs[xs.length - 1]);
  const all = series.flatMap((s) => [...s.values, ...(s.band || []).flat()]).filter((v) => v != null);
  const ymax = Math.max(...all) * 1.08 || 1;
  const X = (x) => m.l + ((fx(x) - x0) / (x1 - x0)) * (W - m.l - m.r);
  const Y = (y) => h - m.b - (y / ymax) * (h - m.t - m.b);
  const path = (vals) => vals.map((v, i) => `${i ? "L" : "M"}${X(xs[i])},${Y(v)}`).join("");
  const area = (band) =>
    band.map((b, i) => `${i ? "L" : "M"}${X(xs[i])},${Y(b[1])}`).join("") +
    band.slice().reverse().map((b, j) => `L${X(xs[band.length - 1 - j])},${Y(b[0])}`).join("") + "Z";
  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    let best = 0;
    xs.forEach((x, i) => { if (Math.abs(X(x) - px) < Math.abs(X(xs[best]) - px)) best = i; });
    setHov({ i: best, x: e.clientX, y: e.clientY });
  };
  const ci = cursor != null ? xs.reduce((b, x, i) => (Math.abs(x - cursor) < Math.abs(xs[b] - cursor) ? i : b), 0) : null;
  const yt = ticks(0, ymax);
  return (
    <div style={{ position: "relative" }}>
      <svg className="chart" viewBox={`0 0 ${W} ${h}`} width="100%" onMouseMove={onMove} onMouseLeave={() => setHov(null)}>
        {yt.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={Y(t)} y2={Y(t)} stroke="var(--grid)" />
            <text x={m.l - 6} y={Y(t) + 4} textAnchor="end">{yfmt(t)}</text>
          </g>
        ))}
        <line x1={m.l} x2={W - m.r} y1={Y(0)} y2={Y(0)} stroke="var(--axis)" />
        {xs.filter((x, i) => xs.length < 8 || [1, 2, 5, 10, 20].includes(x) || i === 0).map((x) => (
          <text key={x} x={X(x)} y={h - 8} textAnchor="middle">{xfmt(x)}</text>
        ))}
        {xlabel && <text x={W - m.r} y={h - 8} textAnchor="start" dx={6}>{xlabel}</text>}
        {series.map((s) => s.band && <path key={s.key + "b"} d={area(s.band)} fill={s.color} opacity={0.16} />)}
        {series.map((s) => (
          <path key={s.key} d={path(s.values)} fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.dash} />
        ))}
        {series.map((s) => (
          <text key={s.key + "l"} className="lbl" x={X(xs[xs.length - 1]) + 6} y={Y(s.values[s.values.length - 1]) + 4}>{s.label}</text>
        ))}
        {ci != null && <line x1={X(xs[ci])} x2={X(xs[ci])} y1={m.t} y2={h - m.b} stroke="var(--ink-2)" strokeDasharray="2 3" />}
        {ci != null && series.map((s) => <circle key={s.key + "c"} cx={X(xs[ci])} cy={Y(s.values[ci])} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />)}
        {hov && <line x1={X(xs[hov.i])} x2={X(xs[hov.i])} y1={m.t} y2={h - m.b} stroke="var(--muted)" />}
      </svg>
      <Tip tip={hov && { x: hov.x, y: hov.y, lines: [<b key="x">{xfmt(xs[hov.i])}</b>, ...series.map((s) => (
        <span key={s.key}><i style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: s.color, marginRight: 6 }} />
          {s.label}: {yfmt(s.values[hov.i])}</span>))] }} />
    </div>
  );
}

/** rows: [{label, value, lo?, hi?, color, note?}] horizontal bars */
export function Bars({ rows, max, vfmt = (v) => fmt(v, 2), h = 26 }) {
  const [tip, setTip] = useState(null);
  const mx = max ?? Math.max(...rows.map((r) => r.hi ?? r.value)) * 1.05;
  const L = 132, R = 52;
  const X = (v) => L + (Math.max(0, v) / mx) * (W - L - R);
  return (
    <div>
      <svg className="chart" viewBox={`0 0 ${W} ${rows.length * h + 4}`} width="100%">
        {rows.map((r, i) => (
          <g key={r.label} transform={`translate(0,${i * h})`}
            onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, lines: [<b key="a">{r.label}</b>, `${vfmt(r.value)}${r.lo != null ? `  (95% CI ${vfmt(r.lo)}–${vfmt(r.hi)})` : ""}`, r.note].filter(Boolean) })}
            onMouseLeave={() => setTip(null)}>
            <rect x={0} y={0} width={W} height={h} fill="transparent" />
            <text className="lbl" x={0} y={h / 2 + 4}>{r.label}</text>
            <rect x={L} y={6} width={Math.max(2, X(r.value) - L)} height={h - 12} rx={4} fill={r.color} />
            {r.lo != null && <line x1={X(r.lo)} x2={X(r.hi)} y1={h / 2} y2={h / 2} stroke="var(--ink)" strokeWidth={1.5} />}
            <text className="lbl" x={X(r.hi ?? r.value) + 6} y={h / 2 + 4}>{vfmt(r.value)}</text>
          </g>
        ))}
      </svg>
      <Tip tip={tip} />
    </div>
  );
}

/** histogram of Δ minutes; edges length = counts + 1 */
export function Hist({ counts, edges, color = "var(--city)", h = 150, xlabel = "Δ minutes" }) {
  const [tip, setTip] = useState(null);
  const m = { l: 10, r: 10, t: 8, b: 26 };
  const n = counts.reduce((a, b) => a + b, 0) || 1;
  const mx = Math.max(...counts) / n;
  const bw = (W - m.l - m.r) / counts.length;
  const X = (v) => m.l + ((v - edges[0]) / (edges[edges.length - 1] - edges[0])) * (W - m.l - m.r);
  return (
    <div>
      <svg className="chart" viewBox={`0 0 ${W} ${h}`} width="100%">
        {counts.map((c, i) => {
          const hh = ((c / n) / mx) * (h - m.t - m.b);
          return (
            <rect key={i} x={m.l + i * bw + 1} y={h - m.b - hh} width={bw - 2} height={Math.max(0, hh)} rx={2}
              fill={edges[i] < 0 ? "var(--model)" : color}
              onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, lines: [`${edges[i]} to ${edges[i + 1]} min`, `${fmt((100 * c) / n, 1)}% of households`] })}
              onMouseLeave={() => setTip(null)} />
          );
        })}
        <line x1={X(0)} x2={X(0)} y1={m.t} y2={h - m.b} stroke="var(--ink-2)" strokeDasharray="2 3" />
        {[-30, 0, 30, 60, 90].map((t) => <text key={t} x={X(t)} y={h - 8} textAnchor="middle">{t}</text>)}
        <text x={W - m.r} y={h - 8} textAnchor="end" dy={0}>{""}</text>
      </svg>
      <div className="small">{xlabel} (≤ −30 and ≥ 90 pooled at the ends). Blue = shorter trip after the move.</div>
      <Tip tip={tip} />
    </div>
  );
}

/** fan: {q:{10:[..],25,50,75,90}, paths:[[..]], realized:[..]} over years */
export function Fan({ fan, years, h = 200, showRealized = true, progress = 1 }) {
  const m = { l: 34, r: 70, t: 10, b: 26 };
  const ymax = Math.max(...fan.q["90"], ...fan.realized, ...fan.paths.flat()) * 1.08 || 1;
  const X = (i) => m.l + (i / (years.length - 1)) * (W - m.l - m.r);
  const Y = (y) => h - m.b - (y / ymax) * (h - m.t - m.b);
  const line = (v) => v.map((y, i) => `${i ? "L" : "M"}${X(i)},${Y(y)}`).join("");
  const band = (lo, hi) => hi.map((y, i) => `${i ? "L" : "M"}${X(i)},${Y(y)}`).join("") + lo.slice().reverse().map((y, j) => `L${X(lo.length - 1 - j)},${Y(y)}`).join("") + "Z";
  const np = Math.round(fan.paths.length * progress);
  const yt = ticks(0, ymax);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${h}`} width="100%">
      {yt.map((t) => (
        <g key={t}><line x1={m.l} x2={W - m.r} y1={Y(t)} y2={Y(t)} stroke="var(--grid)" />
          <text x={m.l - 6} y={Y(t) + 4} textAnchor="end">{fmt(t, 0)}</text></g>
      ))}
      {years.map((y, i) => <text key={y} x={X(i)} y={h - 8} textAnchor="middle">{y}</text>)}
      <path d={band(fan.q["10"], fan.q["90"])} fill="var(--model)" opacity={0.14} />
      <path d={band(fan.q["25"], fan.q["75"])} fill="var(--model)" opacity={0.2} />
      {fan.paths.slice(0, np).map((p, i) => <path key={i} d={line(p)} fill="none" stroke="var(--model)" strokeWidth={1} opacity={0.28} />)}
      <path d={line(fan.q["50"])} fill="none" stroke="var(--model)" strokeWidth={2} />
      <text className="lbl" x={X(years.length - 1) + 6} y={Y(fan.q["50"][years.length - 1]) + 4}>model median</text>
      {showRealized && <>
        <path d={line(fan.realized)} fill="none" stroke="var(--event)" strokeWidth={2.5} />
        <text className="lbl" x={X(years.length - 1) + 6} y={Y(fan.realized[years.length - 1]) + 4}>actual</text>
      </>}
    </svg>
  );
}

export function useCountUp(target, ms = 900) {
  const [v, setV] = useState(target ?? 0);
  const ref = useRef(v);
  ref.current = v;
  useEffect(() => {
    if (target == null) return;
    const from = ref.current, t0 = performance.now();
    let raf;
    const step = (t) => {
      const k = Math.min(1, (t - t0) / ms);
      setV(from + (target - from) * (1 - Math.pow(1 - k, 3)));
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}
