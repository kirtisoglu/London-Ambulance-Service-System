"""
Build the LAS ensemble diagnostics dashboard (self-contained HTML).

Reads data/derived/chain_v3/chain_v3_sample_s{seed}_T{T}.json produced by
run_chain_v3.py and renders:
  - KPI row (acceptance, throughput, Rhat on E, |P1| / |P2| at T)
  - energy traces for the four chains (burn-in shaded)
  - |P1| and |P2| traces (small multiples)
  - post-burn-in energy distributions (per-chain frequency polygons)

Chart styling follows the dataviz reference palette (categorical slots 1-4,
light surface #fcfcfb) with direct end-labels, a legend, hairline grid, and a
nearest-point hover tooltip per chart.

Run:
    python -m falcomchain_experiments.las.plot_ensemble_diagnostics \
        --seeds 1 2 3 4 --steps 10000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IN_DIR = REPO / "data/derived/chain_v3"
OUT_HTML = IN_DIR / "las_ensemble_diagnostics.html"

# dataviz reference palette, light mode, categorical slots 1-4
SERIES = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"


def split_rhat(chains: list[list[float]]) -> float:
    """Split-Rhat (Gelman et al.) on equal-length chains."""
    half = min(len(c) for c in chains) // 2
    seqs = []
    for c in chains:
        seqs.append(c[:half])
        seqs.append(c[half:2 * half])
    m, n = len(seqs), half
    means = [sum(s) / n for s in seqs]
    grand = sum(means) / m
    B = n / (m - 1) * sum((mu - grand) ** 2 for mu in means)
    W = sum(sum((x - mu) ** 2 for x in s) / (n - 1)
            for s, mu in zip(seqs, means)) / m
    var_plus = (n - 1) / n * W + B / n
    return math.sqrt(var_plus / W) if W > 0 else float("nan")


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        if raw <= mult * mag:
            step = mult * mag
            break
    start = math.ceil(lo / step) * step
    ticks = []
    t = start
    while t <= hi + 1e-9:
        ticks.append(t)
        t += step
    return ticks


def fmt(v: float) -> str:
    if abs(v) >= 1e6:
        return f"{v/1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v/1e3:.0f}k"
    return f"{v:.0f}"


def line_chart(cid, title, subtitle, series, x_key, y_key, y_fmt=fmt,
               burnin=None, w=1160, h=300):
    """Multi-series SVG line chart with hover tooltip. series =
    [{name, color, rows:[{x, y}]}]."""
    ml, mr, mt, mb = 64, 118, 30, 34
    pw, ph = w - ml - mr, h - mt - mb
    xs = [r[x_key] for s in series for r in s["rows"]]
    ys = [r[y_key] for s in series for r in s["rows"]]
    x0, x1 = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    pad = (yhi - ylo) * 0.08 or 1
    ylo, yhi = ylo - pad, yhi + pad

    def X(v): return ml + (v - x0) / (x1 - x0 or 1) * pw
    def Y(v): return mt + ph - (v - ylo) / (yhi - ylo) * ph

    parts = []
    if burnin:
        parts.append(
            f'<rect x="{ml}" y="{mt}" width="{X(burnin)-ml:.1f}" height="{ph}" '
            f'fill="{GRID}" opacity="0.35"/>'
            f'<text x="{X(burnin)-6:.1f}" y="{mt+12}" text-anchor="end" '
            f'font-size="10" fill="{MUTED}">burn-in</text>')
    for t in nice_ticks(ylo, yhi):
        parts.append(
            f'<line x1="{ml}" x2="{ml+pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
            f'<text x="{ml-8}" y="{Y(t)+3:.1f}" text-anchor="end" '
            f'font-size="10" fill="{MUTED}">{y_fmt(t)}</text>')
    for t in nice_ticks(x0, x1, 6):
        parts.append(
            f'<text x="{X(t):.1f}" y="{mt+ph+16}" text-anchor="middle" '
            f'font-size="10" fill="{MUTED}">{fmt(t)}</text>')
    parts.append(
        f'<line x1="{ml}" x2="{ml+pw}" y1="{mt+ph}" y2="{mt+ph}" '
        f'stroke="{BASE}" stroke-width="1"/>')

    for s in series:
        pts = " ".join(f"{X(r[x_key]):.1f},{Y(r[y_key]):.1f}" for r in s["rows"])
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" '
            f'stroke-width="2" stroke-linejoin="round"/>')
        last = s["rows"][-1]
        parts.append(
            f'<circle cx="{X(last[x_key]):.1f}" cy="{Y(last[y_key]):.1f}" r="3" '
            f'fill="{s["color"]}" stroke="{SURFACE}" stroke-width="2"/>'
            f'<text x="{ml+pw+8}" y="{Y(last[y_key])+3:.1f}" font-size="11" '
            f'fill="{INK2}"><tspan fill="{s["color"]}">&#9632;</tspan> {s["name"]}</text>')

    data_json = json.dumps([
        {"name": s["name"], "color": s["color"],
         "pts": [[round(X(r[x_key]), 1), round(Y(r[y_key]), 1),
                  r[x_key], r[y_key]] for r in s["rows"]]}
        for s in series])
    return f"""
<figure class="card">
  <figcaption><strong>{title}</strong><span class="sub">{subtitle}</span></figcaption>
  <div class="chartwrap">
  <svg id="{cid}" viewBox="0 0 {w} {h}" data-series='{data_json}'
       data-ml="{ml}" data-pw="{pw}">
    {''.join(parts)}
    <line class="xhair" y1="{mt}" y2="{mt+ph}" stroke="{BASE}" stroke-width="1"
          stroke-dasharray="3 3" visibility="hidden"/>
  </svg>
  <div class="tip" hidden></div>
  </div>
</figure>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=10_000)
    args = ap.parse_args()

    runs = []
    for i, seed in enumerate(args.seeds):
        p = IN_DIR / f"chain_v3_sample_s{seed}_T{args.steps}.json"
        d = json.loads(p.read_text())
        d["_color"] = SERIES[i % len(SERIES)]
        d["_name"] = f"chain {seed}"
        runs.append(d)

    burnin = args.steps // 10
    post = [[r["E"] for r in d["diagnostics"] if r["step"] > burnin]
            for d in runs]
    rhat = split_rhat(post)
    all_post = [e for c in post for e in c]
    e_mean = sum(all_post) / len(all_post)

    acc = [d["acceptance_rate"] for d in runs]
    sps = [d["steps_per_second"] for d in runs]
    p1_final = [d["final_summary_global"].get("n_districts") for d in runs]
    p2_final = [d["final_summary_global"].get("n_super") for d in runs]

    kpis = [
        ("Acceptance", f"{min(acc):.1%}&ndash;{max(acc):.1%}", "4 chains, always-accept sampling"),
        ("Throughput", f"{sum(sps)/len(sps):.0f} steps/s", "single workstation"),
        ("R&#770; on E", f"{rhat:.3f}", f"post burn-in ({burnin:,} steps)"),
        ("E&#772; post burn-in", fmt(e_mean), "demand-weighted minutes"),
        ("|P&sup1;| at T", f"{min(p1_final)}&ndash;{max(p1_final)}", "districts (66 real stations)"),
        ("|P&sup2;| at T", f"{min(p2_final)}&ndash;{max(p2_final)}", "super-districts (21 real Groups)"),
    ]
    kpi_html = "".join(
        f'<div class="tile"><div class="k">{k}</div>'
        f'<div class="v">{v}</div><div class="s">{s}</div></div>'
        for k, v, s in kpis)

    e_chart = line_chart(
        "cE", "Energy trace", "hierarchical p-median E(s), all four chains",
        [{"name": d["_name"], "color": d["_color"],
          "rows": [{"x": r["step"], "y": r["E"]} for r in d["diagnostics"]]}
         for d in runs], "x", "y", burnin=burnin)

    p1_chart = line_chart(
        "cP1", "|P&sup1;| trace", "level-1 district count",
        [{"name": d["_name"], "color": d["_color"],
          "rows": [{"x": r["step"], "y": r["P1"]} for r in d["diagnostics"]]}
         for d in runs], "x", "y", y_fmt=lambda v: f"{v:.0f}",
        burnin=burnin, h=240)

    p2_chart = line_chart(
        "cP2", "|P&sup2;| trace", "level-2 super-district count (real LAS: 21 Groups)",
        [{"name": d["_name"], "color": d["_color"],
          "rows": [{"x": r["step"], "y": r["P2"]} for r in d["diagnostics"]]}
         for d in runs], "x", "y", y_fmt=lambda v: f"{v:.0f}",
        burnin=burnin, h=240)

    # post-burn-in E frequency polygons on a shared grid
    lo, hi = min(all_post), max(all_post)
    nb = 18
    def poly(vals):
        counts = [0] * nb
        for v in vals:
            j = min(nb - 1, int((v - lo) / (hi - lo or 1) * nb))
            counts[j] += 1
        return [{"x": lo + (j + 0.5) * (hi - lo) / nb, "y": c}
                for j, c in enumerate(counts)]
    hist_chart = line_chart(
        "cH", "Post burn-in energy distribution",
        "frequency polygons over the final 9,000 steps (thinned)",
        [{"name": d["_name"], "color": d["_color"], "rows": poly(c)}
         for d, c in zip(runs, post)], "x", "y",
        y_fmt=lambda v: f"{v:.0f}", h=260)

    cfg = runs[0]["params"]
    html = f"""<title>LAS ensemble diagnostics</title>
<style>
  .viz-root {{ background:{PAGE}; color:{INK}; font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
    padding:28px; max-width:1280px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .cfg {{ color:{INK2}; font-size:13px; margin-bottom:20px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:20px; }}
  .tile {{ background:{SURFACE}; border:1px solid rgba(11,11,11,0.10); border-radius:10px; padding:14px 16px; }}
  .tile .k {{ font-size:12px; color:{INK2}; }}
  .tile .v {{ font-size:26px; font-weight:650; margin:2px 0; }}
  .tile .s {{ font-size:11px; color:{MUTED}; }}
  .card {{ background:{SURFACE}; border:1px solid rgba(11,11,11,0.10); border-radius:10px;
    padding:16px; margin:0 0 16px; }}
  figcaption {{ font-size:14px; margin-bottom:8px; }}
  figcaption .sub {{ color:{MUTED}; font-size:12px; margin-left:10px; }}
  .chartwrap {{ position:relative; }}
  svg {{ width:100%; height:auto; display:block; }}
  .tip {{ position:absolute; pointer-events:none; background:{INK}; color:#fff;
    font-size:11px; padding:6px 8px; border-radius:6px; white-space:nowrap; z-index:2; }}
  .note {{ color:{INK2}; font-size:12.5px; margin-top:8px; }}
</style>
<div class="viz-root">
  <h1>LAS ensemble diagnostics: 4 chains &times; T = {args.steps:,}</h1>
  <div class="cfg">capacity-block calibration k=3 &middot; w_unit = {cfg['w']:,} &middot;
    c&sup1; &isin; [{cfg['c_min_base']}, {cfg['c_max_base']}] units &middot;
    c&sup2; &isin; [{cfg['c_min_super']}, {cfg['c_max_super']}] &middot;
    &kappa;&sup2;<sub>min</sub> = {cfg['min_districts_super']} &middot;
    &epsilon; = {cfg['eps_base']} &middot; &gamma; = {cfg['gamma_base']:.0f} &middot;
    unmodified main-branch FalcomChain</div>
  <div class="kpis">{kpi_html}</div>
  {e_chart}
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">{p1_chart}{p2_chart}</div>
  {hist_chart}
  <p class="note">Rejections are proposal-level: the recursion cannot close (base level, rare)
  or the supergraph strands a final supernode (dominant, &asymp;51% of steps). The chain holds
  its state on rejection; all accepted states are feasible by construction.</p>
</div>
<script>
for (const svg of document.querySelectorAll('svg[data-series]')) {{
  const wrap = svg.parentElement, tip = wrap.querySelector('.tip');
  const xh = svg.querySelector('.xhair');
  const series = JSON.parse(svg.dataset.series);
  svg.addEventListener('mousemove', ev => {{
    const r = svg.getBoundingClientRect();
    const sx = (ev.clientX - r.left) * svg.viewBox.baseVal.width / r.width;
    let best = null;
    for (const s of series) for (const p of s.pts) {{
      const d = Math.abs(p[0] - sx);
      if (!best || d < best.d) best = {{d, p, s}};
    }}
    if (!best) return;
    xh.setAttribute('x1', best.p[0]); xh.setAttribute('x2', best.p[0]);
    xh.setAttribute('visibility', 'visible');
    const rows = series.map(s => {{
      let q = s.pts[0];
      for (const p of s.pts) if (Math.abs(p[2] - best.p[2]) < Math.abs(q[2] - best.p[2])) q = p;
      return `<span style="color:${{s.color}}">&#9632;</span> ${{s.name}}: ${{Number(q[3]).toLocaleString()}}`;
    }});
    tip.innerHTML = `<div>x = ${{Number(best.p[2]).toLocaleString()}}</div>` + rows.join('<br>');
    tip.hidden = false;
    const px = best.p[0] * r.width / svg.viewBox.baseVal.width;
    tip.style.left = Math.min(px + 12, r.width - tip.offsetWidth - 8) + 'px';
    tip.style.top = '18px';
  }});
  svg.addEventListener('mouseleave', () => {{
    tip.hidden = true; xh.setAttribute('visibility', 'hidden');
  }});
}}
</script>"""
    OUT_HTML.write_text(html)
    print(f"wrote {OUT_HTML.relative_to(REPO)} "
          f"(Rhat={rhat:.4f}, acceptance {min(acc):.1%}-{max(acc):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
