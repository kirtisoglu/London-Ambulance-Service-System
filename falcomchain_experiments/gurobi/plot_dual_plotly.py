"""
Interactive plotly visualisation of a scalability grid's **dual graph**
(the unit adjacency graph $G^1$ from the paper).

Layers, all independently toggleable from the legend:

- Edges as light grey line segments (single Scattergl trace using
  None-separator trick).
- Base nodes as small grey dots with hover tooltip
  (id / position / demand / facility role).
- L1 facility candidates as blue circles, larger.
- L2 super-candidates as red diamonds, drawn on top.

Usage::

    python3 plot_dual_plotly.py            # default: grid_10000
    python3 plot_dual_plotly.py 20000

Output: ``figures/grid_{N}_dual.html``
"""

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_dual(path: Path):
    """Return rows, cols, nodes, edges from a generate_grids.py JSON."""
    with open(path) as f:
        d = json.load(f)
    edges = [(e["source"], e["target"]) for e in d["adjacency"]]
    return d["graph"]["rows"], d["graph"]["cols"], d["nodes"], edges


def render(n_nodes: int):
    path = DATA_DIR / f"grid_{n_nodes}.json"
    if not path.exists():
        raise SystemExit(f"Grid file not found: {path}")

    rows, cols, nodes, edges = load_dual(path)
    pos = {n["id"]: (int(n["C_X"]), int(n["C_Y"])) for n in nodes}

    # ---- edges trace (single Scattergl, None-separator) ----
    edge_x, edge_y = [], []
    for u, v in edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # ---- per-category node lists ----
    base_x, base_y, base_cd = [], [], []
    l1_x, l1_y, l1_cd = [], [], []
    l2_x, l2_y, l2_cd = [], [], []
    for n in nodes:
        nid = n["id"]
        x, y = pos[nid]
        is_l1 = bool(n.get("candidate", 0))
        is_l2 = bool(n.get("super_candidate", 0))
        role = "L2 super-candidate" if is_l2 else "L1 candidate" if is_l1 else "base"
        rec = (nid, n["demand"], role)
        if is_l2:
            l2_x.append(x); l2_y.append(y); l2_cd.append(rec)
        elif is_l1:
            l1_x.append(x); l1_y.append(y); l1_cd.append(rec)
        else:
            base_x.append(x); base_y.append(y); base_cd.append(rec)

    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="rgba(140,140,140,0.45)", width=0.6),
        hoverinfo="skip",
        name=f"edges ({len(edges):,})",
        showlegend=True,
    ))

    base_tpl = (
        "<b>%{customdata[2]}</b><br>"
        "node id = %{customdata[0]}<br>"
        "C_X = %{x} C_Y = %{y}<br>"
        "demand = %{customdata[1]}"
        "<extra></extra>"
    )

    node_size = 10  # match base-node size in plot_falcom_solution.py

    fig.add_trace(go.Scattergl(
        x=base_x, y=base_y,
        mode="markers",
        marker=dict(symbol="circle", size=node_size, color="#3f3f46",
                    line=dict(width=0)),
        name=f"base node ({len(base_x):,})",
        customdata=base_cd,
        hovertemplate=base_tpl,
    ))

    fig.add_trace(go.Scattergl(
        x=l1_x, y=l1_y,
        mode="markers",
        marker=dict(symbol="circle", size=node_size, color="#2563eb",
                    line=dict(width=0), opacity=0.95),
        name=f"L1 candidate ({len(l1_x):,})",
        customdata=l1_cd,
        hovertemplate=base_tpl,
    ))

    fig.add_trace(go.Scattergl(
        x=l2_x, y=l2_y,
        mode="markers",
        marker=dict(symbol="diamond", size=node_size, color="#dc2626",
                    line=dict(width=0), opacity=0.95),
        name=f"L2 super-candidate ({len(l2_x):,})",
        customdata=l2_cd,
        hovertemplate=base_tpl,
    ))

    title = (
        f"Dual graph G<sup>1</sup> of scalability instance — "
        f"|V| = {n_nodes:,} ({rows}×{cols}), |E| = {len(edges):,}; "
        f"L1: {len(l1_x):,}, L2: {len(l2_x):,}"
    )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        xaxis=dict(
            scaleanchor="y", scaleratio=1, constrain="domain",
            title=None, zeroline=False, showgrid=False,
            showticklabels=False, ticks="",
            range=[-1, cols],
        ),
        yaxis=dict(
            title=None, zeroline=False, showgrid=False,
            showticklabels=False, ticks="",
            range=[-1, rows],
        ),
        legend=dict(
            x=1.02, y=0.5, xanchor="left", yanchor="middle",
            bgcolor="rgba(255,255,255,0.92)", bordercolor="#bbb",
            borderwidth=1, itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(l=50, r=210, t=60, b=50),
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=12),
    )

    out = FIG_DIR / f"grid_{n_nodes}_dual.html"
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    size_mb = out.stat().st_size / 1e6
    print(
        f"Wrote {out}  ({size_mb:.2f} MB) — "
        f"{len(nodes):,} nodes, {len(edges):,} edges"
    )


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    render(n)
