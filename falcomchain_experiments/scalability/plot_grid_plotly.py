"""
Interactive plotly map of a scalability grid.

Renders a synthetic FalCom grid as three layered traces:

1. A demand heatmap (`Heatmap`) over the full grid, hover shows
   demand at the cell.
2. L1 facility candidates as small blue dots
   (`Scattergl`), hover tooltip = facility role + node ID + demand.
3. L2 super-candidates as larger red diamonds (`Scattergl`),
   hover tooltip = facility role + node ID + demand.

Both facility layers are independently toggleable from the legend.
Output is a self-contained HTML file with embedded plotly JS.

Usage::

    python3 plot_grid_plotly.py            # default: grid_10000
    python3 plot_grid_plotly.py 20000

Output: ``figures/grid_{N}_plotly.html``
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


def load_grid(path: Path):
    with open(path) as f:
        d = json.load(f)
    return d["graph"]["rows"], d["graph"]["cols"], d["nodes"]


def render(n_nodes: int):
    path = DATA_DIR / f"grid_{n_nodes}.json"
    if not path.exists():
        raise SystemExit(f"Grid file not found: {path}")
    rows, cols, nodes = load_grid(path)

    # Build a (rows, cols) demand matrix + parallel arrays for candidates.
    demand = np.zeros((rows, cols), dtype=np.int32)
    l1_x, l1_y, l1_node, l1_demand = [], [], [], []
    l2_x, l2_y, l2_node, l2_demand = [], [], [], []
    for n in nodes:
        x, y = int(n["C_X"]), int(n["C_Y"])
        demand[y, x] = n["demand"]
        if n.get("super_candidate"):
            l2_x.append(x); l2_y.append(y)
            l2_node.append(n["id"]); l2_demand.append(n["demand"])
        elif n.get("candidate"):
            l1_x.append(x); l1_y.append(y)
            l1_node.append(n["id"]); l1_demand.append(n["demand"])

    fig = go.Figure()

    # Demand heatmap. We use a Heatmap (not contour) so each cell is
    # visible individually; tooltip shows the per-cell demand.
    fig.add_trace(go.Heatmap(
        z=demand,
        x=np.arange(cols),
        y=np.arange(rows),
        colorscale="YlGnBu",
        colorbar=dict(title="demand w<sub>v</sub>", thickness=14, len=0.6),
        hovertemplate=(
            "<b>Base node</b><br>"
            "C_X = %{x}<br>"
            "C_Y = %{y}<br>"
            "demand = %{z}"
            "<extra></extra>"
        ),
        name="demand",
        zmin=80, zmax=120,
    ))

    # L1 candidates — blue circles, click-to-hide via legend.
    fig.add_trace(go.Scattergl(
        x=l1_x, y=l1_y,
        mode="markers",
        marker=dict(
            symbol="circle",
            size=6, color="#2563eb",
            line=dict(color="white", width=0.6),
            opacity=0.85,
        ),
        name=f"L1 candidate ({len(l1_x):,})",
        customdata=np.column_stack([l1_node, l1_demand]),
        hovertemplate=(
            "<b>L1 facility candidate</b><br>"
            "node id = %{customdata[0]}<br>"
            "C_X = %{x} C_Y = %{y}<br>"
            "demand at node = %{customdata[1]}"
            "<extra></extra>"
        ),
    ))

    # L2 super-candidates — red diamonds, drawn on top.
    fig.add_trace(go.Scattergl(
        x=l2_x, y=l2_y,
        mode="markers",
        marker=dict(
            symbol="diamond",
            size=11, color="#dc2626",
            line=dict(color="black", width=0.8),
            opacity=0.95,
        ),
        name=f"L2 super-candidate ({len(l2_x):,})",
        customdata=np.column_stack([l2_node, l2_demand]),
        hovertemplate=(
            "<b>L2 super-candidate (HQ/EOC equivalent)</b><br>"
            "node id = %{customdata[0]}<br>"
            "C_X = %{x} C_Y = %{y}<br>"
            "demand at node = %{customdata[1]}"
            "<extra></extra>"
        ),
    ))

    title = (
        f"FalCom scalability grid — |V| = {n_nodes:,} "
        f"({rows}×{cols}); L1: {len(l1_x):,}, L2: {len(l2_x):,}"
    )
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        xaxis=dict(
            scaleanchor="y", scaleratio=1,
            constrain="domain", title="C_X",
            zeroline=False, showgrid=False,
        ),
        yaxis=dict(title="C_Y", zeroline=False, showgrid=False),
        legend=dict(
            x=1.02, y=0.5, xanchor="left", yanchor="middle",
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#bbb",
            borderwidth=1, itemclick="toggle", itemdoubleclick="toggleothers",
        ),
        margin=dict(l=50, r=200, t=60, b=50),
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=12),
    )

    out = FIG_DIR / f"grid_{n_nodes}_plotly.html"
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    size_mb = out.stat().st_size / 1e6
    print(f"Wrote {out}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    render(n)
