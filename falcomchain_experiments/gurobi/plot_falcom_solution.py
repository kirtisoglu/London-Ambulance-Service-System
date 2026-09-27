"""
Reusable interactive plotter for a FalCom MILP / chain solution on a
synthetic grid instance.

Designed so it can later be dropped into ``falcomplot.grid`` as

    from falcomplot.grid import plot_falcom_solution
    fig = plot_falcom_solution(graph, solution, params=...)
    fig.write_html("out.html")

Inputs are intentionally framework-agnostic:
- ``graph`` is a NetworkX graph with node attributes
    ``C_X``, ``C_Y``, ``demand``, ``candidate``, ``super_candidate``.
- ``solution`` is a mapping with at least the keys
    ``x1``, ``y1``, ``c1``, ``x2``, ``y2``, ``c2``
  (the convention produced by ``solve_milp.py``). Optional keys
  ``R1``, ``R2``, ``obj_value``, ``objective``, ``params``,
  ``wall_time_s``, ``mip_gap`` enrich the title.

Layers (all individually toggleable from the legend):
1. Edges of the dual graph G^1.
2. One trace per L1 district, base nodes colour-coded.
3. L2 super-district outlines (one black curve per super-district,
   tracing the boundary of the union of member unit-cells).
4. L1 candidate markers: hollow blue rings for *closed* candidates,
   solid blue rings with black border for *open* L1 facilities.
5. L2 super-candidate markers: hollow red stars for *closed*
   super-candidates, solid red stars with black border for *open*
   super-facilities.

Tooltips are populated from the underlying data; hovering reveals
demand, district / super-district membership, and the capacity of
open facilities.
"""

from __future__ import annotations

import colorsys
from collections import Counter, defaultdict
from typing import Mapping

import networkx as nx
import plotly.graph_objects as go


# Vivid, perceptually-distinct categorical palette — every L1 district
# gets a unique colour from this set, so neighbours never look alike.
# (The hierarchy / super-grouping is still visible via the thick L2
# super-district contours drawn separately in the plot.)
DISTRICT_PALETTE = [
    "#e6194B", "#4363d8", "#3cb44b", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990", "#9A6324", "#800000",
    "#808000", "#000075", "#e6beff", "#aaffc3",
]


def _hierarchical_colors(open_facs, assign2):
    """Assign each L1 district a distinct colour from a vivid categorical
    palette. Districts are sorted by (super id, district id) so that
    members of the same super-district occupy a contiguous palette slice
    — handy when there are many districts — but every district keeps its
    own clearly-separated colour."""
    by_super = defaultdict(list)
    for j1 in open_facs:
        by_super[assign2.get(j1, j1)].append(j1)
    color_of = {}
    palette_idx = 0
    for sid in sorted(by_super):
        for j1 in sorted(by_super[sid]):
            color_of[j1] = DISTRICT_PALETTE[palette_idx % len(DISTRICT_PALETTE)]
            palette_idx += 1
    return color_of

# Marker style constants.
NODE_SIZE = 10
CANDIDATE_SIZE = 16
SUPER_SIZE = 20


def _build_title(graph, solution) -> str:
    """One-line title + a smaller subtitle, both derived from solution data."""
    n = graph.number_of_nodes()
    e = graph.number_of_edges()
    rows = graph.graph.get("rows")
    cols = graph.graph.get("cols")
    shape = f"{rows}x{cols}" if rows and cols else f"|V|={n}"

    y1 = solution.get("y1", {})
    y2 = solution.get("y2", {})
    open_l1 = sum(1 for v in y1.values() if int(v) == 1)
    open_l2 = sum(1 for v in y2.values() if int(v) == 1)
    total_l1 = len(y1) if y1 else sum(
        1 for _, d in graph.nodes(data=True) if d.get("candidate")
    )
    total_l2 = len(y2) if y2 else sum(
        1 for _, d in graph.nodes(data=True) if d.get("super_candidate")
    )

    obj_name = solution.get("objective", "?")
    obj_val = solution.get("obj_value")
    R1 = solution.get("R1")
    R2 = solution.get("R2")
    obj_part = f"obj({obj_name}) = {obj_val:g}" if obj_val is not None else ""
    if R1 is not None and R2 is not None:
        obj_part += f"  =  R¹ ({R1:g}) + R² ({R2:g})"

    head = (
        f"FalCom solution on grid {shape}, |V|={n:,}, |E|={e:,}"
    )
    counts = (
        f"open L1: {open_l1} / {total_l1}, "
        f"open L2: {open_l2} / {total_l2}"
    )
    sub_pieces = [counts, obj_part]

    params = solution.get("params")
    if params:
        sub_pieces.append(
            f"w={params.get('w')}, "
            f"ε¹={params.get('eps_l1')}, "
            f"ε²={params.get('eps_l2')}, "
            f"c¹ in [{params.get('c_min_l1')}, "
            f"{params.get('c_max_l1')}], "
            f"c² in [{params.get('c_min_l2')}, "
            f"{params.get('c_max_l2')}], "
            f"min L1 per L2: {params.get('min_l1_per_l2')}, "
            f"contiguity: {params.get('contiguity')}"
        )
    if solution.get("wall_time_s") is not None:
        sub_pieces.append(f"solve: {solution['wall_time_s']:.2f}s")

    sub = "<br>".join(p for p in sub_pieces if p)
    return f"{head}<br><span style='font-size:11px;color:#555'>{sub}</span>"


def _add_edges(fig, graph, pos):
    edge_x, edge_y = [], []
    for u, v in graph.edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    fig.add_trace(go.Scattergl(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(140,140,140,0.45)", width=0.6),
        hoverinfo="skip",
        name=f"edges ({graph.number_of_edges():,})",
    ))


def _add_super_outlines(fig, graph, pos, assign2, open_super, assign1, demand):
    """One black curve per L2 super-district, tracing the union of unit-cells."""
    if not open_super:
        return
    from shapely.geometry import box, Polygon, MultiPolygon
    from shapely.ops import unary_union

    for j2 in open_super:
        members = [n for n in graph.nodes if assign2.get(n) == j2]
        if not members:
            continue
        sd_demand = sum(demand[i] for i in members)
        l1_in_super = sorted({assign1[i] for i in members})
        squares = [box(pos[i][0] - 0.5, pos[i][1] - 0.5,
                       pos[i][0] + 0.5, pos[i][1] + 0.5)
                   for i in members]
        footprint = unary_union(squares)
        polys = (list(footprint.geoms)
                 if isinstance(footprint, MultiPolygon)
                 else [footprint])
        xs, ys = [], []
        for poly in polys:
            for ring in [poly.exterior, *poly.interiors]:
                rx, ry = ring.xy
                xs.extend(list(rx) + [None])
                ys.extend(list(ry) + [None])
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="black", width=2.0),
            hoverinfo="skip",
            name=(f"L2 super-district {j2}  "
                  f"({len(members)} nodes, "
                  f"{len(l1_in_super)} L1, "
                  f"demand {sd_demand:,})"),
            legendgroup="L2",
        ))


def _classify(node, y1_map, y2_map, attrs):
    """Disjoint category assignment with priority L2-open > L1-open
    > base. Only *open* facilities get a distinctive marker; closed
    candidates (and closed super-candidates) are rendered as normal
    base nodes (same size, no star). Each node is in exactly one
    trace so toggling a legend entry removes it from the map."""
    is_l1 = bool(attrs.get("candidate"))
    is_l2 = bool(attrs.get("super_candidate"))
    if is_l2 and int(y2_map.get(node, 0)) == 1:
        return "l2_open"
    if is_l1 and int(y1_map.get(node, 0)) == 1:
        return "l1_open"
    return "base"


def _add_base_nodes_by_district(fig, graph, pos, demand, assign1, c1_map,
                                color_of, category_of):
    """One trace per L1 district holding only nodes whose category is
    'base' (non-candidate). Districts whose only nodes are candidates
    are still listed as a zero-marker entry so the legend remains
    complete."""
    for j in sorted(color_of):
        member_ids = [i for i in graph.nodes
                      if assign1[i] == j and category_of[i] == "base"]
        all_in_district = [i for i in graph.nodes if assign1[i] == j]
        d_total = sum(demand[i] for i in all_in_district)
        cap = c1_map.get(j)
        xs = [pos[i][0] for i in member_ids]
        ys = [pos[i][1] for i in member_ids]
        cd = [(i, demand[i], j, d_total, len(all_in_district), cap)
              for i in member_ids]
        tpl = (
            "<b>L1 district %{customdata[2]} — base node</b><br>"
            "node id = %{customdata[0]}<br>"
            "C_X = %{x} C_Y = %{y}<br>"
            "demand = %{customdata[1]}<br>"
            "district size = %{customdata[4]} nodes<br>"
            "district demand = %{customdata[3]}<br>"
            "c¹ = %{customdata[5]}"
            "<extra></extra>"
        )
        fig.add_trace(go.Scattergl(
            x=xs, y=ys, mode="markers",
            marker=dict(symbol="circle", size=NODE_SIZE, color=color_of[j],
                        line=dict(width=0)),
            name=(f"L1 district {j}  "
                  f"({len(all_in_district)} nodes, demand {d_total:,}, "
                  f"c¹={cap})"),
            customdata=cd,
            hovertemplate=tpl,
        ))


def _add_candidate_category(fig, graph, pos, demand, assign1, color_of,
                            category_of, *, key, label, symbol,
                            border_color, border_width,
                            c_map_for_tooltip=None,
                            level_name="L1"):
    """One trace for a single candidate category, with per-node fill
    inherited from the L1 district colour (so the district structure
    is still visible inside the marker)."""
    ids = [n for n in graph.nodes if category_of[n] == key]
    if not ids:
        return
    xs = [pos[i][0] for i in ids]
    ys = [pos[i][1] for i in ids]
    fills = [color_of[assign1[i]] for i in ids]
    cd = [(i, demand[i], assign1[i],
           (c_map_for_tooltip or {}).get(i, "-"))
          for i in ids]
    tpl = (
        f"<b>{level_name} {label}</b><br>"
        "node id = %{customdata[0]}<br>"
        "C_X = %{x} C_Y = %{y}<br>"
        "demand at node = %{customdata[1]}<br>"
        "L1 district = %{customdata[2]}<br>"
        f"c = %{{customdata[3]}}"
        "<extra></extra>"
    )
    fig.add_trace(go.Scattergl(
        x=xs, y=ys, mode="markers",
        marker=dict(
            symbol=symbol, size=CANDIDATE_SIZE,
            color=fills,
            line=dict(color=border_color, width=border_width),
        ),
        name=f"{level_name} {label}  ({len(ids)})",
        customdata=cd,
        hovertemplate=tpl,
    ))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def plot_falcom_solution(graph: nx.Graph, solution: Mapping) -> go.Figure:
    """Render a FalCom MILP / chain solution on a grid as an interactive
    plotly figure. See module docstring for input shape and layer list."""
    pos = {n: (int(graph.nodes[n]["C_X"]), int(graph.nodes[n]["C_Y"]))
           for n in graph.nodes}
    demand = {n: int(graph.nodes[n]["demand"]) for n in graph.nodes}

    # Normalise solution dict's keys to integer ids where useful.
    assign1 = {int(k): int(v) for k, v in solution["x1"].items()}
    assign2 = {int(k): int(v) for k, v in solution.get("x2", {}).items()}
    y1_map = {int(k): int(v) for k, v in solution["y1"].items()}
    y2_map = {int(k): int(v) for k, v in solution.get("y2", {}).items()}
    c1_map = {int(k): int(v) for k, v in solution.get("c1", {}).items()}
    c2_map = {int(k): int(v) for k, v in solution.get("c2", {}).items()}

    open_facs = sorted(j for j, v in y1_map.items() if v == 1)
    open_super = sorted(j for j, v in y2_map.items() if v == 1)
    # Hierarchical colouring: one base hue per L2 super-district, and a
    # lightness-varied tone per L1 district inside it — so districts in
    # the same super-district read as a colour family, distinct
    # super-districts as different hues.
    color_of = _hierarchical_colors(open_facs, assign2)

    rows = graph.graph.get("rows")
    cols = graph.graph.get("cols")

    # Disjoint category per node (priority L2-open > L1-open > base).
    # Closed candidates are folded into 'base' so they render as normal
    # nodes (no bigger marker, no star); only open facilities stand out.
    category_of = {
        n: _classify(n, y1_map, y2_map, graph.nodes[n]) for n in graph.nodes
    }

    fig = go.Figure()

    # ---- z-order: edges -> outlines -> nodes -> open facilities --------
    _add_edges(fig, graph, pos)
    _add_super_outlines(fig, graph, pos, assign2, open_super, assign1, demand)
    _add_base_nodes_by_district(
        fig, graph, pos, demand, assign1, c1_map, color_of, category_of,
    )
    _add_candidate_category(
        fig, graph, pos, demand, assign1, color_of, category_of,
        key="l1_open", label="facility (open)",
        symbol="circle", border_color="black", border_width=2.5,
        c_map_for_tooltip=c1_map, level_name="L1",
    )
    _add_candidate_category(
        fig, graph, pos, demand, assign1, color_of, category_of,
        key="l2_open", label="super-facility (open)",
        symbol="star", border_color="black", border_width=2.5,
        c_map_for_tooltip=c2_map, level_name="L2",
    )

    fig.update_layout(
        title=dict(text=_build_title(graph, solution), x=0.02, xanchor="left"),
        xaxis=dict(
            scaleanchor="y", scaleratio=1, constrain="domain",
            title=None, zeroline=False, showgrid=False,
            showticklabels=False, ticks="",
            range=[-1, cols] if cols else None,
        ),
        yaxis=dict(
            title=None, zeroline=False, showgrid=False,
            showticklabels=False, ticks="",
            range=[-1, rows] if rows else None,
        ),
        legend=dict(
            x=1.02, y=0.5, xanchor="left", yanchor="middle",
            bgcolor="rgba(255,255,255,0.92)", bordercolor="#bbb",
            borderwidth=1, itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(l=50, r=280, t=110, b=50),
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    return fig


# --------------------------------------------------------------------------
# CLI wrapper (kept thin so the falcomplot-bound function stays clean)
# --------------------------------------------------------------------------

def _cli():
    import argparse, json
    from pathlib import Path
    here = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument("instance", type=str, default="100", nargs="?")
    ap.add_argument("obj", default="radius_minmax", nargs="?")
    args = ap.parse_args()

    grid_path = here / "data" / f"grid_{args.instance}.json"
    sol_path = here / f"solution_{args.instance}_{args.obj}.json"
    with open(grid_path) as f:
        gd = json.load(f)
    graph = nx.node_link_graph(gd, edges="adjacency")
    with open(sol_path) as f:
        solution = json.load(f)

    fig = plot_falcom_solution(graph, solution)
    out = here / "figures" / f"solution_{args.instance}_{args.obj}.html"
    out.parent.mkdir(exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    print(f"Wrote {out}  ({out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    _cli()
