"""Facility-region stability map for the LAS ensemble.

Groups the 1,831 CDBA candidates into travel-time regions (greedy ball
cover, seeds in descending per-candidate opening frequency, symmetrized
drive seconds, radius 240 s) and colors every candidate by the pooled
opening frequency of its region across the 360 post-burn-in snapshots.
Real stations are overlaid as black open circles.

Outputs fig_facility_regions.png next to the chain_v3 data and prints
the region statistics quoted in the paper.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DIR = REPO / "data/derived/chain_v3"
RADIUS_S = 240
BURN_IN_STEP = 1000

graph = json.load(open(REPO / "data/raw/london_graph.json"))
node_code = {n["id"]: n["LSOA21CD"] for n in graph["nodes"]}

cands = pd.read_csv(REPO / "data/derived/cdba_candidates_w10887_eps15_cmin1_dmax12.csv")
cand_codes = list(cands["LSOA21CD"])
is_real = dict(zip(cands["LSOA21CD"], cands["is_real"]))

trav = pd.read_parquet(REPO / "data/derived/cdba_travel_times_w10887_eps15_cmin1_dmax12.parquet")
o_idx = {c: i for i, c in enumerate(cand_codes)}
dests = sorted(trav["dest"].unique())
d_idx = {c: i for i, c in enumerate(dests)}
T = np.full((len(cand_codes), len(dests)), np.inf, dtype=np.float32)
T[trav["origin"].map(o_idx).to_numpy(), trav["dest"].map(d_idx).to_numpy()] = \
    trav["seconds"].to_numpy(dtype=np.float32)
cc = T[:, [d_idx[c] for c in cand_codes]]
cc_sym = np.minimum(cc, cc.T)

plans, site_freq = [], defaultdict(float)
for s in (1, 2, 3, 4):
    d = json.load(open(DIR / f"chain_v3_snapshots_sample_s{s}_T10000.json"))
    for sn in d["snapshots"]:
        if sn["step"] > BURN_IN_STEP:
            codes = [node_code[c] for c in sn["centers"].values() if c is not None]
            plans.append(codes)
for codes in plans:
    for c in set(codes):
        site_freq[c] += 1 / len(plans)

order = sorted(range(len(cand_codes)), key=lambda i: -site_freq.get(cand_codes[i], 0.0))
assigned = np.full(len(cand_codes), -1)
seeds = []
for i in order:
    if assigned[i] == -1:
        rid = len(seeds)
        seeds.append(i)
        assigned[(cc_sym[i] <= RADIUS_S) & (assigned == -1)] = rid
        assigned[i] = rid
region_of = {cand_codes[i]: int(assigned[i]) for i in range(len(cand_codes))}

pooled = defaultdict(float)
core_share_terms = []
for codes in plans:
    regs = {region_of[c] for c in codes}
    for rg in regs:
        pooled[rg] += 1 / len(plans)
core = {rg for rg, f in pooled.items() if f >= 0.9}
for codes in plans:
    core_share_terms.append(np.mean([region_of[c] in core for c in codes]))
region_has_real = defaultdict(int)
for c in cand_codes:
    if is_real[c]:
        region_has_real[region_of[c]] += 1

print(f"plans {len(plans)} | regions {len(seeds)} | ever-open {len(pooled)} | "
      f"core>=0.9 {len(core)} (with real station "
      f"{sum(1 for rg in core if region_has_real.get(rg, 0) > 0)}) | "
      f"share of facilities in core regions {np.mean(core_share_terms):.2f}")

lsoas = gpd.read_file(REPO / "data/raw/LSOA_2021_London.gpkg").to_crs(4326)
lsoas = lsoas[lsoas["LSOA21CD"].isin(set(dests))]

freq_of = np.array([pooled.get(region_of[c], 0.0) for c in cand_codes])
fig, ax = plt.subplots(figsize=(8.6, 6.4))
lsoas.plot(ax=ax, facecolor="#f2f2f2", edgecolor="white", linewidth=0.15)
sc = ax.scatter(cands["LONG"], cands["LAT"], c=freq_of, cmap="YlOrRd",
                vmin=0, vmax=1, s=8, linewidths=0, zorder=3)
real = cands[cands["is_real"] == 1]
ax.scatter(real["LONG"], real["LAT"], facecolors="none", edgecolors="black",
           s=34, linewidths=0.9, zorder=4, label="real station")
cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
cbar.set_label("opening frequency of the candidate's region", fontsize=10)
ax.legend(loc="lower right", frameon=False, fontsize=10)
ax.set_axis_off()
fig.savefig(DIR / "fig_facility_regions.png", dpi=200, bbox_inches="tight")
print(f"wrote {DIR / 'fig_facility_regions.png'}")
