"""Fixes for the blocking items of the fourth referee round (A, B, D, E, F, G)."""
import json, pathlib, re
P = pathlib.Path("src/main_arxiv.tex"); s = P.read_text()
LAS = pathlib.Path("/home/user/London-Ambulance-Service-System")
def rep(old, new, cnt=1):
    global s
    pat = re.escape(old).replace('\\ ', r'\s+'); n = len(re.findall(pat, s)); assert n == cnt, (old[:70], n)
    s = re.sub(pat, lambda m: new, s)

# A1: novelty sentence of the summary
rep("FalCom is, to our knowledge, the first sampler that enforces contiguity, hierarchy and capacity constraints at once and supports ensemble-based distributional analyses.",
    "FalCom is, to our knowledge, the first sampler whose state is a hierarchy of contiguous districts with a facility and an integer capacity in every district and a number of districts that is not fixed in advance, the multiscale samplers of redistricting being its closest predecessors, and it supports ensemble-based distributional analyses.")
# A2 / E1 / G1: summary of the enumeration
rep("On instances small enough to enumerate, the chain visits every feasible state and only feasible states, its empirical law is independent of the start to within a total-variation distance of $0.01$ to $0.02$, and that law spreads mass across capacity mixes roughly uniformly while following the spanning-tree law within a mix only loosely, at a total-variation distance of $0.11$ to $0.30$.",
    "On instances small enough to enumerate, the chain visits only feasible states, every one of them on the two smallest instances and all but $0.1\\%$ on the third, its empirical law is independent of the start to within sampling noise (total-variation distances of $0.01$ to $0.05$ against iid floors of $0.01$ to $0.035$), and that law spreads mass across capacity mixes nearly uniformly on $12$ and $16$ units but between the uniform and spanning-tree laws on $24$, while following the spanning-tree law within a mix only loosely, at a total-variation distance of $0.11$ to $0.28$.")
rep("weights feasible plans by factors of up to several thousand, through spanning-tree counts",
    "weights feasible plans by factors of up to $64{,}000$ on the largest enumerated instance, through spanning-tree counts")
rep("On the London Ambulance Service, run on the $66$ real stations, four chains started from different seeds agree on every observable to within the noise of a single chain and forget their starts within a few hundred steps.",
    "On the London Ambulance Service, run on the $66$ real stations, five chains, four from a sector-wise start and one from a Group-based start, agree on every observable to within the noise of a single chain and forget their starts within a few hundred steps.")
rep("and an upper bound, within a static assignment model, on the access cost that workload balance would exact from the operational layout.",
    "and an upper bound, within a static assignment model, on the station-leg access cost that workload balance would exact from the operational layout.")
# B1: level-2 centre on the grids
rep("Capacities are $c^1 \\in [1,2]$ and $c^2 \\in [2,5]$ with $\\kappa^2_{\\min} = 2$, so that superdistricts are meaningful.",
    "Capacities are $c^1 \\in [1,2]$ and $c^2 \\in [2,5]$ with $\\kappa^2_{\\min} = 2$, so that superdistricts are meaningful. On the grids the level-2 windows of the chain are centred at $w$, the model's centre, which differs from $\\bar d = d/\\lceil d/w \\rceil$ by $0.9\\%$ to $2.3\\%$ across the five grids; the London chains use $\\bar d$ (Section~\\ref{sec:exp-emergency}).")
# D1: safe range and the supergraph
rep("This is allowed at the supergraph, where Lemma~\\ref{lem:super-convex} makes per-team balance automatic, and the debt mechanism is needed only at level 1, where $c^1_{\\max} = 2$.",
    "The safe range concerns the debt-corrected window of level $1$, where $c^1_{\\max} = 2$; the supergraph carries no debt term, so the range does not constrain $c^2_{\\max}$, and the supergraph recursion has no closing guarantee (Appendix~\\ref{app:debt}), whose demand failures are reported in Section~\\ref{sec:exp-multistart}.")
# D2: granularity floor and the enumeration instances
rep("and all our calibrations satisfy it.", "the grid and London calibrations satisfy it, and the enumeration instances of Section~\\ref{sec:exp-enumeration} violate it and are feasible nonetheless.")
# D3: London zone-wise start
rep("each sector cut against the global per-unit target with its share of the $97$ units fixed by largest remainder (Lemma~\\ref{lem:app-telescoping}), so that every initial state carries exactly $\\lceil d/w \\rceil$ units;",
    "each sector cut against its own per-unit target $d_Z / c_Z$ (Lemma~\\ref{lem:app-telescoping}) with its share $c_Z$ of the $97$ units fixed by largest remainder, so that every initial state carries exactly $\\lceil d/w \\rceil$ units and the sector targets lie between $0.98$ and $1.03$ times $\\bar d$, inside the level-2 window;")
# E2: density table rows and sentences from the stored sweep
def sweep(n):
    d = json.load(open(LAS / f"falcomchain_experiments/scalability/results/grid_{n}_seed42.json"))
    rr = d["rejection_report"]; acc = 1 - d["n_chain_steps_rejected"] / max(1, d["n_chain_steps_recorded"])
    det = rr.get("causes_detail") or {}; base = sum(v for k, v in det.items() if k.startswith("cut_search_exhausted:base")) if det else rr["causes"].get("cut_search_exhausted:base", 0)
    return acc, base, d["n_init_partition_retries"], d["init_partition_time_s"]
a1, b1, r1, t1 = sweep(1000); a10, b10, r10, t10 = sweep(10000)
m = re.search(r"grid\\_1000\s+& \$1\.5\$ & \$\d+\$ & \$(\d+)\\%\$ & \$(\d+)\$ & \$(\d+)\$ & ([^\n]*?) \\\\", s); assert m, "grid_1000 row"
row = m.group(0); new_row = row.replace(f"${m.group(1)}\\%$", f"${100 * a1:.0f}\\%$", 1).replace(f"& ${m.group(2)}$ & ${m.group(3)}$", f"& ${b1}$ & ${r1 + 1}$", 1)
s = s.replace(row, new_row); print("grid_1000 row:", new_row)
m = re.search(r"grid\\_10000\s+& \$1\.5\$ & \$\d+\$ & \$(\d+)\\%\$ & \$(\d+)\$ & \$(\d+)\$ & ([^\n]*?) \\\\", s); assert m, "grid_10000 row"
row = m.group(0); new_row = row.replace(f"${m.group(1)}\\%$", f"${100 * a10:.0f}\\%$", 1).replace(f"& ${m.group(2)}$ & ${m.group(3)}$", f"& ${b10}$ & ${r10 + 1}$", 1).replace(m.group(4), f"${t10:.0f}$\\,s")
s = s.replace(row, new_row); print("grid_10000 row:", new_row)
rep("Acceptance falls from $92\\%$ to $91\\%$ on grid\\_1000 and from $87\\%$ to $79\\%$ on grid\\_10000",
    f"Acceptance falls from ${100 * a1:.0f}\\%$ to $91\\%$ on grid\\_1000 and from ${100 * a10:.0f}\\%$ to $79\\%$ on grid\\_10000")
pts = max(abs(100 * a1 - 91), abs(100 * a10 - 79))
words = {8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen"}
rep("Acceptance changes by at most eight points,", f"Acceptance changes by at most {words.get(int(round(pts)), str(int(round(pts))))} points,")
# E3: abstract
rep("with the same capacity mix and open facilities, under a slightly wider balance tolerance than the model's.",
    "with the same capacity mix and level-2 facilities and, up to one site, the same open level-1 facilities, as the best of eight annealing runs under a slightly wider balance tolerance than the model's.")
# F1: incident total
rep("about $1.065$ million incidents a year, apportioned to the LSOAs of each borough in proportion to the ONS mid-2024 population estimates~\\citep{ons_sape2024}.",
    "about $1.056$ million incidents a year across the $33$ boroughs, all of them in the catchment, apportioned to the LSOAs of each borough in proportion to the ONS mid-2024 population estimates~\\citep{ons_sape2024}.")
rep("about $1.065$ million incidents a year, of which $1{,}056{,}000$ fall in the catchment, apportioned",
    "about $1.056$ million incidents a year across the $33$ boroughs, all of them in the catchment, apportioned")
# F2: speed factor
rep("A uniform speed factor rescales every road drive time and leaves the energies, the $1$-medians, the open frequencies and the ratio of the benchmark's energy to the ensemble's unchanged, although not the eight-minute share;",
    "A uniform speed factor rescales every road drive time, and hence every energy, by the same amount and leaves the ordering of plans by energy, the $1$-medians, the open frequencies and the ratio of the benchmark's energy to the ensemble's unchanged, up to the four fallback LSOAs, although not the eight-minute share;")
# F3: the bound on the station leg
bp = json.load(open(LAS / "data/derived/real_stations_u97/best_plans.json")); opt = json.load(open(LAS / "data/derived/real_stations_u97_opt/optimizer_summary_T50000.json"))
E1_slas = 3083626.102; E1_min = bp["sample_extremes"]["E1_min"]; E1_opt = min(r["best_E1"] for r in opt["runs"])
g1 = (E1_min - E1_slas) / 1e6; p1 = 100 * (E1_min - E1_slas) / E1_slas; g2 = (E1_opt - E1_slas) / 1e6; p2 = 100 * (E1_opt - E1_slas) / E1_slas
rep("Against the best stored plan the bound is $1.18$ million demand-weighted minutes, $16\\%$ of the layout's energy, of which $0.86$ million is on the station leg.",
    f"The layout's Group leg is evaluated at the documented headquarters and the sampled plans' at their $1$-medians, so the two sides of the total energy are not computed under one rule and the valid bound is the station-leg one, where both sides are: against the lowest station leg in the sample, ${E1_min/1e6:.2f}$ million, it is ${g1:.2f}$ million demand-weighted minutes, ${p1:.0f}\\%$ of the layout's station leg. On the total energy the gap to the best stored plan is $1.18$ million, $16\\%$, which is at most the bound under a single rule.")
rep("Against it the bound tightens to $0.70$ million demand-weighted minutes, $9\\%$ of the layout's energy, of which $0.55$ million is on the station leg, and in the station-leg metrics to",
    f"Against it the station-leg bound tightens to ${g2:.2f}$ million, ${p2:.0f}\\%$ of the layout's station leg (its gap on the total energy is $0.70$ million, $9\\%$), and in the station-leg metrics to")
# G3: the per-team safety check is rare, not unreachable
rep("makes this check unreachable under the debt-corrected window. Should a numerical edge case violate it, the proposal is rejected and the chain remains at $s^{(i)}$.",
    "makes this check fail only through numerical edge cases: it rejected $14$ of $120{,}000$ proposals on grid\\_50000 and none elsewhere (Section~\\ref{sec:exp-multistart}). When it fails, the proposal is rejected and the chain remains at $s^{(i)}$.")
rep("and at most $19$ base-level failures per $30{,}000$ proposals, the price of a candidate set with $1.5$ sites per team;",
    "and at most $19$ base-level failures per $30{,}000$ proposals, the price of a candidate set with $1.5$ sites per team, plus $14$ proposals in $120{,}000$ on grid\\_50000 that failed the per-team safety check of Section~\\ref{sec:recursive} after a cut;")
P.write_text(s); print("v4 fixes applied; station-leg bounds", round(g1, 3), round(p1, 1), round(g2, 3), round(p2, 1))
# highlights (built by make_ejor.py)
q = pathlib.Path("make_ejor.py"); t = q.read_text()
old = '"A London ensemble bounds the access cost that workload balance would exact",'
new = '"A London ensemble bounds the station-leg access cost of workload balance",'
assert t.count(old) == 1; q.write_text(t.replace(old, new)); print("highlight updated")
