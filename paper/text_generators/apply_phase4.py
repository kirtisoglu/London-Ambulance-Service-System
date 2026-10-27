"""Regenerate Sections 7.2.2 and 7.2.3, contribution (iii) and the Discussion sentence on fixed-size
instances from the phase-4 and phase-5 grid results (idempotent)."""
import csv, json, pathlib, re, statistics, sys
import numpy as np
SC = pathlib.Path("/home/user/London-Ambulance-Service-System/falcomchain_experiments/scalability")
P = pathlib.Path("src/main_arxiv.tex"); s = P.read_text()

def rep(old, new, cnt=1):
    global s
    pat = re.escape(old).replace('\\ ', r'\s+'); n = len(re.findall(pat, s)); assert n == cnt, (old[:70], n)
    s = re.sub(pat, lambda m: new, s)

def causes(rep_):
    c = dict(rep_.get("causes", {}) or {})
    det = rep_.get("causes_detail") or {}
    if det:   # the detailed split replaces the coarse cut-search entries; the other causes stay
        c = {k: v for k, v in c.items() if not k.startswith("cut_search_exhausted")}
        c.update(det)
    out = {"kappa": 0, "super_tree": 0, "super_demand": 0, "super": 0, "base": 0}
    for k, v in c.items():
        if k.startswith("super_district_below_kappa"): out["kappa"] += v
        elif k.startswith("cut_search_exhausted:super"):
            out["super"] += v
            if ":demand" in k: out["super_demand"] += v
            else: out["super_tree"] += v
        elif k.startswith("cut_search_exhausted:base"): out["base"] += v
    return out

# ---------------- sweep ----------------
rows = list(csv.DictReader(open(SC / "results/summary.csv")))
rows = sorted(rows, key=lambda r: int(r["n_nodes"]))
runs = {int(r["n_nodes"]): json.load(open(SC / f"results/grid_{r['n_nodes']}_seed42.json")) for r in rows}
def pct(x): return f"{100 * x:.0f}\\%"
tab5 = []
for r in rows:
    n = int(r["n_nodes"]); run = runs[n]
    rej = run["n_chain_steps_rejected"] / max(1, run["n_chain_steps_recorded"])
    mean_all_ms = 1000 * run["chain_total_time_s"] / max(1, run["n_chain_steps_succeeded"])
    minutes = run["chain_total_time_s"] / 60
    tab5.append((n, float(r["k_districts_median"]), float(r["H_mean"]), float(r["H_over_V_mean"]),
                 float(r["t_step_median_ms"]), float(r["t_step_p95_ms"]), float(r["super_share_of_step_time"]), rej, minutes, mean_all_ms,
                 run["init_partition_time_s"], run["n_init_partition_retries"], causes(run.get("rejection_report", {})), run["k_teams_coverage"]))

def rg(lo, hi, fmt="{:.0f}"):
    a, b = fmt.format(lo), fmt.format(hi)
    return f"${a}$" if a == b else f"${a}$ to ${b}$"

def fmt_int(x): return f"{int(round(x)):,}".replace(",", "{,}")
lines = []
for (n, k, H, HV, med, p95, ss, rej, minutes, mean_all, *_r) in tab5:
    lines.append(f"${fmt_int(n)}$ & ${k:.0f}$ & ${fmt_int(H)}$ & ${HV:.3f}$ & ${med:.1f}$\\,ms & ${mean_all:.0f}$\\,ms & ${p95:.0f}$\\,ms & ${100 * ss:.0f}\\%$ & ${100 * rej:.0f}\\%$ & ${minutes:.1f}$\\,min \\\\")
table5 = r"""\begin{table}[!htbp]
\centering
\small\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{\begin{tabular}{rrrrrrrrrr}
\toprule
$|V|$ & $|\mathcal{P}^1|$ & mean $|H|$ & $|H|/|V|$ & median $t_{\mathrm{step}}$ & mean, all proposals & $95$th pct & supergraph share & rejected & $10{,}000$ steps \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}}
\caption{Scalability sweep across the five synthetic grids, one chain of $10{,}000$ steps each. $|\mathcal{P}^1|$ is the median number of districts. The median and the $95$th percentile are over accepted steps; ``mean, all proposals'' is the wall-clock time per proposal including rejected ones. ``Supergraph share'' is the share of the step time spent re-partitioning the supergraph. Rejected steps are proposals whose recursion could not close within its retry budget or that would have violated $\kappa^2_{\min}$; they leave the state unchanged.}
\label{tab:scalability}
\end{table}"""
first, last = tab5[0], tab5[-1]
f_med = last[4] / first[4]; f_mean = last[9] / first[9]; f_H = last[2] / first[2]; f_V = last[0] / first[0]
slope = np.log(last[4] / first[4]) / np.log(last[0] / first[0])
k_first, k_last = first[13], last[13]
ss_first, ss_last = first[6], last[6]
base_max = max(t[12]["base"] for t in tab5)
rej_first, rej_last = first[7], last[7]
init_last = last[10]
super_tree_last = last[12]["super_tree"] / max(1, runs[last[0]]["n_chain_steps_recorded"]); super_dem_last = last[12]["super_demand"] / max(1, runs[last[0]]["n_chain_steps_recorded"])
kappa_last = last[12]["kappa"] / max(1, runs[last[0]]["n_chain_steps_recorded"])

# ---------------- fixed-size runs ----------------
fixed = []
for w in (10000, 5000, 2500):
    rr = []
    for sd in (42, 43):
        f = SC / f"results/fixed_size/grid_10000_w{w}_seed{sd}.json"
        if f.exists():
            d = json.load(open(f)); acc = [x for x in d["steps"] if x["accepted"]]
            rr.append(dict(k=d["k_teams_coverage"], P1=statistics.median(x["n_districts"] for x in d["steps"]),
                           H=statistics.mean(x["H_size"] for x in acc), med=1000 * statistics.median(x["t_step"] for x in acc),
                           ss=sum(x.get("t_super", 0) for x in d["steps"]) / max(1e-9, sum(x["t_step"] for x in d["steps"])),
                           rej=d["n_chain_steps_rejected"] / max(1, d["n_chain_steps_recorded"]),
                           init=d["init_partition_time_s"], retries=d["n_init_partition_retries"], cand=d["n_l1_candidates"]))
    fixed.append((w, rr))
fixed_txt = []
for w, rr in fixed:
    if not rr:
        kk = {10000: 101, 5000: 201, 2500: 401}[w]
        fixed_txt.append(f"with $k = {kk}$ teams the zone-wise start, two zones of about $200$ teams each, did not close in twelve attempts (nineteen minutes), so the chain could not start")
        continue
    k = rr[0]["k"]; med = [r["med"] for r in rr]; ss = [r["ss"] for r in rr]; init = [r["init"] for r in rr]; P1 = [r["P1"] for r in rr]; H = [r["H"] for r in rr]; rej = [r["rej"] for r in rr]
    fixed_txt.append(f"with $k = {k}$ teams ({rr[0]['cand']} candidates, {rg(min(P1), max(P1))} districts) the zone-wise start closes in {rg(min(init), max(init))} seconds, the median accepted step takes {rg(min(med), max(med))}\\,ms with $|H| \\approx {statistics.mean(H):.0f}$, the supergraph recursion takes {rg(100 * min(ss), 100 * max(ss))}\\% of a step and {rg(100 * min(rej), 100 * max(rej))}\\% of proposals are rejected")
fixed_para = ("With $w = 200\\sqrt{|V|}$ the team count and the district size grow together, so the sweep cannot separate the two parts; holding the graph fixed and raising the team count does. On grid\\_10000 rebuilt with $w = 10{,}000$, $5{,}000$ and $2{,}500$ at $1.5$ candidate sites per team, " +
              "; ".join(fixed_txt) + ". The local part is set by the merged region and the global part by $k$, and with many hundreds of teams the level-2 re-partition is the part that scales.")

scal_text = (r"""\subsubsection{Scalability}\label{sec:exp-scalability}

Contribution~(iii) in Section~\ref{sec:intro} claims that FalCom handles graphs with up to $50{,}000$ basic units. A step re-cuts one superdistrict at level $1$ and re-partitions the supergraph of $k$ nodes at level $2$, so its cost has a local part, set by the merged subgraph $H$, and a global part, set by $k$. We run the sampler on all five grids for $T = 10{,}000$ steps and record, for every step, the wall-clock time, its split between the two parts, and the merged-subgraph size $|H| = \sum_{D \in D^2_j} |D|$ of the superdistrict $D^2_j$ that is re-partitioned. Travel times are Manhattan distances on grid coordinates evaluated lazily. Table~\ref{tab:scalability} reports the per-step cost.

""" + table5 + "\n\n" +
    f"Across $|V| \\in [100, 50{{,}}000]$, a ${f_V:.0f}$-fold range in which $k$ grows from ${k_first}$ to ${k_last}$, the median time of an accepted step rises from ${first[4]:.1f}$ to ${last[4]:.0f}$\\,ms, a factor of ${f_med:.0f}$ or $|V|^{{{slope:.2f}}}$, and the mean time per proposal including rejected ones by a factor of ${f_mean:.0f}$, while $|H|/|V|$ falls from ${first[3]:.2f}$ to ${last[3]:.3f}$. The step time grows faster than the mean $|H|$, which rises by a factor of ${f_H:.0f}$; the difference is the supergraph re-partition, ${100 * ss_first:.0f}\\%$ of a step on grid\\_{first[0]} and ${100 * ss_last:.0f}\\%$ on grid\\_50000. A proposal that cannot close spends its whole retry budget of $M = 1{{,}}000$ trees, so on grid\\_50000 the $10{{,}}000$ steps take ${last[8]:.0f}$ minutes after ${init_last:.0f}$ seconds for the zone-wise initial partition; rejections rise from ${pct(rej_first)}$ on grid\\_100 to ${pct(rej_last)}$ on grid\\_50000, almost all of them supergraph tree failures (${pct(super_tree_last)}$) and re-cuts that would leave a superdistrict with one district (${pct(kappa_last)}$), with at most ${base_max}$ base-level failures in $10{{,}}000$ proposals. Times are for the cloud container used for all runs in this section, roughly half the speed of the laptop of Section~\\ref{{sec:exp-milp}}; the ratios, not the absolute times, are the claim.\n\n"
    + fixed_para + "\n\n% =========================================================================\n\\FloatBarrier\n")
start = s.index("\\subsubsection{Scalability}\\label{sec:exp-scalability}")
end = s.index("\\subsection{Comparison with Exact Optima}")
s = s[:start] + scal_text + s[end:]

# ---------------- multi-start ----------------
ms = {n: json.load(open(SC / f"results/multistart/multistart_{n}.json")) for n in (10000, 50000)}
def rng_(d): return min(d.values()), max(d.values())
tab4 = []
for n in (10000, 50000):
    m = ms[n]; acc = rng_(m["acceptance"]); tf = rng_({k: v for k, v in m["forgetting"]["steps_to_baseline"].items() if v is not None})
    obs = ("R1", "dspread", "n_districts", "n_superdistricts", "cut_edges", "share_unit_capacity_districts")
    between = " & ".join(f"${m[o]['max_pairwise_ks']:.3f}$" for o in obs)
    within = " & ".join(f"${m[o]['within_chain_halves_ks_range'][0]:.3f}$--${m[o]['within_chain_halves_ks_range'][1]:.3f}$" for o in obs)
    tab4.append(f"grid\\_{n}, between chains & ${100 * acc[0]:.1f}$--${100 * acc[1]:.1f}\\%$ & {between} & ${tf[0]}$--${tf[1]}$ & ${m['R1']['split_rhat']:.3f}$ & ${fmt_int(m['R1']['ess'])}$ \\\\")
    tab4.append(f"\\quad within one chain, halves & & {within} & & & \\\\")
table4 = r"""\begin{table}[!htbp]
\centering
\footnotesize\setlength{\tabcolsep}{3pt}
\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrrrrrrr}
\toprule
 & acceptance & $R^1$ & spread & $|\Pc^1|$ & $|\Pc^2|$ & cut edges & unit share & steps to forget & $\hat R(R^1)$ & ESS$(R^1)$ \\
\midrule
""" + "\n".join(tab4) + r"""
\bottomrule
\end{tabular}}
\caption{Multi-start diagnostics on the two large grids, four chains of $30{,}000$ steps each after a $20\%$ burn-in. The six middle columns are Kolmogorov--Smirnov distances: on the first row of each instance the largest pairwise distance among the four chains, on the second the range over chains of the distance between the two halves of one chain, for the level-1 access cost $R^1$, the per-team demand spread and the number of districts (every twentieth post-burn-in step), and for the number of superdistricts, the cut-edge count and the share of unit-capacity districts (stored plans, every $60$th step). ``Steps to forget'' is the range over chains of the number of steps after which the share of the initial boundary still in place is within $10\%$ of the share two independent plans have in common. $\hat R$ and ESS refer to $R^1$, the ESS pooled over the four chains.}
\label{tab:multistart}
\end{table}"""
m1, m5 = ms[10000], ms[50000]
def cause_share(m):
    tot = {}
    for sd, c in m["rejection_causes"].items():
        det = (m.get("rejection_causes_detail") or {}).get(sd)
        cc = causes({"causes": c, "causes_detail": det}); n = m["steps"][sd]
        for k, v in cc.items(): tot.setdefault(k, []).append(v / n)
    return {k: (min(v), max(v)) for k, v in tot.items()}
c1, c5 = cause_share(m1), cause_share(m5)
base_max_ms = max(max(causes({"causes": c, "causes_detail": (m.get("rejection_causes_detail") or {}).get(sd)})["base"] for sd, c in m["rejection_causes"].items()) for m in ms.values())
crit1 = 1.36 * np.sqrt(2 / (m1["R1"]["ess"] / 4)); crit5 = 1.36 * np.sqrt(2 / (m5["R1"]["ess"] / 4))
P1_1 = rng_(m1["n_districts"]["means"]); P1_5 = rng_(m5["n_districts"]["means"])
tf1 = rng_({k: v for k, v in m1["forgetting"]["steps_to_baseline"].items() if v is not None}); tf5 = rng_({k: v for k, v in m5["forgetting"]["steps_to_baseline"].items() if v is not None})
obs_names = {"R1": "$R^1$", "dspread": "the spread", "n_districts": "the number of districts", "n_superdistricts": "the number of superdistricts", "cut_edges": "the cut-edge count", "share_unit_capacity_districts": "the unit-capacity share"}
exceed = []
for n_, m_ in ((10000, m1), (50000, m5)):
    for o in obs_names:
        if m_[o]["max_pairwise_ks"] > m_[o]["within_chain_halves_ks_range"][1]:
            exceed.append((n_, o, m_[o]["max_pairwise_ks"], m_[o]["within_chain_halves_ks_range"][1]))
if not exceed:
    agree = "On every observable the chains differ from each other no more than one chain differs from itself."
else:
    items = [f"{obs_names[o]} on grid\\_{n_} (${a:.3f}$ against a within-chain maximum of ${b:.3f}$)" for n_, o, a, b in exceed]
    parts = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    agree = f"On {12 - len(exceed)} of the twelve instance and observable pairs the chains differ from each other no more than one chain differs from itself; the exceptions are {parts}."
ms_text = (r"""\subsubsection{Multi-start agreement on large grids}
\label{sec:exp-multistart}

On grid\_10000 and grid\_50000, where enumeration is impossible, we run four chains from independently drawn zone-wise initial partitions, each carrying exactly $\lceil d/w \rceil$ units, and ask whether they forget their starts and agree with each other. The observables are the level-1 access cost $R^1 = \sum_{D \in \Pc^1} \sum_{v \in D} d_v\, \mathrm{d}(f^1(D), v)$, the base term of the median objective~\eqref{eq:milp-median} and the slow mode of the chain since one superdistrict is re-cut per step, the per-team demand spread $\max_D d(D)/c^1(D) - \min_D d(D)/c^1(D)$, the number of districts and, on the stored plans, the number of superdistricts, the cut-edge count and the share of unit-capacity districts. Agreement is the largest pairwise Kolmogorov--Smirnov distance between the post-burn-in distributions of an observable, compared with the same distance between the two halves of one chain; forgetting is the number of steps after which the share of the initial plan's boundary edges still in place reaches the share two independent plans have in common. Table~\ref{tab:multistart} reports the diagnostics.

""" + table4 + "\n\n" +
    agree + f" The series are autocorrelated, so the critical values of the two-sample test do not apply; at the effective sample size of $R^1$ per chain, ${m1['R1']['ess'] / 4:.0f}$ and ${m5['R1']['ess'] / 4:.0f}$, a difference in distribution below about ${crit1:.2f}$ and ${crit5:.2f}$ is not detectable, and the within-chain reference calibrates the noise of the statistic rather than proving convergence, since two halves of a chain confined to one region would agree as well; what the agreement of independently started chains adds is that the law reached does not depend on the start. Split-$\\hat R$ is at most ${max(m1['R1']['split_rhat'], m5['R1']['split_rhat'], m1['dspread']['split_rhat'], m5['dspread']['split_rhat']):.3f}$. The chains forget their starts within ${tf1[0]}$ to ${tf1[1]}$ steps on grid\\_10000 and ${tf5[0]}$ to ${tf5[1]}$ on grid\\_50000, for plans of {rg(*P1_1)} and {rg(*P1_5)} districts, after which the boundary share settles at the independent-plan value, ${m1['forgetting']['independent_baseline']:.2f}$ and ${m5['forgetting']['independent_baseline']:.2f}$. The rejections are supergraph trees without an admissible pendant subtree (${100 * c1['super_tree'][0]:.0f}$--${100 * c1['super_tree'][1]:.0f}\\%$ and ${100 * c5['super_tree'][0]:.0f}$--${100 * c5['super_tree'][1]:.0f}\\%$ of proposals), demand failures of the supergraph recursion (at most ${100 * c1['super_demand'][1]:.1f}\\%$ and ${100 * c5['super_demand'][1]:.1f}\\%$), re-cuts that would leave a superdistrict with a single district (at most ${100 * c1['kappa'][1]:.0f}\\%$ and ${100 * c5['kappa'][1]:.0f}\\%$) and at most ${base_max_ms}$ base-level failures per $30{{,}}000$ proposals, the price of a candidate set with $1.5$ sites per team; the larger grid has superdistricts of two to five units, so a re-cut of merged superdistricts more often draws a tree with no pendant subtree in the level-2 window. Every rejection is a repeated proposal, not an error; the four chains sample the same law, and the holding times of rejected proposals enter that law as quantified on the London instance (Section~\\ref{{sec:las-protocol}}).\n\n\\FloatBarrier\n")
start = s.index("\\subsubsection{Multi-start agreement on large grids}")
end = s.index("\\subsubsection{Scalability}\\label{sec:exp-scalability}")
s = s[:start] + ms_text + s[end:]

# ---------------- contribution (iii) ----------------
s2, n = re.subn(r"On the tested grids, where \$k\$ grows from \$\d+\$ to \$\d+\$, the median time of an accepted step grows by a factor of \$\d+\$ and the mean step including rejected proposals by a factor of about \$\d+\$, while \$\|V\|\$ grows \$500\$-fold(?:; at a fixed graph size the cost grows with the team count through the supergraph re-partition)? \(Section~\\ref\{sec:exp-scalability\}\)\.",
    lambda m: f"On the tested grids, where $k$ grows from ${k_first}$ to ${k_last}$, the median time of an accepted step grows by a factor of ${f_med:.0f}$ and the mean step including rejected proposals by a factor of about ${f_mean:.0f}$, while $|V|$ grows $500$-fold; at a fixed graph size the cost grows with the team count through the supergraph re-partition (Section~\\ref{{sec:exp-scalability}}).", s)
assert n == 1, n; s = s2
# ---------------- Discussion sentence on fixed-workload variants ----------------
ok_ws = [w for w, rr in fixed if rr]
k_ok = max([r["k"] for w, rr in fixed for r in rr], default=None)
if k_ok is None:
    disc = "fixed-size variants with several hundred teams are reported in Section~\\ref{sec:exp-scalability}"
elif len(ok_ws) == 3:
    disc = f"at a fixed graph size the zone-wise start still closes with ${k_ok}$ teams (Section~\\ref{{sec:exp-scalability}})"
else:
    disc = f"at a fixed graph size the zone-wise start closes with ${k_ok}$ teams but not, in twelve attempts, with twice as many (Section~\\ref{{sec:exp-scalability}})"
s2, n = re.subn(r"(?:fixed-workload variants with several hundred teams did not initialize at all|(?:fixed-size variants with|at a fixed graph size the zone-wise start)[^.]*?\(Section~\\ref\{sec:exp-scalability\}\))", lambda m: disc, s)
assert n == 1, n; s = s2
P.write_text(s); print("phase 4 text applied; fixed-size runs found:", [(w, len(rr)) for w, rr in fixed])
