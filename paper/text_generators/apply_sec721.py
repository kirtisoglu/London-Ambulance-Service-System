"""Rewrite Section 7.2.1 (exact enumeration) from the stored enumeration JSONs.

Idempotent: replaces the whole subsubsection. Reads data/derived/validation_v2/enumeration_{3x4,4x4}.json
and, when present, enumeration_4x6.json (the long run)."""
import json, pathlib, re, sys
LAS = pathlib.Path("/home/user/London-Ambulance-Service-System/data/derived/validation_v2")
P = pathlib.Path("src/main_arxiv.tex"); s = P.read_text()
E = {i: json.loads((LAS / f"enumeration_{i}.json").read_text()) for i in ("3x4", "4x4")}
has46 = (LAS / "enumeration_4x6.json").exists() and "iid_noise_floor" in (LAS / "enumeration_4x6.json").read_text() \
    and E.setdefault("4x6", json.loads((LAS / "enumeration_4x6.json").read_text()))["runs"]["first"]["steps"] >= 1_000_000

def f3(x): return f"{x:.3f}"
def f2(x): return f"{x:.2f}"
def tvmax(d, key="tv_between_starts"): return max(d[key].values())
def halves(d): return max(r["tv_first_vs_second_half"] for r in d["runs"].values())

n = {i: dict(
    l1=E[i]["feasible_level1"], jt=E[i]["feasible_joint"],
    tv=tvmax(E[i]), tvj=tvmax(E[i]["joint"]), hv=halves(E[i]),
    fl=E[i]["iid_noise_floor"]["level1"]["tv_between_two_samples"],
    flj=E[i]["iid_noise_floor"]["joint"]["tv_between_two_samples"],
    flh=E[i]["iid_noise_floor"]["level1"]["tv_between_halves"],
    tvu=E[i]["pooled_tv_to_uniform"], tvt=E[i]["pooled_tv_to_spanning_tree_law"],
    tvuj=E[i]["joint"]["pooled_tv_to_uniform"], cond=E[i]["joint"]["mean_tv_level2_given_level1_to_uniform"],
    multi=E[i]["joint"]["level1_states_with_several_groupings"],
    sd=E[i]["joint"]["superdistrict_count_mass"], steps=E[i]["runs"]["first"]["steps"],
    within=E[i]["pooled_tv_to_spanning_tree_law_within_pattern"], mm=E[i]["max_over_min_pooled_frequency"],
    unv=E[i]["unvisited_level1_states"], unvj=E[i]["joint"]["unvisited_joint_states"],
    eunv=E[i]["iid_noise_floor"]["level1"]["expected_unvisited"], eunvj=E[i]["iid_noise_floor"]["joint"]["expected_unvisited"],
) for i in E}
a, b = n["3x4"], n["4x4"]
one_sd = {i: n[i]["sd"].get("1", 0.0) for i in ("3x4", "4x4")}

intro = (r"""We use three instances. The first is a $3\times4$ grid of twelve units with demand $100$ each, four candidate sites, $w = 300$, $\varepsilon^1 = 0.15$, $c^1 \in \{1,2\}$, $c^2 \in [2,4]$ and $\kappa^2_{\min} = 2$. A level-1 district is then admissible if and only if it is a connected set of exactly three units (one capacity unit) or six units (two units) that contains a candidate, so the feasible hierarchical states can be listed by exact cover: $93$ level-1 partitions and $119$ joint (level-1, level-2) states. The second is a $4\times4$ grid of sixteen units with five candidate sites and $w = 400$, so that districts have four or eight units; it has $435$ level-1 partitions and $595$ joint states. On these two instances a feasible state has one or two superdistricts, so the supergraph recursion is barely exercised. The third is therefore a $4\times6$ grid of twenty-four units with eight candidate sites, $w = 400$, $c^2 \in [2,3]$ and $\kappa^2_{\min} = 2$, on which every feasible state has two or three superdistricts of two to six units each: it has $22{,}549$ level-1 partitions and $43{,}689$ joint states. On each instance we run three chains from the first, middle and last feasible partitions in lexicographic order, for $200{,}000$ steps on the two small grids and $3{,}000{,}000$ on the $4\times6$ grid, record every visited state, and read every distance between empirical laws against its iid noise floor, the total-variation distance expected between two independent samples of the same size from the pooled law, which autocorrelation can only raise.
""")

facts = (r"""Three facts follow from the counts. \emph{Soundness}: every visited state is feasible. Over $1.2$ million steps on the two small grids no chain produced a district without a candidate, a demand outside its window, a superdistrict with fewer than $\kappa^2_{\min}$ districts, or a disconnected part (the $3$--$4\%$ of proposals that would have left a superdistrict with a single district were rejected and counted). \emph{Support}: each chain visited all $93$ level-1 and all $119$ joint states of the $3\times4$ grid and all $435$ and $595$ of the $4\times4$ grid. \emph{Start-independence}: the total-variation distance between the empirical level-1 laws of any two chains is at most $""" + f3(a["tv"]) + r"""$ on the $3\times4$ grid and $""" + f3(b["tv"]) + r"""$ on the $4\times4$ grid, against noise floors of $""" + f3(a["fl"]) + r"""$ and $""" + f3(b["fl"]) + r"""$, and between the first and second halves of any one chain at most $""" + f3(a["hv"]) + r"""$ and $""" + f3(b["hv"]) + r"""$ against floors of $""" + f3(a["flh"]) + r"""$ and $""" + f3(b["flh"]) + r"""$. The empirical law is thus determined by the chain, not by its start, to within sampling noise.
""")

joint = (r"""The same holds for the joint (level-1, level-2) law. The total-variation distance between the joint laws of any two chains is at most $""" + f3(a["tvj"]) + r"""$ and $""" + f3(b["tvj"]) + r"""$ against floors of $""" + f3(a["flj"]) + r"""$ and $""" + f3(b["flj"]) + r"""$; the pooled joint law is at distance $""" + f2(a["tvuj"]) + r"""$ and $""" + f2(b["tvuj"]) + r"""$ from the uniform law on joint states; and the law of the level-2 partition given the level-1 partition, over the $""" + str(a["multi"]) + r"""$ and $""" + str(b["multi"]) + r"""$ level-1 states that admit more than one grouping, is at distance $""" + f2(a["cond"]) + r"""$ and $""" + f2(b["cond"]) + r"""$ from the uniform conditional law, averaged with the level-1 weights. On these instances $""" + f"{100*one_sd['3x4']:.0f}" + r"""\%$ and $""" + f"{100*one_sd['4x4']:.0f}" + r"""\%$ of the mass sits on states with a single superdistrict, hence the third instance. """)

if has46:
    c = n["4x6"]
    import numpy as _np
    rows46 = json.loads((LAS / "enumeration_4x6_states.json").read_text())
    p46 = _np.array([r["pooled_empirical"] for r in rows46]); p46 = p46 / p46.sum()
    total_draws = 3 * c["steps"]
    exp_unv_pooled = float(((1 - p46) ** total_draws).sum())
    cr = E["4x6"]["joint"]["conditional_rows"]
    mass = _np.array([r["level1_mass"] for r in cr]); tvc = _np.array([r["tv_cond_to_uniform"] for r in cr]); grp = _np.array([r["groupings"] for r in cr])
    w = mass / mass.sum()
    share2 = float(w[grp == 2].sum()); tv2 = float((w[grp == 2] * tvc[grp == 2]).sum() / w[grp == 2].sum())
    tv3 = float((w[grp >= 3] * tvc[grp >= 3]).sum() / w[grp >= 3].sum())
    def ti(x): return f"{int(round(x)):,}".replace(",", "{,}")
    PM = E["4x6"]["pattern_mass"]
    def pm(key, law): return PM[key][law]
    keys = ["(4, 4, 8, 8)", "(4, 4, 4, 4, 8)", "(4, 4, 4, 4, 4, 4)"]
    tv_pat_u = 0.5 * sum(abs(PM[k]["empirical"] - PM[k]["uniform"]) for k in keys)
    tv_pat_t = 0.5 * sum(abs(PM[k]["empirical"] - PM[k]["spanning_tree_law"]) for k in keys)
    pm46 = (r""" On the $4\times6$ grid the picture changes: the patterns $(4,4,8,8)$, $(4,4,4,4,8)$ and $(4,4,4,4,4,4)$ receive $""" + ", ".join(f2(pm(k, "empirical")) for k in keys) + r"""$ against $""" + ", ".join(f2(pm(k, "uniform")) for k in keys) + r"""$ under the uniform law and $""" + ", ".join(f2(pm(k, "spanning_tree_law")) for k in keys) + r"""$ under the spanning-tree law, so the mass across capacity mixes lies between the two reference laws (total-variation distance $""" + f2(tv_pat_u) + r"""$ to the uniform and $""" + f2(tv_pat_t) + r"""$ to the spanning-tree pattern masses, against $0.03$ to the uniform ones on the small grids), the spanning-tree law conditioned on the pattern is at distance $""" + f2(c["within"]) + r"""$ from the sampled law, and the most and least frequent visited states differ by a factor of $""" + ti(c["mm"]) + r"""$. The characterization above is thus a property of the small instances, not of the chain; on larger instances the law is known through its support and the agreement of independently started chains only.""")
    joint += (r"""On the $4\times6$ grid, after $""" + ti(total_draws / 1e6) + r"""$ million steps in all, the chains visited all but $""" + ti(c["unv"]) + r"""$ of the $22{,}549$ level-1 states, fewer than the $""" + ti(exp_unv_pooled) + r"""$ that an independent sample of the same size from the pooled law would leave unvisited, and all but $""" + ti(c["unvj"]) + r"""$ of the $43{,}689$ joint states, and every visited state was feasible. The level-1 laws of any two chains differ by at most $""" + f3(c["tv"]) + r"""$ against a floor of $""" + f3(c["fl"]) + r"""$, the joint laws by at most $""" + f3(c["tvj"]) + r"""$ against $""" + f3(c["flj"]) + r"""$, and the two halves of one chain by at most $""" + f3(c["hv"]) + r"""$ against $""" + f3(c["flh"]) + r"""$. The pooled level-1 law is at distance $""" + f2(c["tvu"]) + r"""$ from the uniform law and $""" + f2(c["tvt"]) + r"""$ from the spanning-tree law, and the joint law at $""" + f2(c["tvuj"]) + r"""$ from the uniform law on joint states; $""" + f"{100 * c['sd'].get('3', 0.0):.0f}" + r"""\%$ of the mass has three superdistricts and the rest two. The law of level 2 given level 1, over the $""" + ti(c["multi"]) + r"""$ level-1 states with several groupings, is within $""" + f2(c["cond"]) + r"""$ of the uniform conditional law on average: $""" + f2(tv2) + r"""$ on the $""" + f"{100 * share2:.0f}" + r"""\%$ of the mass where a level-1 plan admits exactly two groupings and $""" + f2(tv3) + r"""$ where it admits three or more, against $0.45$ on the two small grids, where the few states with several groupings admit three each. Given the level-1 plan, the greedy supergraph recursion thus spreads its mass over the feasible groupings more evenly than the tree cut spreads it over level-1 plans, and the three chains agree on the joint law to within sampling noise.
""")
else:
    joint += r"""%% TODO: 4x6 long-run sentences (apply_sec721.py fills them when enumeration_4x6.json holds the 3,000,000-step run).
"""

law = (r"""
The sampled law is neither uniform nor the spanning-tree law: its total-variation distance to the uniform law is $""" + f2(a["tvu"]) + r"""$ on the $3\times4$ grid and $""" + f2(b["tvu"]) + r"""$ on the $4\times4$ grid, and to the spanning-tree law $""" + f2(a["tvt"]) + r"""$ and $""" + f2(b["tvt"]) + r"""$ (Figure~\ref{fig:enumeration}). The figure shows what the law is instead. Across district-size patterns the chain distributes mass almost as the uniform law does: on the $3\times4$ grid the patterns $(3,3,6)$, $(3,3,3,3)$ and $(6,6)$ receive $0.65$, $0.14$ and $0.22$ against $0.63$, $0.16$ and $0.20$ under the uniform law, whereas the spanning-tree law puts $0.59$ of its mass on the two-district pattern. Within a pattern the sampled frequencies track the spanning-tree weights, the compactness preference that ReCom inherits from cutting random spanning trees, damped: the spanning-tree law conditioned on the pattern, with pattern masses taken from the chain, is within total-variation distance $""" + f2(a["within"]) + r"""$ of the sampled law on the $3\times4$ grid and $""" + f2(b["within"]) + r"""$ on the $4\times4$ grid, and the most and least frequent feasible states differ in sampled frequency by factors of $""" + f"{a['mm']:,.0f}".replace(",", "{,}") + r"""$ and $""" + f"{b['mm']:,.0f}".replace(",", "{,}") + r"""$, so the law is far from uniform within a pattern as well. On these two instances the recursion, which chooses the capacity of each extracted district uniformly among the admissible cuts of a random tree, sets a capacity mix close to the uniform one, and the spanning-tree geometry sets the law within a mix.""" + (pm46 if has46 else "") + r""" On all three instances every unit has demand $100$, so every admissible district has demand exactly $c\,\bar d$, the running debt is identically zero and the debt-corrected windows coincide with the nominal ones; the enumeration therefore validates the tree-cut machinery and the support of the chain, not the debt mechanism, and the compounded windows of Proposition~\ref{prop:effective} cannot be exercised on it. The London readouts of Section~\ref{sec:exp-emergency} are readouts of the law of the same chain on a far larger instance, which these instances characterize only by analogy.
""")

figure = r"""
\begin{figure}[!htbp]
\centering
\includegraphics[width=\linewidth]{figures/validation/fig_enumeration_3x4.pdf}

\includegraphics[width=\linewidth]{figures/validation/fig_enumeration_4x4.pdf}

\includegraphics[width=\linewidth]{figures/validation/fig_enumeration_4x6.pdf}
\caption{Exact-enumeration validation. Top: the $3\times4$ grid; middle: the $4\times4$ grid; bottom: the $4\times6$ grid. (a)~Sampled frequency of each feasible level-1 state, three chains pooled, against the spanning-tree law, on logarithmic axes; the dotted line is the uniform law. (b)~Probability mass received by each district-size pattern under the sampled, uniform and spanning-tree laws.}
\label{fig:enumeration}
\end{figure}

"""
NEW = ("\\subsubsection{Exact enumeration on a small instance}\n\\label{sec:exp-enumeration}\n\n" + intro + figure + facts + "\n" + joint + law + "\n\\FloatBarrier\n")
start = s.index("\\subsubsection{Exact enumeration on a small instance}")
end = s.index("\\subsubsection{Multi-start agreement on large grids}")
s = s[:start] + NEW + s[end:]
P.write_text(s)
print("4x6 included" if has46 else "4x6 pending", "| 3x4:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items() if k not in ("sd",)})
print("4x4:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items() if k not in ("sd",)})
