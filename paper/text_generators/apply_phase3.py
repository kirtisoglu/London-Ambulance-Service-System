"""Fill the demand-proxy paragraph of Section 7.4 from the uniform-demand ensemble (idempotent)."""
import json, pathlib, re
import numpy as np
LAS = pathlib.Path("/home/user/London-Ambulance-Service-System/data/derived")
P = pathlib.Path("src/main_arxiv.tex"); s = P.read_text()
base = json.loads((LAS / "real_stations_u97/ensemble_real_stations.json").read_text())
uni = json.loads((LAS / "real_stations_u97_uniform/ensemble_real_stations.json").read_text())
ub = json.loads((LAS / "real_stations_u97_uniform/best_plans.json").read_text())
ux = json.loads((LAS / "real_stations_u97_uniform/london_extras.json").read_text())
of_b = base["stations"]["open_frequency_by_station"]; of_u = uni["stations"]["open_frequency_by_station"]
diff = {c: of_u[c] - of_b[c] for c in of_b}
mx = max(abs(v) for v in diff.values()); mean_abs = float(np.mean([abs(v) for v in diff.values()]))
n_big = sum(1 for v in diff.values() if abs(v) > 0.05)
set_b = {c for c, v in of_b.items() if v > 0.9}; set_u = {c for c, v in of_u.items() if v > 0.9}
import pandas as pd
names = dict(pd.read_csv("/home/user/London-Ambulance-Service-System/data/raw/LAS_stations.csv")[["station_code", "station_name"]].values)
leave = sorted(names[c] for c in set_b - set_u); join = sorted(names[c] for c in set_u - set_b)
movers = sorted(diff.items(), key=lambda kv: -abs(kv[1]))[:4]
mov_txt = ", ".join(f"{names[c]} ({of_b[c]*100:.0f}\\% to {of_u[c]*100:.0f}\\%)" for c, v in movers)
acc = uni["acceptance_rate"]; accs = sorted(acc.values())
sl = uni["s_LAS"]; n_nom = round(sl["stations_outside_unit_window_share"] * 66)
E_u = uni["ensemble_E"]; ems_u = uni["ensemble_ems"]
ks = {m: uni["post_burn_in"][m]["ks_max"] for m in ("E", "P1", "cut_edges")}
g = uni["groups"]
set_change = ("the set of stations open in more than $90\\%$ of plans is unchanged" if not leave and not join else
              "the set of stations open in more than $90\\%$ of plans " + ("loses " + ", ".join(leave) if leave else "") +
              (" and " if leave and join else "") + ("gains " + ", ".join(join) if join else ""))
para = (r"""Nor does it depend on the within-borough apportionment, as far as a second proxy can tell. We repeat the four chains with each borough's incidents spread uniformly over its LSOAs instead of in proportion to population, the crudest alternative, with the same calibration ($97$ units) and protocol. """
        + f"Acceptance is ${100*accs[0]:.1f}\\%$ to ${100*accs[-1]:.1f}\\%$ and the cross-chain Kolmogorov--Smirnov distances are ${ks['E']:.3f}$ for the energy, ${ks['P1']:.3f}$ for the number of stations and ${ks['cut_edges']:.3f}$ for the cut-edge count. "
        + f"The plans open ${uni['stations']['n_open_per_plan_mean']:.1f}$ stations on average, the open frequency of a station moves by at most ${mx:.2f}$ (mean ${mean_abs:.2f}$ over stations), ${n_big}$ stations move by more than $0.05$, the largest movers being {mov_txt}, and {set_change}. "
        + f"Two stations of the same real Group share a superdistrict in ${100*g['co_membership_mean_same_real_group']:.0f}\\%$ of plans in which both are open and the adjusted Rand index against the real Groups is ${g['ari_vs_real_groups_mean']:.2f}$. "
        + f"Under this demand the operational layout has ${n_nom}$ stations outside the nominal windows and loads from ${sl['station_load_in_units_range'][0]:.2f}$ to ${sl['station_load_in_units_range'][1]:.2f}$ units, its energy is ${sl['E']/1e6:.2f}$ million against an ensemble mean of ${E_u['mean']/1e6:.2f}$ million (lowest stored plan ${ub['sample_extremes']['E_min']/1e6:.2f}$ million), and its mean drive time is ${sl['ems']['T_mean']:.1f}$ minutes against ${ems_u['T_mean']['mean']:.1f}$. "
        + "The frequencies move, as they must when the demand surface moves, but the structure of the ensemble and the position of the layout below it do not.")
marker = "%% PHASE3-PARAGRAPH"
if marker in s:
    s = s.replace(marker, para + "\n%% PHASE3-DONE")
else:
    s = re.sub(r"Nor does it depend on the within-borough apportionment.*?\n%% PHASE3-DONE", lambda m: para + "\n%% PHASE3-DONE", s, flags=re.S)
P.write_text(s); print(para)
