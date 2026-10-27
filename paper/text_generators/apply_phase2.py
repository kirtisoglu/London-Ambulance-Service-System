"""Fill the annealer paragraph of Section 7.4 from optimizer_summary_T50000.json (idempotent)."""
import json, pathlib, re, statistics
LAS = pathlib.Path("/home/user/London-Ambulance-Service-System/data/derived")
P = pathlib.Path("src/main_arxiv.tex"); s = P.read_text()
d = json.loads((LAS / "real_stations_u97_opt/optimizer_summary_T50000.json").read_text())
runs = sorted(d["runs"], key=lambda r: r["best_E"])
assert len(runs) == 8, len(runs)
slas = json.loads((LAS / "real_stations_u97/ensemble_real_stations.json").read_text())["s_LAS"]
bp = json.loads((LAS / "real_stations_u97/best_plans.json").read_text())
best = runs[0]
E_lo, E_hi = runs[0]["best_E"] / 1e6, runs[-1]["best_E"] / 1e6
E_med = statistics.median(r["best_E"] for r in runs) / 1e6
acc = [r["sa"]["accepted"] for r in runs]; wors = [r["sa"]["worsening_accepted"] for r in runs]
mins = [r["runtime_seconds"] / 60 for r in runs]
gap = best["best_E"] - slas["E"]; gap1 = best["best_E1"] - slas["E1"]
pct = 100 * gap / slas["E"]
below = 100 * (bp["sample_extremes"]["E_min"] - best["best_E"]) / bp["sample_extremes"]["E_min"]
dT = best["T_mean"] - 2.92; dT90 = best["T90"] - 4.78; dC8 = 100 * (0.995 - best["C8"])
para = (r"""To tighten the bound we run FalCom in the optimizer mode of Section~\ref{sec:exp-milp} on this instance, eight seeds of $50{,}000$ steps with $E$ as the objective and the calibration unchanged, """
        + f"${min(mins):.0f}$ to ${max(mins):.0f}$ minutes per seed on one core. The annealer accepts few moves here, ${min(acc)}$ to ${max(acc)}$ per seed of which ${min(wors)}$ to ${max(wors)}$ worsen the energy, and its best plans lie between ${E_lo:.2f}$ and ${E_hi:.2f}$ million with a median of ${E_med:.2f}$ million. The best of the eight has energy ${best['best_E']/1e6:.2f}$ million, ${best['best_E1']/1e6:.2f}$ million on the station leg, with ${best['open_stations']}$ open stations, ${below:.1f}\\%$ below the lowest stored plan of the sampling runs. Against it the bound tightens to ${gap/1e6:.2f}$ million demand-weighted minutes, ${pct:.0f}\\%$ of the layout's energy, of which ${gap1/1e6:.2f}$ million is on the station leg, and in the station-leg metrics to ${dT:.1f}$ minutes of mean drive time (${best['T_mean']:.2f}$ against $2.92$), ${dT90:.1f}$ minutes at the $90$th percentile (${best['T90']:.2f}$ against $4.78$) and ${dC8:.1f}$ points of eight-minute share (${100*best['C8']:.1f}\\%$ against $99.5\\%$). The operational layout remains below every balanced plan found, and the bound remains an upper bound: a better optimizer would lower it further.")
marker = "%% PHASE2-PARAGRAPH"
if marker in s:
    s = s.replace(marker, para + "\n%% PHASE2-DONE")
else:
    s = re.sub(r"To tighten the bound we run FalCom in the optimizer mode.*?\n%% PHASE2-DONE", lambda m: para + "\n%% PHASE2-DONE", s, flags=re.S)
P.write_text(s); print(para[:400])
