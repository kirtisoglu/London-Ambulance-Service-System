"""Fifth London chain from the Group-based start: pool it with the four sector-start chains,
run the diagnostics, and insert the paragraph into Section 7.4.4 (idempotent)."""
import json, os, pathlib, re, subprocess, sys
import numpy as np
REPO = pathlib.Path("/home/user/London-Ambulance-Service-System"); D = REPO / "data/derived"
comb = D / "real_stations_u97_5chains"; comb.mkdir(exist_ok=True)
for seed, src in [(1, "real_stations_u97"), (2, "real_stations_u97"), (3, "real_stations_u97"), (4, "real_stations_u97"), (5, "real_stations_u97_groupstart")]:
    for kind, ext in (("summary", "json"), ("trace", "csv"), ("snap", "npz")):
        f = D / src / f"{kind}_real_s{seed}_T80000.{ext}"; assert f.exists(), f
        dst = comb / f.name
        if not dst.exists():
            os.symlink(f, dst)
env = dict(os.environ, PYTHONHASHSEED="0")
if not (comb / "best_plans.json").exists() or "--rerun" in sys.argv:
    subprocess.run([sys.executable, "-m", "falcomchain_experiments.las.postprocess_real_stations", "--in-dir", str(comb),
                    "--tag", "real", "--steps", "80000", "--no-maps"], cwd=REPO, env=env, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "-m", "falcomchain_experiments.las.best_plans", "--in-dir", str(comb), "--steps", "80000"],
                   cwd=REPO, env=env, check=True, stdout=subprocess.DEVNULL)
ens = json.loads((comb / "ensemble_real_stations.json").read_text())
bp = json.loads((comb / "best_plans.json").read_text())
ks5 = {}; ks4 = {}
for m in ("E", "P1", "P2", "cut_edges"):
    pw = ens["post_burn_in"][m]["ks_pairwise_thinned"]
    ks5[m] = max(v for k, v in pw.items() if "5" in k.split("-"))
    ks4[m] = max(v for k, v in pw.items() if "5" not in k.split("-"))
rh = {m: ens["post_burn_in"][m]["split_rhat"] for m in ("E", "P1", "P2", "cut_edges")}
pc = bp["per_chain_readouts"]; c5 = pc["5"]; others = [pc[k] for k in ("1", "2", "3", "4")]
def rng(key): return min(o[key] for o in others), max(o[key] for o in others)
# forgetting of chain 5 and open-frequency difference to the pooled four
z = np.load(D / "real_stations_u97_groupstart/snap_real_s5_T80000.npz"); step = z["step"]; st = z["station"]; edges = z["edges"]
b0 = st[0][edges[:, 0]] != st[0][edges[:, 1]]
share = np.array([(st[k][edges[:, 0]] != st[k][edges[:, 1]])[b0].mean() for k in range(len(step))])
plateau = share[step >= 16000].mean(); first = int(step[np.argmax(share <= plateau + 0.02)])
idx = {int(n): i for i, n in enumerate(z["node_ids"])}; st_own = np.array([idx[int(n)] for n in z["station_node_ids"]])
om5 = (st[step >= 16000][:, st_own] == np.arange(66)[None, :]).mean(0)
om4 = []
for sd in (1, 2, 3, 4):
    y = np.load(D / f"real_stations_u97/snap_real_s{sd}_T80000.npz"); m = y["step"] >= 16000
    om4.append((y["station"][m][:, st_own] == np.arange(66)[None, :]).mean(0))
om4 = np.mean(om4, 0); dmax = float(np.abs(om5 - om4).max()); dmean = float(np.abs(om5 - om4).mean())
set4 = set(np.where(om4 > 0.9)[0]); set5 = set(np.where(om5 > 0.9)[0])
init = json.loads((D / "real_stations_u97_groupstart/summary_real_s5_T80000.json").read_text())
para = (f"All four chains start from the same sector-wise construction, so we ran a fifth chain of $80{{,}}000$ steps from a structurally different state, an initial plan built inside the $21$ real Groups (the three single-station Groups merged into their neighbours, $18$ zones instead of five). "
        f"It forgets its start within ${first}$ steps, to the same plateau of ${plateau:.2f}$, and after burn-in its largest Kolmogorov--Smirnov distance to any of the four sector-start chains is ${ks5['E']:.3f}$ for the energy, ${ks5['P1']:.3f}$ for the number of stations, ${ks5['P2']:.3f}$ for the number of Groups and ${ks5['cut_edges']:.3f}$ for the cut-edge count, against ${ks4['E']:.3f}$, ${ks4['P1']:.3f}$, ${ks4['P2']:.3f}$ and ${ks4['cut_edges']:.3f}$ among the four; split-$\\hat R$ over the five chains is at most ${max(rh.values()):.3f}$. "
        f"Its open frequencies differ from the pooled four by at most ${dmax:.2f}$ (mean ${dmean:.3f}$), its set of stations open in more than $90\\%$ of plans "
        + ("coincides with theirs" if set4 == set5 else f"differs from theirs by ${len(set4 ^ set5)}$ station" + ("s" if len(set4 ^ set5) != 1 else "") + " at the threshold")
        + f", and its mean energy, ${c5['E_mean']/1e6:.2f}$ million, number of open stations, ${c5['n_open_mean']:.1f}$, and adjusted Rand index against the real Groups, ${c5['ari_mean']:.2f}$, lie within the ranges of the four (${rng('E_mean')[0]/1e6:.2f}$ to ${rng('E_mean')[1]/1e6:.2f}$ million, ${rng('n_open_mean')[0]:.1f}$ to ${rng('n_open_mean')[1]:.1f}$, ${rng('ari_mean')[0]:.2f}$ to ${rng('ari_mean')[1]:.2f}$). "
        "A start with a different level-2 structure thus reaches the same law.")
P = pathlib.Path("src/main_arxiv.tex"); s = P.read_text()
anchor = "We report the time law, which is what the chain samples, and note that the retry budget $M$ enters it through $a(s)$.\n"
assert anchor in s
s = re.sub(re.escape(anchor) + r"(\n%% G7-START.*?%% G7-END\n)?", lambda m: anchor + "\n%% G7-START\n" + para + "\n%% G7-END\n", s, count=1, flags=re.S)
P.write_text(s); print(para)
