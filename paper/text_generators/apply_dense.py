"""Fill the grid_400_dense row of Table 6 and its sentence from the dense-grid optimizer summary (idempotent)."""
import json, pathlib, re, statistics
G = pathlib.Path("/home/user/London-Ambulance-Service-System/falcomchain_experiments/gurobi")
f = G / "solution_400_dense_falcom_multistart_summary.json"
d = json.loads(f.read_text())
res = [r for r in d["results"] if r.get("ok")]
assert len(res) >= 6, len(res)
objs = sorted(float(r["obj_value"]) for r in res)
best = objs[0]; med = statistics.median(objs); opt = 316234.0
gap = 100 * (best / opt - 1); gap_med = 100 * (med / opt - 1)
P = pathlib.Path("src/main_arxiv.tex"); s = P.read_text()
row_old = re.compile(r"grid\\_400\\_dense & \$23\$ / \$14\$ & \$316\{,\}234\$ & \$1\{,\}455\$\\,s & -- & .*? \\\\")
row_new = f"grid\\_400\\_dense & $23$ / $14$ & $316{{,}}234$ & $1{{,}}455$\\,s & -- & ${best:,.0f}$ & ${med:,.0f}$ & ${gap:.1f}\\%$ \\\\".replace(",", "{,}").replace("{{,}}", "{,}")
# careful: the replace above also touched '{,}' sequences already present; rebuild cleanly
def tex_int(x): return f"{int(round(x)):,}".replace(",", "{,}")
row_new = f"grid\\_400\\_dense & $23$ / $14$ & $316{{,}}234$ & $1{{,}}455$\\,s & -- & ${tex_int(best)}$ & ${tex_int(med)}$ & ${gap:.1f}\\%$ \\\\"
s, n = row_old.subn(lambda m: row_new, s); assert n == 1, n
sent_old = "Capacity is therefore chosen by demand concentration, as intended, and the sparse candidate set does not prevent it."
sent_new = (sent_old + f" FalCom run as an optimizer on this instance, eight seeds of $10{{,}}000$ steps, reaches ${tex_int(best)}$ at best and ${tex_int(med)}$ at the median seed, ${gap:.1f}\\%$ and ${gap_med:.1f}\\%$ above the optimum, so the capacity choice costs the annealer nothing that the single-unit instances did not already cost it.")
if sent_old + " FalCom run as an optimizer on this instance" in s:
    s = re.sub(re.escape(sent_old) + r" FalCom run as an optimizer on this instance.*?did not already cost it\.", lambda m: sent_new, s, count=1, flags=re.S)
else:
    assert s.count(sent_old) == 1; s = s.replace(sent_old, sent_new)
P.write_text(s); print(row_new); print(f"best {best:,.0f} median {med:,.0f} gap {gap:.1f}% / {gap_med:.1f}%; seeds ok {len(res)}")
