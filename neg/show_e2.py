"""Print the recovery table of one e2 file. Usage: python -m neg.show_e2 <file.npz> [min layer] [max layer] [q]"""
import json, sys
import numpy as np

z = np.load(sys.argv[1], allow_pickle=True)
meta = json.loads(str(z["meta"])); s, sR, sP = z["s"], z["sR"], z["sP"]
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3]) if len(sys.argv) > 3 else 99
m = np.ones(len(sR), bool)
if len(sys.argv) > 4:
    m = np.array([int(i.split("-")[5]) for i in meta["ids"]]) == int(sys.argv[4])
rec = (np.nanmean(s[m], 0) - sR[m].mean()) / (sP[m].mean() - sR[m].mean())
cols, units = meta["cols"], meta["units"]
print(sys.argv[1], "n", int(m.sum()), "sR", round(float(sR[m].mean()), 2), "sP", round(float(sP[m].mean()), 2))
if meta["kind"] == "head":
    flat = [(rec[u, c], units[u], cols[c]) for u in range(len(units)) for c in range(len(cols)) if not np.isnan(rec[u, c])]
    for c in cols:
        x = sorted([f for f in flat if f[2] == c], key=lambda f: -abs(f[0]))[:14]
        print(f"{c:>5}: " + " ".join(f"L{u[0]}H{u[1]}:{v:+.2f}" for v, u, _ in x))
else:
    show = [i for i in range(len(cols)) if not np.all(np.isnan(rec[:, i])) and np.nanmax(np.abs(rec[:, i])) >= 0.04]
    print("layer " + " ".join(f"{cols[i]:>5}" for i in show))
    for u, (l, _) in enumerate(units):
        if lo <= l <= hi:
            print(f"{l:5d} " + " ".join("    ." if np.isnan(rec[u, i]) else f"{rec[u, i]:5.2f}" for i in show))
