"""Tables for the checks of 20 November: head ablation, H3, DAS seeds, quantifier patching, Hindi exception.
Usage: python -m neg.show_checks [results dir]"""
import glob, json, os, sys
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "results"


def ld(p):
    p = os.path.join(R, p)
    return json.load(open(p)) if os.path.exists(p) else None


print("== head sets: patch from the partner, and resample ablation (test split)")
for lang in ("hi", "fr"):
    d = ld(f"e2_4b/ablate_{lang}.json")
    if not d: continue
    print(f"{lang}: n={d['n']} clean logit difference {d['clean']:.2f}; resampled item has the same negated mention in {d['same_frac']:.2f}")
    for st, v in d["steps"].items():
        rows = v["rows"]
        rd = [x for x in rows if x["set"].startswith("rand")]
        for name, rs in (("top", rows[:1]), ("random", rd), ("all", rows[-1:])):
            print(f"  {st:4s} {name:7s} rec={np.mean([x['rec'] for x in rs]):+.3f}  resample: kept={np.mean([x['kept'] for x in rs]):.3f} "
                  f"acc={np.mean([x['acc'] for x in rs]):.2f} | other mention: kept={np.mean([x['kept_diff'] for x in rs]):+.3f} "
                  f"acc={np.mean([x['acc_diff'] for x in rs]):.2f} | same mention: kept={np.mean([x['kept_same'] for x in rs]):.3f}")
    h = d.get("h3")
    if not h: continue
    eff = np.array(h["eff"])
    d0, d1 = eff[:, 2], eff[:, 3]
    print(f"  H3: {len(eff)} heads, layers {h['layers'][0]}-{h['layers'][-1]}, base q0={h['base_q0']:.2f} q1={h['base_q1']:.2f}")
    print(f"    correlation of the effects on q0 and q1: {np.corrcoef(d0, d1)[0, 1]:.3f}; largest |d_q1 - d_q0|: {np.abs(d1 - d0).max():.2f}")
    print(f"    heads with d_q1 > 0.3: {(d1 > 0.3).sum()}, of which d_q0 < 0: {((d1 > 0.3) & (d0 < 0)).sum()}")
    for name, o in (("helps most on q1", np.argsort(-d1)[:6]), ("q1 minus q0 largest", np.argsort(-(d1 - d0))[:6]), ("hurts most on q1", np.argsort(d1)[:6])):
        print(f"    {name}: " + " ".join(f"L{int(eff[i, 0])}H{int(eff[i, 1])}:{d0[i]:+.2f}/{d1[i]:+.2f}" for i in o))
    for q, rows in h["curves"].items():
        print(f"    top sets ranked on q={q}: " + " ".join(f"k={r['k']}:{r['d_q0']:+.2f}/{r['d_q1']:+.2f}" for r in rows))

print("\n== DAS seeds (interchange accuracy)")
cells = {}
for lang, f0 in (("hi", ["e3/das_hi_a.json", "e3/das_hi_b.json"]), ("fr", ["e3/das_fr.json"])):
    files = [(0, f) for f in f0] + [(int(os.path.basename(f).split("_")[1][1:]), os.path.relpath(f, R)) for f in sorted(glob.glob(os.path.join(R, f"e3/seeds/{lang}_s*.json")))]
    for seed, f in files:
        d = ld(f)
        for r in d["runs"]:
            for name, e in r["eval"].items():
                cells.setdefault((lang, r["role"], r["layer"], r["var"], name), {})[seed] = e["das"]["iia"]
want = [("c12", 14, "W"), ("opt", 18, "Z"), ("opt", 19, "ANS"), ("end", 27, "ANS"), ("oA", 18, "Z")]
for (lang, role, layer, var, name), v in sorted(cells.items()):
    if (role, layer, var) in want and len(v) > 1:
        x = [v[s] for s in sorted(v)]
        print(f"  {lang} {role} L{layer} {var:3s} {name:16s} seeds {sorted(v)}: " + " ".join(f"{y:.2f}" for y in x) + f"  mean {np.mean(x):.2f} sd {np.std(x):.2f}")

print("\n== worlds DAS seeds")
for f in sorted(glob.glob(os.path.join(R, "e3w/w*_fact_*.json")) + glob.glob(os.path.join(R, "e3w/seeds/*.json"))):
    d = json.load(open(f))
    for r in d["runs"]:
        for name, e in r["eval"].items():
            s = " ".join(f"{k}={v['das']:.2f}" for k, v in e.items() if "das" in v)
            print(f"  {os.path.relpath(f, R)} seed {d['args']['seed']} L{r['layer']} {r['var']:5s} {name}: {s}")

print("\n== quantifier patching (recovery)")
for lang in ("hi", "fr"):
    p = os.path.join(R, f"e2q/quant_{lang}.npz")
    if not os.path.exists(p): continue
    z = np.load(p, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    s, sR, sP = z["s"], z["sR"], z["sP"]
    cols, layers, cs = meta["cols"], [u[0] for u in meta["units"]], np.array(meta["conds"])
    print(f"{lang}: correct on both items: {meta['correct']}")
    for cond in sorted(set(cs)):
        mk = cs == cond
        eff = (np.nanmean(s[mk], 0) - sR[mk].mean()) / (sP[mk].mean() - sR[mk].mean())
        out = []
        for c in ("qt", "f1b", "f1c", "f1e", "xn", "x", "sent", "ctx", "qn", "qst", "opt", "end"):
            i = cols.index(c)
            if np.all(np.isnan(eff[:, i])): continue
            j = int(np.nanargmax(eff[:, i]))
            last = max([l for l, e in zip(layers, eff[:, i]) if e >= 0.5], default=None)
            first = min([l for l, e in zip(layers, eff[:, i]) if e >= 0.5], default=None)
            out.append(f"{c}:{eff[j, i]:.2f}@{layers[j]}[{first}-{last}]")
        print(f"  {cond:18s} n={mk.sum():3d} " + " ".join(out))

d = ld("e1_except_4b.json")
if d:
    print("\n== Hindi exception wordings (accuracy, rate of yes)")
    for r in d["table"]:
        print(f"  {r['kind']:5s} {r['var']:6s} {r['who']:4s} n={r['n']} acc={r['acc']:.3f} yes={r['yes']:.3f}")
