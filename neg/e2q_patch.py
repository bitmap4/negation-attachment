"""E2 on the quantifier set: residual patching between two quantifier sentences.

R and D have the same persons, object, question and option order. They differ in the quantifier:
  all <-> none      "everyone likes X" / "nobody likes X"
  allx <-> nonex    the same with an exception ("except Z"); -exc: the question is about Z, -rest: about another person
The answer of D is the opposite of the answer of R. Positions are aligned by role.
recovery = (s_patched - s_R) / (s_D - s_R), where s is the logit difference toward the letter of D.

Usage: python -m neg.e2q_patch <out.npz> --lang hi [--n 24 --split test --stride 2]
"""
import argparse, json, os, random, time
import numpy as np
import torch
from . import common as C

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--lang", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=24, help="pairs for each condition")
ap.add_argument("--stride", type=int, default=2)
ap.add_argument("--pb", type=int, default=4)
ap.add_argument("--bs", type=int, default=48)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

tok = C.get_tok()
net = C.Net(C.get_model())
A, B = C.letter_ids(tok)
items = C.load(f"{a.lang}_quant_{a.split}")
grp = {}
for it in items:
    grp.setdefault((it["a"], it["b"], it["c"], it["obj"], it["f"], it["qi"], it["ord"]), {})[it["kind"]] = it


def sgn(it): return 1.0 if it["ans"] == "A" else -1.0


def base_ld(its):
    e = C.encode(tok, its)
    out = []
    for b in C.batches(len(its), 32):
        ids, mask, _ = C.pad(tok, [e[i][0] for i in b])
        lg, _ = net.run(ids, mask)
        out += (lg[:, A] - lg[:, B]).tolist()
    return out


cand = []
for key, g in grp.items():
    for k1, k2 in (("all", "none"), ("none", "all"), ("allx", "nonex"), ("nonex", "allx")):
        if k1 in g and k2 in g:
            cond = f"{k1}>{k2}" + ("" if k1 in ("all", "none") else "-exc" if key[5] == 2 else "-rest")
            cand.append((g[k1], g[k2], cond))
R = random.Random(a.seed)
R.shuffle(cand)
ldR, ldD = base_ld([c[0] for c in cand]), base_ld([c[1] for c in cand])
cnt, tot, trip, sR, sD = {}, {}, [], [], []
for i, (r, d, cond) in enumerate(cand):
    ok = ldR[i] * sgn(r) > 0 and ldD[i] * sgn(d) > 0
    tot.setdefault(cond, [0, 0]); tot[cond][0] += ok; tot[cond][1] += 1
    if ok and cnt.get(cond, 0) < a.n:
        cnt[cond] = cnt.get(cond, 0) + 1
        trip.append((r, d, cond)); sR.append(ldR[i] * sgn(d)); sD.append(ldD[i] * sgn(d))
print("correct on R and D:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(tot.items())}, flush=True)
sR, sD = np.array(sR), np.array(sD)
n = len(trip)
conds = [t[2] for t in trip]

single = ["t", "xn", "x", "qt", "f1b", "f1c", "f1e", "q", "qn", "qo", "lA", "oA", "lB", "oB", "tail", "end"]
wide = {"sent": ["xn", "x", "qt", "f1b", "f1c", "f1e"], "ctx": ["t", "xn", "x", "qt", "f1b", "f1c", "f1e"],
        "qst": ["q", "qn", "qo"], "opt": ["lA", "oA", "lB", "oB"]}
wide["post"] = wide["qst"] + wide["opt"] + ["tail", "end"]
cols = single + list(wide)
groups = [[r] for r in single] + list(wide.values())
layers = list(range(0, net.L + 1, a.stride))
res = np.full((n, len(layers), len(cols)), np.nan, dtype=np.float32)
encR, encD = C.encode(tok, [t[0] for t in trip]), C.encode(tok, [t[1] for t in trip])
order = sorted(range(n), key=lambda i: len(encR[i][0]))
t0 = time.time()
for ci, b in enumerate(C.batches(n, a.pb)):
    idx = [order[i] for i in b]
    idsD, maskD, padD = C.pad(tok, [encD[i][0] for i in idx])
    _, cache = net.run(idsD, maskD, read=[("resid", l) for l in layers])
    idsR, maskR, padR = C.pad(tok, [encR[i][0] for i in idx])
    runs = []
    for j, i in enumerate(idx):
        for c, g in enumerate(groups):
            pr, pd = C.align(encR[i][1], encD[i][1], g)
            if pr:
                runs.append((j, i, c, [p + padR[j] for p in pr], [p + padD[j] for p in pd]))
    sign = torch.tensor([sgn(trip[i][1]) for i in idx], device="cuda")
    for ui, l in enumerate(layers):
        src = cache[("resid", l)]
        for rb in C.batches(len(runs), a.bs):
            rr = [runs[x] for x in rb]
            rows = torch.tensor([r[0] for r in rr], device="cuda")
            bi = torch.tensor([x for x, r in enumerate(rr) for _ in r[3]], device="cuda")
            pr = torch.tensor([p for r in rr for p in r[3]], device="cuda")
            sj = torch.tensor([r[0] for r in rr for _ in r[3]], device="cuda")
            pd = torch.tensor([p for r in rr for p in r[4]], device="cuda")

            def fn(x):
                x = x.clone()
                x[bi, pr] = src[sj, pd].to(x.dtype)
                return x
            lg, _ = net.run(idsR[rows], maskR[rows], write={("resid", l): fn})
            s = (lg[:, A] - lg[:, B]) * sign[rows]
            for x, r in enumerate(rr):
                res[r[1], ui, r[2]] = s[x].item()
    if ci % 8 == 0:
        print(f"chunk {ci} items {min(n, (ci + 1) * a.pb)}/{n} {time.time() - t0:.0f} s", flush=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
meta = dict(vars(a), kind="resid", model=C.MODEL, cols=cols, units=[(l, None) for l in layers], conds=conds,
            correct={k: v for k, v in tot.items()})
np.savez_compressed(a.out, s=res, sR=sR, sP=sD, meta=json.dumps(meta))
cs = np.array(conds)
for cond in sorted(set(conds)):
    mk = cs == cond
    eff = (np.nanmean(res[mk], 0) - sR[mk].mean()) / (sD[mk].mean() - sR[mk].mean())
    print(f"== {cond} n={mk.sum()} sR={sR[mk].mean():.2f} sD={sD[mk].mean():.2f}")
    show = [i for i in range(len(cols)) if not np.all(np.isnan(eff[:, i])) and np.nanmax(np.abs(eff[:, i])) >= 0.1]
    print("layer " + " ".join(f"{cols[i]:>5}" for i in show))
    for u, l in enumerate(layers):
        print(f"{l:5d} " + " ".join("    ." if np.isnan(eff[u, i]) else f"{eff[u, i]:5.2f}" for i in show))
print(f"time {time.time() - t0:.0f} s")
