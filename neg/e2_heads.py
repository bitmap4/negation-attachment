"""E2: which heads move the attachment? Head ranking and cumulative head sets.

For the heads of some layers and one group of destination tokens:
  1. rank the heads by the recovery of a patch of that head alone (on the first half of the pairs);
  2. on the other half, patch the top k heads together ("only"), and all heads of these layers
     except the top k ("except"). A small k with a high "only" value is a small sufficient set;
     a low "except" value shows that the set is necessary.

Usage: python -m neg.e2_heads <out.json> --lang hi --layers 17-19 --dest opt [--n 64]
"""
import argparse, json, os, time
import numpy as np
import torch
from . import common as C
from .pairs import build, sgn

DEST = {"opt": ["lA", "oA", "lB", "oB"], "oA": ["oA"], "qo": ["q", "qcue", "lA", "oA", "lB", "oB"], "lB": ["lB"],
        "lbt": ["lB", "oB", "tail"], "end": ["end"], "post": ["q", "qcue", "lA", "oA", "lB", "oB", "tail", "end"]}

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--lang", required=True)
ap.add_argument("--mode", default="neg")
ap.add_argument("--layers", required=True, help="first-last")
ap.add_argument("--dest", required=True)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--ks", default="1,2,4,8,16,32,64")
ap.add_argument("--bs", type=int, default=40)
a = ap.parse_args()

tok = C.get_tok()
net = C.Net(C.get_model())
A, B = C.letter_ids(tok)
trip, sR, sP = build(tok, net, a.lang, "disc", a.mode, a.n)
n = len(trip)
lo, hi = (int(x) for x in a.layers.split("-"))
layers = list(range(lo, hi + 1))
heads = [(l, h) for l in layers for h in range(net.H)]
encR, encD = C.encode(tok, [t[0] for t in trip]), C.encode(tok, [t[2] for t in trip])
H, dh = net.H, net.dh


def patched(idx, sets):
    """Scores of the pairs idx; sets[x] is the list of heads (layer, head) to patch for pair idx[x]."""
    idsD, maskD, padD = C.pad(tok, [encD[i][0] for i in idx])
    _, cache = net.run(idsD, maskD, read=[("z", l) for l in layers])
    idsR, maskR, padR = C.pad(tok, [encR[i][0] for i in idx])
    write = {}
    for l in layers:
        bi, pr, sj, pd, hh = [], [], [], [], []
        for x, i in enumerate(idx):
            hs = [h for (ll, h) in sets[x] if ll == l]
            if not hs: continue
            p, q = C.align(encR[i][1], encD[i][1], DEST[a.dest])
            for u, v in zip(p, q):
                for h in hs:
                    bi.append(x); pr.append(u + padR[x]); sj.append(x); pd.append(v + padD[x]); hh.append(h)
        if not bi: continue
        t = [torch.tensor(v, device="cuda") for v in (bi, pr, sj, pd, hh)]

        def fn(x, t=t, l=l):
            x4 = x.clone().view(x.shape[0], x.shape[1], H, dh)
            s4 = cache[("z", l)].view(x.shape[0], -1, H, dh)
            x4[t[0], t[1], t[4]] = s4[t[2], t[3], t[4]].to(x.dtype)
            return x4.view(x.shape)
        write[("z", l)] = fn
    lg, _ = net.run(idsR, maskR, write=write)
    sign = torch.tensor([sgn(trip[i][1]) for i in idx], device="cuda")
    return ((lg[:, A] - lg[:, B]) * sign).tolist()


def rec(s, idx):
    return float((np.mean(s) - sR[idx].mean()) / (sP[idx].mean() - sR[idx].mean()))


half = n // 2
rank_idx, eval_idx = list(range(half)), list(range(half, n))
t0 = time.time()
# 1. one head at a time
eff = {}
jobs = [(i, hd) for hd in heads for i in rank_idx]
tot = {hd: [] for hd in heads}
for b in C.batches(len(jobs), a.bs):
    jb = [jobs[x] for x in b]
    s = patched([j[0] for j in jb], [[j[1]] for j in jb])
    for j, v in zip(jb, s): tot[j[1]].append(v)
ri = np.array(rank_idx)
for hd in heads:
    eff[hd] = rec(tot[hd], ri)
order = sorted(heads, key=lambda hd: -eff[hd])
print(f"ranked {len(heads)} heads in {time.time() - t0:.0f} s; top:", " ".join(f"L{l}H{h}:{eff[(l, h)]:+.3f}" for l, h in order[:16]), flush=True)
# 2. sets of heads
ks = [k for k in (int(x) for x in a.ks.split(",")) if k < len(heads)] + [len(heads)]
ei = np.array(eval_idx)
curve = []
for k in ks:
    row = {"k": k}
    for name, hs in (("only", order[:k]), ("except", order[k:])):
        s = []
        for b in C.batches(len(eval_idx), a.bs):
            idx = [eval_idx[x] for x in b]
            s += patched(idx, [hs] * len(idx)) if hs else [float(sR[i]) for i in idx]
        row[name] = rec(s, ei)
    curve.append(row)
    print(f"k={k:4d} only={row['only']:.3f} except={row['except']:.3f}", flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump({"args": vars(a), "model": C.MODEL, "n_rank": half, "n_eval": n - half,
           "rank": [[l, h, eff[(l, h)]] for l, h in order], "curve": curve}, open(a.out, "w"), indent=1)
