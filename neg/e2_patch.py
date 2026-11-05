"""E2: activation patching with positions aligned by role.

The donor D gives activations; the recipient R receives them. P is the minimal partner of R
(the same words, with the negation in the other clause). The score s is the logit difference
toward the letter that is correct for P (the opposite of the correct letter for R), and
    recovery = (s_patched - s_R) / (s_P - s_R).
Modes: neg, other, ord, q, name (see pairs.py).
Kinds: resid, attn, mlp (layer x role), head (head x role group), and attn_in: the input of the
attention of one layer at the role tokens (a path patch: the residual stream itself is not changed).

Usage: python -m neg.e2_patch <out.npz> --lang hi --mode neg --kind resid [--n 64]
"""
import argparse, json, os, time
import numpy as np
import torch
from . import common as C
from .pairs import build, sgn

ROLES = ["ins", "n1", "body1", "cue", "e1", "conj", "n2", "body2", "e2", "q", "qcue", "lA", "oA", "lB", "oB", "tail", "end"]
GROUPS = {"ctx": ["n1", "body1", "cue", "e1", "conj", "n2", "body2", "e2"], "q": ["q", "qcue"],
          "opt": ["lA", "oA", "lB", "oB"], "tail": ["tail"], "end": ["end"]}
GROUPS["all"] = sum(GROUPS.values(), [])
# groups of roles for the layer x role maps. One role alone can fail when the same fact is in more positions.
C1, C2, QO = ["n1", "body1", "e1"], ["n2", "body2", "e2"], ["q", "qcue", "lA", "oA", "lB", "oB"]
WIDE = {"c1": C1, "c2": C2, "c12": C1 + ["conj"] + C2, "ctx": C1 + ["conj", "cue"] + C2, "opt": ["lA", "oA", "lB", "oB"],
        "qo": QO, "post": QO + ["tail", "end"]}

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--lang", required=True)
ap.add_argument("--set", default="disc")
ap.add_argument("--mode", default="neg")
ap.add_argument("--kind", default="resid")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--pb", type=int, default=4, help="pairs in one chunk")
ap.add_argument("--bs", type=int, default=72, help="sequences in one forward pass")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

tok = C.get_tok()
net = C.Net(C.get_model())
A, B = C.letter_ids(tok)
trip, sR, sP = build(tok, net, a.lang, a.set, a.mode, a.n, a.seed)
n = len(trip)

if a.kind == "head":
    cols = list(GROUPS)
    groups = [GROUPS[c] for c in cols]
    sites = [("z", l) for l in range(net.L)]
    units = [(l, h) for l in range(net.L) for h in range(net.H)]
else:
    cols = ROLES + list(WIDE) + (["all"] if a.kind in ("attn", "mlp") else [])
    groups = [[r] for r in ROLES] + list(WIDE.values()) + ([GROUPS["all"] + ["ins"]] if a.kind in ("attn", "mlp") else [])
    top = net.L + 1 if a.kind == "resid" else net.L
    sites = [(a.kind, l) for l in range(0, top)]
    units = [(l, None) for l in range(0, top)]

res = np.full((n, len(units), len(cols)), np.nan, dtype=np.float32)
encR = C.encode(tok, [t[0] for t in trip])
encD = C.encode(tok, [t[2] for t in trip])
order = sorted(range(n), key=lambda i: len(encR[i][0]))
t0 = time.time()
for ci, b in enumerate(C.batches(n, a.pb)):
    idx = [order[i] for i in b]
    idsD, maskD, padD = C.pad(tok, [encD[i][0] for i in idx])
    _, cache = net.run(idsD, maskD, read=sites)
    idsR, maskR, padR = C.pad(tok, [encR[i][0] for i in idx])
    # one run for each (pair, column): rows of the recipient batch, and where to copy from
    runs = []
    for j, i in enumerate(idx):
        for c, g in enumerate(groups):
            pr, pd = C.align(encR[i][1], encD[i][1], g)
            if pr:
                runs.append((j, i, c, [p + padR[j] for p in pr], [p + padD[j] for p in pd]))
    sign = torch.tensor([sgn(trip[i][1]) for i in idx], device="cuda")
    for ui, (l, h) in enumerate(units):
        site = ("z", l) if a.kind == "head" else (a.kind, l)
        src = cache[site]
        for rb in C.batches(len(runs), a.bs):
            rr = [runs[x] for x in rb]
            rows = torch.tensor([r[0] for r in rr], device="cuda")
            bi = torch.tensor([x for x, r in enumerate(rr) for _ in r[3]], device="cuda")
            pr = torch.tensor([p for r in rr for p in r[3]], device="cuda")
            sj = torch.tensor([r[0] for r in rr for _ in r[3]], device="cuda")
            pd = torch.tensor([p for r in rr for p in r[4]], device="cuda")

            def fn(x, h=h):
                x = x.clone()
                if h is None: x[bi, pr] = src[sj, pd].to(x.dtype)
                else:
                    s = slice(h * net.dh, (h + 1) * net.dh)
                    x[bi, pr, s] = src[sj, pd, s].to(x.dtype)
                return x
            lg, _ = net.run(idsR[rows], maskR[rows], write={site: fn})
            s = (lg[:, A] - lg[:, B]) * sign[rows]
            for x, r in enumerate(rr):
                res[r[1], ui, r[2]] = s[x].item()
    if ci % 4 == 0:
        print(f"chunk {ci} pairs {min(n, (ci + 1) * a.pb)}/{n} {time.time() - t0:.0f} s", flush=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
meta = dict(vars(a), model=C.MODEL, cols=cols, units=units, ids=[t[0]["id"] for t in trip], donors=[t[2]["id"] for t in trip])
np.savez_compressed(a.out, s=res, sR=sR, sP=sP, meta=json.dumps(meta))
rec = (np.nanmean(res, 0) - sR.mean()) / (sP.mean() - sR.mean())       # ratio of means
print(f"n={n} sR={sR.mean():.2f} sP={sP.mean():.2f}  time {time.time() - t0:.0f} s")
if a.kind == "head":
    flat = [(rec[u, c], units[u], cols[c]) for u in range(len(units)) for c in range(len(cols)) if not np.isnan(rec[u, c])]
    for name, key in (("highest", lambda x: -x[0]), ("lowest", lambda x: x[0])):
        print(name, " ".join(f"L{u[0]}H{u[1]}@{c}:{v:.2f}" for v, u, c in sorted(flat, key=key)[:25]))
else:
    print("layer " + " ".join(f"{c:>6}" for c in cols))
    for u, (l, _) in enumerate(units):
        print(f"{l:5d} " + " ".join("     ." if np.isnan(v) else f"{v:6.2f}" for v in rec[u]))
