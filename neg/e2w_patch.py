"""E2 on the explicit worlds (sets w1, w2, nest): residual patching with role alignment.

Recipient R: a world and a yes/no question about one fact, the target fact.
T: R with the polarity of the target fact changed (the answer changes).
Donor D, by mode:
  target  D = T
  sib     D = R with the other fact about the same person changed (w2 only; the answer does not change)
  cross   D = R with the fact about the other person and the same object changed (the answer does not change)
Score s: logit difference toward the letter that is correct for T. effect = (s_patched - s_R) / (s_T - s_R).
For mode target this is the recovery. For the other modes it must stay near 0: a larger value is a
collateral effect of a fact that the question does not ask about.
Roles are relative to the question: T target fact, S same person other object, X other person same
object, Y other person other object; n name, b body, c cue, e end of the sentence, x outer negation.

Usage: python -m neg.e2w_patch <out.npz> --lang hi --set w2 --mode target [--n 64]
"""
import argparse, json, os, random, sys, time
import numpy as np
import torch
from . import common as C

sys.path.insert(0, C.DATA)
import gen2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--lang", required=True)
ap.add_argument("--set", default="w1")
ap.add_argument("--split", default="disc")
ap.add_argument("--mode", default="target")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--pb", type=int, default=4)
ap.add_argument("--bs", type=int, default=48)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

tok = C.get_tok()
net = C.Net(C.get_model())
A, B = C.letter_ids(tok)
NI = {n[0]: i for i, n in enumerate(gen2.N)}
items = C.load(f"{a.lang}_{a.set}_{a.split}")
npred = 2 if a.set == "w2" else 1
REL = {(0, 0): "T", (0, 1): "S", (1, 0): "X", (1, 1): "Y"}


def variant(it, fact):
    """The item with the polarity of one fact changed. fact: index in mention order."""
    pol = list(it["pol"]); pol[fact] = 1 - pol[fact]
    new = gen2.make_world(a.lang, a.set, int(it["fam"][-1]), NI[it["a"]], NI[it["b"]], it["objs"], pol,
                          it["qe"], it["qp"], it["ord"], it["outer"])
    new["id"] = f"{it['id']}~{fact}"
    return new


def fact_of(it, rel):
    """Index of the fact with relation rel to the target (T, S, X)."""
    e, p = it["qe"], it["qp"]
    if rel == "S": p = 1 - p
    if rel == "X": e = 1 - e
    return e * npred + p


def rel_roles(it, roles):
    """Rename f<i>* roles relative to the target fact."""
    out = []
    for r in roles:
        if r[0] == "f" and r[1].isdigit():
            i = int(r[1]) - 1
            out.append(REL[(int(i // npred != it["qe"]), int(i % npred != it["qp"]))] + r[2:])
        else:
            out.append(r)
    return out


def enc(its):
    return [(ids, rel_roles(it, rl)) for it, (ids, rl) in zip(its, C.encode(tok, its))]


def sgn(it): return 1.0 if it["ans"] == "A" else -1.0


def base_ld(its):
    e = C.encode(tok, its)
    out = []
    for b in C.batches(len(its), 32):
        ids, mask, _ = C.pad(tok, [e[i][0] for i in b])
        lg, _ = net.run(ids, mask)
        out += (lg[:, A] - lg[:, B]).tolist()
    return out


R = random.Random(a.seed)
R.shuffle(items)
cand = items[: 2 * a.n]
trip = [(r, variant(r, fact_of(r, "T")), variant(r, fact_of(r, {"target": "T", "sib": "S", "cross": "X"}[a.mode]))) for r in cand]
ldR, ldT, ldD = (base_ld([t[i] for t in trip]) for i in range(3))
keep = [i for i, (r, t, dn) in enumerate(trip) if ldR[i] * sgn(r) > 0 and ldT[i] * sgn(t) > 0 and ldD[i] * sgn(dn) > 0]
print(f"correct on R, T and D: {len(keep)}/{len(trip)}", flush=True)
keep = keep[: a.n]
trip = [trip[i] for i in keep]
sR = np.array([ldR[i] * sgn(trip[j][1]) for j, i in enumerate(keep)])
sT = np.array([ldT[i] * sgn(trip[j][1]) for j, i in enumerate(keep)])
n = len(trip)

rels = ["T", "X"] if npred == 1 else ["T", "S", "X", "Y"]
single = [f"{r}{x}" for r in rels for x in ("x", "n", "b", "c", "e")] + ["q", "qn", "qo", "lA", "oA", "lB", "oB", "tail", "end"]
wide = {f"{r}*": [f"{r}{x}" for x in ("x", "n", "b", "c", "e")] for r in rels}
wide["ctx"] = sum(wide.values(), [])
wide["qst"] = ["q", "qn", "qo"]
wide["opt"] = ["lA", "oA", "lB", "oB"]
wide["post"] = wide["qst"] + wide["opt"] + ["tail", "end"]
cols = single + list(wide)
groups = [[r] for r in single] + list(wide.values())
layers = list(range(0, net.L + 1, a.stride))
res = np.full((n, len(layers), len(cols)), np.nan, dtype=np.float32)
encR, encD = enc([t[0] for t in trip]), enc([t[2] for t in trip])
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
    if ci % 4 == 0:
        print(f"chunk {ci} items {min(n, (ci + 1) * a.pb)}/{n} {time.time() - t0:.0f} s", flush=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
meta = dict(vars(a), kind="resid", model=C.MODEL, cols=cols, units=[(l, None) for l in layers], ids=[t[0]["id"] for t in trip])
np.savez_compressed(a.out, s=res, sR=sR, sP=sT, meta=json.dumps(meta))
eff = (np.nanmean(res, 0) - sR.mean()) / (sT.mean() - sR.mean())
print(f"n={n} sR={sR.mean():.2f} sT={sT.mean():.2f}  time {time.time() - t0:.0f} s")
show = [i for i in range(len(cols)) if not np.all(np.isnan(eff[:, i])) and np.nanmax(np.abs(eff[:, i])) >= 0.03]
print("layer " + " ".join(f"{cols[i]:>5}" for i in show))
for u, l in enumerate(layers):
    print(f"{l:5d} " + " ".join("    ." if np.isnan(eff[u, i]) else f"{eff[u, i]:5.2f}" for i in show))
