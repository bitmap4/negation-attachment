"""E2: necessity of the head sets, a control with random heads, and the test of H3 (shortcut heads).

Part A, for each step of the route. The heads were ranked by e2_heads.py on the disc split; the items
here come from another split. Sets of heads: the top k, k random heads of the same layers (several
draws), all heads of the layers. For each set, at the destination tokens of the step:
  patch     the head outputs come from the minimal partner P. rec = recovery (sufficiency).
  resample  the head outputs come from a random item with the same family, question polarity and
            option order, other names and another object. Its negated mention is random.
            kept = logit difference toward the correct letter / the same with no ablation (necessity).
Part B (H3): mean ablation of one head at the last position, layers --h3. The change of the logit
difference toward the correct letter is given for affirmative (q=0) and negative (q=1) questions.
H3 predicts heads whose removal helps on negative questions. The heads are ranked on one half of the
items; the top sets are removed together on the other half.

Usage: python -m neg.e2_ablate <out.json> --lang hi [--split test --n 64 --k 8 --draws 5 --h3 20-35]
"""
import argparse, json, os, random, time
import numpy as np
import torch
from . import common as C
from .pairs import build, sgn

DEST = {"opt": ["lA", "oA", "lB", "oB"], "lbt": ["lB", "oB", "tail"], "end": ["end"]}

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--lang", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--draws", type=int, default=5)
ap.add_argument("--steps", default="opt,lbt,end")
ap.add_argument("--heads", default="results/e2_4b")
ap.add_argument("--h3", default="20-35")
ap.add_argument("--bs", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

tok = C.get_tok()
net = C.Net(C.get_model())
A, B = C.letter_ids(tok)
H, dh = net.H, net.dh
trip, sR, sP = build(tok, net, a.lang, a.split, "neg", a.n)
n = len(trip)
Rit, Pit = [t[0] for t in trip], [t[1] for t in trip]
pool = {}
for it in C.load(f"{a.lang}_{a.split}"):
    if "neg" in it and it["fam"] in (1, 2):
        pool.setdefault((it["fam"], it["q"], it["ord"]), []).append(it)
rng = random.Random(a.seed)


def resample(it):
    v = pool[(it["fam"], it["q"], it["ord"])]
    while True:
        d = v[rng.randrange(len(v))]
        if not {d["a"], d["b"]} & {it["a"], it["b"]} and d["obj"] != it["obj"]: return d


Xit = [resample(it) for it in Rit]
encR, encP, encX = (C.encode(tok, v) for v in (Rit, Pit, Xit))
sign = np.array([sgn(it) for it in Rit])              # toward the correct letter of R
same = np.array([x["neg"] == r["neg"] for x, r in zip(Xit, Rit)])
clean = -sR                                            # sR is signed toward the letter of P
# print(n, clean.mean(), sP.mean(), same.mean())


def run_set(encD, hs, dest, layers):
    """Logit difference toward the correct letter of R; the heads hs come from the donor at the dest tokens."""
    out = []
    for b in C.batches(n, a.bs):
        idx = list(b)
        idsD, maskD, padD = C.pad(tok, [encD[i][0] for i in idx])
        _, cache = net.run(idsD, maskD, read=[("z", l) for l in layers])
        idsR, maskR, padR = C.pad(tok, [encR[i][0] for i in idx])
        write = {}
        for l in layers:
            hl = [h for (ll, h) in hs if ll == l]
            if not hl: continue
            bi, pr, pd, hh = [], [], [], []
            for x, i in enumerate(idx):
                p, q = C.align(encR[i][1], encD[i][1], DEST[dest])
                for u, v in zip(p, q):
                    for h in hl:
                        bi.append(x); pr.append(u + padR[x]); pd.append(v + padD[x]); hh.append(h)
            t = [torch.tensor(v, device="cuda") for v in (bi, pr, pd, hh)]

            def fn(x, t=t, l=l, cache=cache):
                x4 = x.clone().view(x.shape[0], x.shape[1], H, dh)
                s4 = cache[("z", l)].view(x.shape[0], -1, H, dh)
                x4[t[0], t[1], t[3]] = s4[t[0], t[2], t[3]].to(x.dtype)
                return x4.view(x.shape)
            write[("z", l)] = fn
        lg, _ = net.run(idsR, maskR, write=write)
        out += (lg[:, A] - lg[:, B]).tolist()
    return np.array(out) * sign


res = {"args": vars(a), "model": C.MODEL, "n": n, "clean": float(clean.mean()), "same_frac": float(same.mean()), "steps": {}}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
t0 = time.time()
for step in a.steps.split(","):
    d = json.load(open(f"{a.heads}/heads_{step}_{a.lang}.json"))
    lo, hi = (int(x) for x in d["args"]["layers"].split("-"))
    layers, dest = list(range(lo, hi + 1)), d["args"]["dest"]
    order = [(l, h) for l, h, _ in d["rank"]]
    sets = [("top", order[:a.k])] + [(f"rand{j}", rng.sample(order[a.k:], a.k)) for j in range(a.draws)] + [("all", order)]
    rows = []
    for name, hs in sets:
        p, x = run_set(encP, hs, dest, layers), run_set(encX, hs, dest, layers)
        row = {"set": name, "heads": hs, "rec": float((-p.mean() - sR.mean()) / (sP.mean() - sR.mean())),
               "kept": float(x.mean() / clean.mean()), "acc": float((x > 0).mean()),
               "kept_diff": float(x[~same].mean() / clean[~same].mean()), "acc_diff": float((x[~same] > 0).mean()),
               "kept_same": float(x[same].mean() / clean[same].mean()), "acc_same": float((x[same] > 0).mean())}
        rows.append(row)
        print(f"{step} {name:6s} rec={row['rec']:.3f} kept={row['kept']:.3f} acc={row['acc']:.2f} "
              f"kept_diff={row['kept_diff']:.3f} acc_diff={row['acc_diff']:.2f} kept_same={row['kept_same']:.3f}", flush=True)
    res["steps"][step] = {"layers": layers, "dest": dest, "rows": rows}
    json.dump(res, open(a.out, "w"), indent=1)
print(f"part A {time.time() - t0:.0f} s", flush=True)

# ---------- part B: H3 ----------
lo, hi = (int(x) for x in a.h3.split("-"))
L3 = list(range(lo, min(hi, net.L - 1) + 1))
its, enc = Rit + Pit, encR + encP
m = len(its)
sg = np.array([sgn(it) for it in its])
qv = np.array([it["q"] for it in its])
acc, base = {l: 0 for l in L3}, []
for b in C.batches(m, a.bs):
    ids, mask, _ = C.pad(tok, [enc[i][0] for i in b])
    lg, st = net.run(ids, mask, read=[("z", l) for l in L3])
    base += (lg[:, A] - lg[:, B]).tolist()
    for l in L3: acc[l] = acc[l] + st[("z", l)][:, -1].float().sum(0)
mean = {l: (acc[l] / m).view(H, dh) for l in L3}
base = np.array(base) * sg


def abl(idx, sets):
    """Logit difference toward the correct letter; sets[x]: heads with the mean output at the last position, row x."""
    ids, mask, _ = C.pad(tok, [enc[i][0] for i in idx])
    write = {}
    for l in L3:
        bi, hh = [], []
        for x in range(len(idx)):
            for ll, h in sets[x]:
                if ll == l: bi.append(x); hh.append(h)
        if not bi: continue
        t = (torch.tensor(bi, device="cuda"), torch.tensor(hh, device="cuda"))

        def fn(x, t=t, l=l):
            x4 = x.clone().view(x.shape[0], x.shape[1], H, dh)
            x4[t[0], x.shape[1] - 1, t[1]] = mean[l][t[1]].to(x.dtype)
            return x4.view(x.shape)
        write[("z", l)] = fn
    lg, _ = net.run(ids, mask, write=write)
    return (lg[:, A] - lg[:, B]).cpu().numpy() * sg[idx]


perm = list(range(m)); rng.shuffle(perm)
h1, h2 = perm[: m // 2], perm[m // 2:]
heads = [(l, h) for l in L3 for h in range(H)]
jobs = [(i, hd) for hd in heads for i in h1]
tot = {hd: {0: [], 1: []} for hd in heads}
t0 = time.time()
for b in C.batches(len(jobs), a.bs):
    jb = [jobs[x] for x in b]
    s = abl([j[0] for j in jb], [[j[1]] for j in jb])
    for j, v in zip(jb, s): tot[j[1]][int(qv[j[0]])].append(float(v - base[j[0]]))
eff = {hd: [float(np.mean(tot[hd][0])), float(np.mean(tot[hd][1]))] for hd in heads}
print(f"H3: {len(heads)} heads on {len(h1)} items in {time.time() - t0:.0f} s; base q0={base[qv == 0].mean():.2f} q1={base[qv == 1].mean():.2f}", flush=True)
h2 = np.array(h2)
curves = {}
for q in (0, 1):
    order = sorted(heads, key=lambda hd: -eff[hd][q])
    print(f"q={q} top:", " ".join(f"L{l}H{h}:{eff[(l, h)][q]:+.2f}/{eff[(l, h)][1 - q]:+.2f}" for l, h in order[:8]), flush=True)
    rows = []
    for k in (1, 2, 4, 8, 16):
        s = np.concatenate([abl(list(h2[list(b)]), [order[:k]] * len(b)) for b in C.batches(len(h2), a.bs)])
        dl = s - base[h2]
        row = {"k": k, "heads": order[:k]}
        for qq in (0, 1):
            mk = qv[h2] == qq
            row[f"d_q{qq}"] = float(dl[mk].mean()); row[f"acc_q{qq}"] = float((s[mk] > 0).mean())
        rows.append(row)
        print(f"  k={k:2d} d_q0={row['d_q0']:+.2f} d_q1={row['d_q1']:+.2f} acc_q0={row['acc_q0']:.2f} acc_q1={row['acc_q1']:.2f}", flush=True)
    curves[q] = rows
res["h3"] = {"layers": L3, "n_rank": len(h1), "n_eval": len(h2), "base_q0": float(base[qv == 0].mean()), "base_q1": float(base[qv == 1].mean()),
             "eff": [[l, h, eff[(l, h)][0], eff[(l, h)][1]] for l, h in heads], "curves": curves}
json.dump(res, open(a.out, "w"), indent=1)
print("done", flush=True)
