"""E3 on the explicit worlds: is the polarity stored for each fact, in one code for all facts?

A world has 2 or 4 facts (person, object, polarity) and one yes/no question about one fact.
Interchange: the tokens of fact i of the base item (after the name: body and sentence end) get a
subspace from the tokens of fact j of the source item. One basis Q is used for all facts.
High-level models (they give the label):
  P      the subspace holds the polarity of the fact. Fact i takes the polarity of source fact j.
  INNER  the subspace holds only the polarity of the inner clause (negation cue or not), before an
         outer "it is not true that" is applied. The two models differ only on nested facts.
Scores on held-out items:
  target      the question is about fact i, and the label is not the base answer: accuracy of the label
  same        the question is about fact i, and the label is the base answer
  collateral  the question is about a different fact: the answer must not change
Each is given for slot = (i equal to j) and cross (i different from j).

Usage: python -m neg.e3w_das <out.json> --train hi_w1_disc,hi_nest_disc --eval hi_w1_test hi_nest_test fr_w1_test \
           --layers 9,11,13,15,17 --vars P,INNER --k 8
"""
import argparse, json, os, random, time
import torch
from . import common as C

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--train", required=True)
ap.add_argument("--eval", nargs="+", default=[])
ap.add_argument("--layers", required=True)
ap.add_argument("--vars", default="P")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--neval", type=int, default=150, help="pairs for each group")
ap.add_argument("--scale", type=float, default=64.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--save", default="")
ap.add_argument("--site", default="be", help="be or fact (see fact_pos)")
a = ap.parse_args()

torch.manual_seed(a.seed)
tok = C.get_tok()
net = C.Net(C.get_model())
A, B = C.letter_ids(tok)
d = net.m.config.hidden_size
_data = {}


def data(names):
    """names: data sets joined with commas. 'X<Y' is handled by the caller."""
    if names not in _data:
        its = [it for n in names.split(",") for it in C.load(n)]
        _data[names] = (its, C.encode(tok, its))
    return _data[names]


def nfact(it): return len(it["pol"])
def qfact(it): return it["qe"] * (nfact(it) // 2) + it["qp"]
def outer(it, i): return int(bool(it.get("outer")) and i == qfact(it))


def new_truth(b, s, i, j, var):
    """Truth value of the base question after fact i takes the variable of source fact j."""
    if qfact(b) != i: return b["truth"]
    if var == "P": return s["pol"][j]
    inner = s["pol"][j] ^ outer(s, j)
    return inner ^ outer(b, i)


def letter(it, truth): return "A" if (truth == 1) == (it["ord"] == 0) else "B"


def sample(R, base, src, n, var, group=None, cross=None, differ=None):
    """(base index, source index, i, j, label). group: target, same, collateral. cross: i != j.
    differ: only pairs where the models P and INNER give different labels."""
    (bi, _), (si, _) = base, src
    out, tries = [], 0
    while len(out) < n and tries < 200000:
        tries += 1
        x, y = R.randrange(len(bi)), R.randrange(len(si))
        b, s = bi[x], si[y]
        if {b["a"], b["b"]} & {s["a"], s["b"]}: continue
        i, j = R.randrange(nfact(b)), R.randrange(nfact(s))
        if group in ("target", "same"): i = qfact(b)
        t = new_truth(b, s, i, j, var)
        if group == "target" and t == b["truth"]: continue
        if group == "same" and t != b["truth"]: continue
        if group == "collateral" and qfact(b) == i: continue
        if cross is not None and (i != j) != cross: continue
        if differ is not None and (new_truth(b, s, i, j, "P") != new_truth(b, s, i, j, "INNER")) != differ: continue
        out.append((x, y, i, j, letter(b, t)))
    return out


def fact_pos(rb, rs, i, j):
    """Site 'be': body and sentence end, aligned role by role (the cue token is not in the site).
    Site 'fact': all tokens of the fact after the name, aligned from the end; a cue token of one item
    is then aligned with a different token of the other item."""
    pb, ps = [], []
    for parts in ((("b",), ("e",)) if a.site == "be" else (("c", "b", "e"),)):
        u = [p for p, r in enumerate(rb) if r in [f"f{i + 1}{x}" for x in parts]]
        v = [p for p, r in enumerate(rs) if r in [f"f{j + 1}{x}" for x in parts]]
        k = min(len(u), len(v))
        pb += u[len(u) - k:]; ps += v[len(v) - k:]
    return pb, ps


def run(Q, site, base, src, pairs, grad=False):
    (bi, be), (si, se) = base, src
    ids, mask, padb = C.pad(tok, [be[p[0]][0] for p in pairs])
    sids, smask, pads = C.pad(tok, [se[p[1]][0] for p in pairs])
    rb, pb, ps = [], [], []
    for x, (bx, sy, i, j, _) in enumerate(pairs):
        u, v = fact_pos(be[bx][1], se[sy][1], i, j)
        rb += [x] * len(u); pb += [p + padb[x] for p in u]; ps += [p + pads[x] for p in v]
    rb, pb, ps = (torch.tensor(v, device="cuda") for v in (rb, pb, ps))
    _, st = net.run(sids, smask, read=[site])
    hs = st[site][rb, ps].float()

    def fn(x):
        hb = x[rb, pb].float()
        new = hs if isinstance(Q, str) else hb + ((hs - hb) @ Q) @ Q.T
        x = x.clone()
        x[rb, pb] = new.to(x.dtype)
        return x
    lg, _ = net.run(ids, mask, write={site: fn}, grad=grad)
    return lg[:, A] - lg[:, B]


def score(Q, site, base, src, pairs):
    if not pairs: return None
    ok = 0
    for b in C.batches(len(pairs), 16):
        pp = [pairs[i] for i in b]
        with torch.no_grad():
            dl = run(Q, site, base, src, pp)
        y = torch.tensor([1.0 if p[4] == "A" else -1.0 for p in pp], device="cuda")
        ok += ((dl * y) > 0).sum().item()
    return ok / len(pairs)


def train(site, var):
    lin = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(a.k, d, bias=False).cuda())
    tr = data(a.train)
    R = random.Random(a.seed)
    opt = torch.optim.Adam(lin.parameters(), lr=a.lr)
    t0, log = time.time(), []
    for step in range(a.steps):
        pairs = sample(R, tr, tr, a.bs, var)
        dl = run(lin.weight, site, tr, tr, pairs, grad=True)
        y = torch.tensor([1.0 if p[4] == "A" else 0.0 for p in pairs], device="cuda")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(dl, y)
        opt.zero_grad()
        (loss * a.scale).backward()
        g = lin.parametrizations.weight.original.grad
        if g is None or not torch.isfinite(g).all():
            print(f"step {step}: gradient is not finite, step skipped", flush=True); continue
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            log.append([step, round(loss.item(), 4), ((dl > 0) == (y > 0.5)).float().mean().item()])
    print(f"  trained L{site[1]} {var}: {log[-1]} {time.time() - t0:.0f} s", flush=True)
    return lin.weight.detach().clone(), log


def rand_Q(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, a.k, generator=g))[0].cuda()


res = {"args": vars(a), "model": C.MODEL, "runs": []}
for layer in (int(x) for x in a.layers.split(",")):
    site = ("resid", layer)
    for var in a.vars.split(","):
        Q, log = train(site, var)
        if a.save:
            os.makedirs(a.save, exist_ok=True)
            torch.save(Q.cpu(), os.path.join(a.save, f"{a.site}_L{layer}_{var}_k{a.k}.pt"))
        r = {"layer": layer, "var": var, "k": a.k, "log": log, "eval": {}}
        for name in a.eval:
            bn, sn = name.split("<") if "<" in name else (name, name)
            base, src = data(bn), data(sn)
            R = random.Random(1000 + a.seed)
            e = {}
            for cross in (False, True):
                tag = "cross" if cross else "slot"
                tg = sample(R, base, src, a.neval, var, "target", cross)
                e[f"target_{tag}"] = {"das": score(Q, site, base, src, tg), "full": score("full", site, base, src, tg),
                                      "rand": score(rand_Q(7), site, base, src, tg[: a.neval // 2])}
                e[f"same_{tag}"] = {"das": score(Q, site, base, src, sample(R, base, src, a.neval // 2, var, "same", cross))}
                co = sample(R, base, src, a.neval, var, "collateral", cross)
                e[f"collateral_{tag}"] = {"das": score(Q, site, base, src, co), "full": score("full", site, base, src, co)}
            df = sample(R, base, src, a.neval, var, "target", None, differ=True)     # the two models give different labels
            if df: e["target_models_differ"] = {"das": score(Q, site, base, src, df), "n": len(df)}
            r["eval"][name] = e
            print(f"L{layer} {var} {name}: " + " ".join(f"{k}={v['das']:.2f}" + (f"(full {v['full']:.2f})" if v.get("full") is not None else "") for k, v in e.items() if v["das"] is not None), flush=True)
        res["runs"].append(r)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
