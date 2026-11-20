"""E3: distributed alignment search (DAS) on the base task.

One orthonormal basis Q (d x k) is learned at a site: a layer and some token positions (see SITES).
The same Q is used at each position of the site.
Interchange intervention: h_base <- h_base + (h_source - h_base) Q Q^T.
The label is the answer of the base item when a high-level variable takes the value of the source item.
Variables (each one, when it changes, changes the answer):
  W    the negated mention (first or second)
  Z    option A holds the negated person (W combined with the option order)
  Q    polarity of the question
  ANS  the answer letter
A site that holds W transfers between items with different option orders; a site that holds Z or ANS does not.

Usage:
  python -m neg.e3_das <out.json> --train hi_disc --eval hi_test fr_test "hi_test<fr_test" \
      --grid "oA:16,18,20;end:26,28" --vars W,Z,ANS --k 8
An eval name "X<Y" takes the base items from X and the source items from Y.
A train name "X+Y" joins two sets, for example hi_disc+fr_disc (one subspace for both languages).
--load <dir> uses saved bases (no training); --save <dir> writes them.
"""
import argparse, json, os, random, time
import torch
from . import common as C

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--train", default="")
ap.add_argument("--eval", nargs="+", default=[])
ap.add_argument("--grid", required=True, help="role:layer,layer;role:layer")
ap.add_argument("--vars", default="W")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--neval", type=int, default=300)
ap.add_argument("--scale", type=float, default=1.0, help="loss scale; use 64 with float16")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--load", default="")
ap.add_argument("--save", default="")
a = ap.parse_args()

torch.manual_seed(a.seed)
tok = C.get_tok()
net = C.Net(C.get_model())
A, B = C.letter_ids(tok)
d = net.m.config.hidden_size
_data = {}


def data(name):
    """Items with a negated mention (not the affirmative controls), and their encodings."""
    if name not in _data:
        its = []
        for nm in name.split("+"):          # "hi_disc+fr_disc": the items of both sets
            its += [it for it in C.load(nm) if "neg" in it and it.get("fam") != 0]
        _data[name] = (its, C.encode(tok, its))
    return _data[name]


def controls(name):
    key = name + "/ctrl"
    if key not in _data:
        its = [it for it in C.load(name) if it.get("fam") == 0]
        _data[key] = (its, C.encode(tok, its))
    return _data[key]


# site name -> (roles, all tokens of the roles; else the last token of each role)
SITES = {"c12": (["body1", "e1", "body2", "e2"], True), "c12l": (["body1", "e1", "body2", "e2"], False),
         "opt": (["oA", "oB"], False)}


def positions(rb, rs, role):
    """Aligned token positions of the site in the base and in the source."""
    roles, allt = SITES.get(role, ([role], False))
    if allt: return C.align(rb, rs, roles)
    return [last(rb, r) for r in roles], [last(rs, r) for r in roles]


def has(roles_of_item, role):
    return all(r in roles_of_item for r in SITES.get(role, ([role], False))[0])


def val(it, var):
    if var == "W": return it["neg"]
    if var == "Z": return int(it["neg"] == it["ord"])
    if var == "Q": return it["q"]
    return int(it["ans"] == "A")


def flip(x): return "B" if x == "A" else "A"


def sample(R, base, src, n, var, role, same=None, ctrl=False):
    """n triples (base index, source index, label). same: None random, True equal value, False different."""
    (bi, be), (si, se) = base, src
    out, tries = [], 0
    while len(out) < n and tries < 100000:        # some sets do not have the roles of the site
        tries += 1
        i, j = R.randrange(len(bi)), R.randrange(len(si))
        b, s = bi[i], si[j]
        if {b["a"], b["b"]} & {s["a"], s["b"]} or not has(be[i][1], role) or not has(se[j][1], role): continue
        if ctrl:
            out.append((i, j, b["ans"])); continue
        eq = val(b, var) == val(s, var)
        if same is not None and eq != same: continue
        out.append((i, j, b["ans"] if eq else flip(b["ans"])))
    return out


def last(roles, r):
    return max(i for i, x in enumerate(roles) if x == r)


def run(Q, site, role, base, src, pairs, grad=False):
    """Logit difference A - B of the base items after the interchange.
    Q None: no intervention; Q 'full': the whole vector is copied."""
    (bi, be), (si, se) = base, src
    ids, mask, padb = C.pad(tok, [be[i][0] for i, _, _ in pairs])
    if Q is None:
        lg, _ = net.run(ids, mask)
        return lg[:, A] - lg[:, B]
    sids, smask, pads = C.pad(tok, [se[j][0] for _, j, _ in pairs])
    rb, pb, ps = [], [], []
    for x, (i, j, _) in enumerate(pairs):
        p, q = positions(be[i][1], se[j][1], role)
        rb += [x] * len(p); pb += [u + padb[x] for u in p]; ps += [v + pads[x] for v in q]
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


def score(Q, site, role, base, src, pairs):
    ok, ld = 0, 0.0
    for b in C.batches(len(pairs), 24):
        pp = [pairs[i] for i in b]
        with torch.no_grad():
            dl = run(Q, site, role, base, src, pp)
        y = torch.tensor([1.0 if p[2] == "A" else -1.0 for p in pp], device="cuda")
        ok += ((dl * y) > 0).sum().item(); ld += (dl * y).sum().item()
    return {"iia": ok / len(pairs), "ld": ld / len(pairs)}


def train(site, role, var):
    lin = torch.nn.Linear(a.k, d, bias=False).cuda()
    lin = torch.nn.utils.parametrizations.orthogonal(lin)
    tr = data(a.train)
    R = random.Random(a.seed)
    opt = torch.optim.Adam(lin.parameters(), lr=a.lr)
    log, t0 = [], time.time()
    for step in range(a.steps):
        pairs = sample(R, tr, tr, a.bs, var, role)
        dl = run(lin.weight, site, role, tr, tr, pairs, grad=True)
        y = torch.tensor([1.0 if p[2] == "A" else 0.0 for p in pairs], device="cuda")
        loss = torch.nn.functional.binary_cross_entropy_with_logits(dl, y)
        opt.zero_grad()
        (loss * a.scale).backward()      # Adam ignores the scale; it keeps float16 gradients above underflow
        g = lin.parametrizations.weight.original.grad
        if g is None or not torch.isfinite(g).all():
            print(f"step {step}: gradient is not finite, step skipped", flush=True); continue
        opt.step()
        if step % 50 == 0 or step == a.steps - 1:
            acc = ((dl > 0) == (y > 0.5)).float().mean().item()
            log.append([step, round(loss.item(), 4), acc])
    print(f"  trained {role} L{site[1]} {var}: {log[-1]} {time.time() - t0:.0f} s", flush=True)
    return lin.weight.detach().clone(), log


def rand_Q(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(d, a.k, generator=g))[0].cuda()


grid = [(r.split(":")[0], int(l)) for r in a.grid.split(";") for l in r.split(":")[1].split(",")]
res = {"args": vars(a), "model": C.MODEL, "runs": []}
for role, layer in grid:
    site = ("resid", layer)
    for var in a.vars.split(","):
        fn = f"{role}_L{layer}_{var}_k{a.k}.pt"
        if a.load:
            Q, log = torch.load(os.path.join(a.load, fn), map_location="cuda").float(), []
        else:
            Q, log = train(site, role, var)
            if a.save:
                os.makedirs(a.save, exist_ok=True)
                torch.save(Q.cpu(), os.path.join(a.save, fn))
        r = {"role": role, "layer": layer, "var": var, "k": a.k, "log": log, "eval": {}}
        for name in a.eval:
            bn, sn = name.split("<") if "<" in name else (name, name)
            base, src = data(bn), data(sn)
            R = random.Random(1000 + a.seed)
            fl = sample(R, base, src, a.neval, var, role, same=False)
            if len(fl) < a.neval:
                print(f"{role} L{layer} {var} {name}: the items do not have this site", flush=True); continue
            sm = sample(R, base, src, a.neval // 2, var, role, same=True)
            e = {"none": score(None, site, role, base, src, fl), "full": score("full", site, role, base, src, fl),
                 "das": score(Q, site, role, base, src, fl), "das_same": score(Q, site, role, base, src, sm),
                 "rand": score(rand_Q(7), site, role, base, src, fl[: a.neval // 2])}
            cb = controls(bn)
            if cb[0] and all(has(x[1], role) for x in cb[1][:5]):
                cp = sample(R, cb, src, a.neval // 2, var, role, ctrl=True)
                e["ctrl_none"] = score(None, site, role, cb, src, cp)
                e["ctrl_das"] = score(Q, site, role, cb, src, cp)     # iia: the answer of the affirmative item does not change
            r["eval"][name] = e
            print(f"{role} L{layer} {var} {name}: " + " ".join(f"{k}={v['iia']:.2f}" for k, v in e.items()), flush=True)
        res["runs"].append(r)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
