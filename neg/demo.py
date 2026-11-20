"""Demo: one minimal pair, and patches from the partner into the recipient.

The recipient R and the partner P have the same words; the negation is in the other clause.
Each patch copies some activations of P into the run of R, at tokens with the same role.
A patch that holds the attachment changes the answer of R to the answer of P.

Usage: python -m neg.demo --lang hi [--i 0,1,2 --split test --heads results/e2_4b]
"""
import argparse, json, random
import torch
from . import common as C
from .pairs import build

ap = argparse.ArgumentParser()
ap.add_argument("--lang", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--i", default="0", help="indices of the pairs, for example 0,1,2")
ap.add_argument("--heads", default="results/e2_4b")
a = ap.parse_args()

tok = C.get_tok()
net = C.Net(C.get_model())
A, B = C.letter_ids(tok)
H, dh = net.H, net.dh
idx = [int(x) for x in a.i.split(",")]
trip, _, _ = build(tok, net, a.lang, a.split, "neg", max(16, max(idx) + 1))
d = json.load(open(f"{a.heads}/heads_opt_{a.lang}.json"))
order = [(l, h) for l, h, _ in d["rank"]]
top = order[:8]
rnd = random.Random(0).sample(order[8:], 8)
name = lambda hs: " ".join(f"L{l}H{h}" for l, h in sorted(hs))
let = lambda x: "A" if x > 0 else "B"
C1, C2, OPT = ["n1", "body1", "e1"], ["n2", "body2", "e2"], ["lA", "oA", "lB", "oB"]
print(f"model: {C.MODEL}")

for it_i in idx:
    r, p, _ = trip[it_i]
    (idr, rr), (idp, rp) = C.encode(tok, [r, p])
    ids_r, mask_r, _ = C.pad(tok, [idr])
    ids_p, mask_p, _ = C.pad(tok, [idp])
    print(f"\n==== pair {it_i}")

    def ld(write=None):
        lg, _ = net.run(ids_r, mask_r, write=write)
        return (lg[0, A] - lg[0, B]).item()


    def text(it):
        ln = it["msg"].split("\n")
        return f"{ln[2]}  {ln[3]}  {ln[4]}  {ln[5]}"


    lg, _ = net.run(ids_p, mask_p)
    sp = (lg[0, A] - lg[0, B]).item()
    sr = ld()
    print(f"R: {text(r)}\n   answer {let(sr)} (logit A - logit B = {sr:+.1f}), correct {r['ans']}")
    print(f"P: {text(p)}\n   answer {let(sp)} (logit A - logit B = {sp:+.1f}), correct {p['ans']}")
    print("\nPatch from P into R".ljust(58) + "A - B   answer  recovery")


    def row(name, v):
        print(f"  {name:54s} {v:+6.1f}   {let(v)}       {(v - sr) / (sp - sr):5.2f}")


    def resid(layer, roles):
        pr, pd = C.align(rr, rp, roles)
        _, st = net.run(ids_p, mask_p, read=[("resid", layer)])
        src = st[("resid", layer)]

        def fn(x):
            x = x.clone(); x[0, pr] = src[0, pd].to(x.dtype); return x
        return ld({("resid", layer): fn})


    def heads(hs, roles):
        pr, pd = C.align(rr, rp, roles)
        layers = sorted({l for l, _ in hs})
        _, st = net.run(ids_p, mask_p, read=[("z", l) for l in layers])
        write = {}
        for l in layers:
            hl = [h for ll, h in hs if ll == l]

            def fn(x, l=l, hl=hl):
                x4 = x.clone().view(1, x.shape[1], H, dh)
                s4 = st[("z", l)].view(1, -1, H, dh)
                for h in hl: x4[0, pr, h] = s4[0, pd, h].to(x.dtype)
                return x4.view(x.shape)
            write[("z", l)] = fn
        return ld(write)


    row("the negation cue, layer 13", resid(13, ["cue"]))
    row("clause 1, layer 13", resid(13, C1))
    row("clause 2, layer 13", resid(13, C2))
    row("both clauses, layer 13", resid(13, C1 + ["conj"] + C2))
    row("both clauses and the cue, layer 15", resid(15, C1 + ["conj", "cue"] + C2))
    row("both clauses, layer 19", resid(19, C1 + ["conj"] + C2))
    row("option tokens, layer 13", resid(13, OPT))
    row("option tokens, layer 19", resid(19, OPT))
    row("option tokens, layer 20", resid(20, OPT))
    row("last position, layer 19", resid(19, ["end"]))
    row("last position, layer 25", resid(25, ["end"]))
    row("8 heads at the option tokens (layers 16-19)", heads(top, OPT))
    print(f"      {name(top)}")
    row("8 random heads of the same layers", heads(rnd, OPT))
    print(f"      {name(rnd)}")
    row("the other 120 heads of these layers", heads(order[8:], OPT))
