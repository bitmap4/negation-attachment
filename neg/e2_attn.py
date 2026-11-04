"""E2: attention patterns. For each head, the attention from some destination tokens to each role.

Destinations: the last token of oA, oB, lB, q, and the last position (end).
The mean is kept for each condition (negated mention x option order), so that we can see if a head
at an option name looks at the clause of the person that the option names.

Usage: python -m neg.e2_attn <out.npz> --lang hi [--n 96] [--layers 8-30]
"""
import argparse, json, os
import numpy as np
import torch
from . import common as C
from .pairs import build

SRC = ["pre", "ins", "n1", "body1", "cue1", "e1", "conj", "n2", "body2", "cue2", "e2", "q", "qcue", "lA", "oA", "lB", "oB", "tail", "end"]
DST = ["oA", "oB", "lB", "q", "end"]

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--lang", required=True)
ap.add_argument("--n", type=int, default=96)
ap.add_argument("--layers", default="8-30")
ap.add_argument("--bs", type=int, default=6)
a = ap.parse_args()

tok = C.get_tok()
model = C.get_model(eager=True)
net = C.Net(model)
trip, _, _ = build(tok, net, a.lang, "disc", "neg", a.n)
items = [t[0] for t in trip]
enc = C.encode(tok, items)
lo, hi = (int(x) for x in a.layers.split("-"))
layers = list(range(lo, hi + 1))
acc = np.zeros((4, len(DST), len(layers), net.H, len(SRC)), dtype=np.float64)
cnt = np.zeros(4)
for b in C.batches(len(items), a.bs):
    ids, mask, padn = C.pad(tok, [enc[i][0] for i in b])
    with torch.no_grad():
        att = model.model(input_ids=ids, attention_mask=mask, output_attentions=True).attentions
    for x, i in enumerate(b):
        it = items[i]
        roles = C.tag_cue(enc[i][1])
        rid = torch.tensor([SRC.index(r) for r in roles], device="cuda")
        c = it["neg"] * 2 + it["ord"]
        cnt[c] += 1
        for di, dname in enumerate(DST):
            pos = max(p for p, r in enumerate(enc[i][1]) if r == dname) + padn[x]
            for li, l in enumerate(layers):
                w = att[l][x, :, pos, padn[x]:].float()                   # heads x tokens
                m = torch.zeros(net.H, len(SRC), device="cuda").index_add_(1, rid, w)
                acc[c, di, li] += m.cpu().numpy()
acc /= cnt[:, None, None, None, None]
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
np.savez_compressed(a.out, att=acc.astype(np.float32), meta=json.dumps(dict(vars(a), model=C.MODEL, src=SRC, dst=DST, layers=layers,
                    conds=["neg0 ord0", "neg0 ord1", "neg1 ord0", "neg1 ord1"])))
# option A names the first mention when ord = 0. Attention from oA to the clause of its person, and to the other clause.
c1 = [SRC.index(r) for r in ("n1", "body1", "cue1", "e1")]; c2 = [SRC.index(r) for r in ("n2", "body2", "cue2", "e2")]
own = (acc[0, 0][..., c1].sum(-1) + acc[2, 0][..., c1].sum(-1) + acc[1, 0][..., c2].sum(-1) + acc[3, 0][..., c2].sum(-1)) / 4
oth = (acc[0, 0][..., c2].sum(-1) + acc[2, 0][..., c2].sum(-1) + acc[1, 0][..., c1].sum(-1) + acc[3, 0][..., c1].sum(-1)) / 4
flat = sorted(((own[li, h] - oth[li, h], layers[li], h, own[li, h], oth[li, h]) for li in range(len(layers)) for h in range(net.H)), reverse=True)
print("heads at oA that look at the clause of the person of option A (own clause - other clause):")
print(" ".join(f"L{l}H{h}:{o:.2f}/{t:.2f}" for _, l, h, o, t in flat[:24]))
