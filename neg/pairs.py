"""Recipient / partner / donor triples of the base task, for the patching scripts.

R is the recipient, P its minimal partner (the negation is in the other clause), D the donor.
Modes (how D is selected for R):
  neg    D = P: only the negated mention changes
  other  D has other names and another object; its negated mention is that of P
  ord    D = P with the option order changed (D has the same correct letter as R)
  q      D = R with the other question polarity
  name   D = R with the two names exchanged in the context
"""
import random
import numpy as np
from . import common as C


def sgn(it):
    return 1.0 if it["ans"] == "A" else -1.0


def build(tok, net, lang, split, mode, n, seed=0):
    """split: disc, dev, test (base task) or <set>_<split> for a set of gen2.py with a moved negation.
    Returns the triples that the model answers correctly, and the scores sR, sP.
    The score is the logit difference toward the letter that is correct for P."""
    A, B = C.letter_ids(tok)
    items = [it for it in C.load(f"{lang}_{split}") if "neg" in it and it["fam"] != 0]
    ext = "set" in items[0]            # a set of gen2.py: the partner has the same pair id; only mode neg
    if not ext:
        items = [it for it in items if it["fam"] in (1, 2)]
    by_id = {it["id"]: it for it in items}
    by_pair = {}
    for it in items:
        by_pair.setdefault(it["pair"], {})[it["neg"]] = it

    def pid(it, **ch):
        """Id of the item that differs from `it` in the given fields."""
        sp, f, x, y, o, q, od, ng = it["id"].split("-")
        d = dict(x=x, y=y, q=q, od=od, n=ng[1:])
        d.update({k: str(v) for k, v in ch.items()})
        return f"{sp}-{f}-{d['x']}-{d['y']}-{o}-{d['q']}-{d['od']}-n{d['n']}"

    def partner(it):
        return by_pair[it["pair"]][1 - it["neg"]]

    R = random.Random(seed)
    cells = {}
    for it in items:
        cells.setdefault((it["fam"], it["neg"], it["q"], it["ord"]), []).append(it)
    for v in cells.values():
        R.shuffle(v)

    def donor(it):
        p = partner(it)
        if mode == "neg": return p
        assert not ext, "the new sets have only mode neg"
        if mode == "ord": return by_id[pid(it, n=1 - it["neg"], od=1 - it["ord"])]
        if mode == "q": return by_id[pid(it, q=1 - it["q"])]
        if mode == "name":
            _, _, x, y, *_ = it["id"].split("-")
            return by_id[pid(it, x=y, y=x, od=1 - it["ord"])]
        if mode == "other":
            pool = cells[(it["fam"], p["neg"], it["q"], it["ord"])]
            while True:
                d = pool[R.randrange(len(pool))]
                if not {d["a"], d["b"]} & {it["a"], it["b"]} and d["obj"] != it["obj"]: return d
        raise ValueError(mode)

    cand, k = [], 0
    while len(cand) < 2 * n:            # balanced over the cells
        for key in sorted(cells, key=str):
            if k < len(cells[key]): cand.append(cells[key][k])
        k += 1
        if k > max(len(v) for v in cells.values()): break
    trip = [(r, partner(r), donor(r)) for r in cand]

    def base_ld(its):
        enc = C.encode(tok, its)
        out = []
        for b in C.batches(len(its), 32):
            ids, mask, _ = C.pad(tok, [enc[i][0] for i in b])
            lg, _ = net.run(ids, mask)
            out += (lg[:, A] - lg[:, B]).tolist()
        return out

    ldR, ldP, ldD = (base_ld([t[i] for t in trip]) for i in range(3))
    keep = [i for i, (r, p, d) in enumerate(trip) if ldR[i] * sgn(r) > 0 and ldP[i] * sgn(p) > 0 and ldD[i] * sgn(d) > 0]
    print(f"correct on R, P and D: {len(keep)}/{len(trip)}", flush=True)
    keep = keep[:n]
    trip = [trip[i] for i in keep]
    sR = np.array([ldR[i] * sgn(trip[j][1]) for j, i in enumerate(keep)])
    sP = np.array([ldP[i] * sgn(trip[j][1]) for j, i in enumerate(keep)])
    return trip, sR, sP
