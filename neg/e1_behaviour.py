"""E1: accuracy and logit difference for each language, family and condition.

Usage: python -m neg.e1_behaviour <out.json> <dataset> [<dataset> ...] [--n N] [--bs B]
A dataset is a file name in data/ without the extension, for example hi_dev.
"""
import argparse, json, os, random, time
from collections import defaultdict
import torch
from . import common as C

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("sets", nargs="+")
ap.add_argument("--n", type=int, default=0, help="use only the first N pairs of each family")
ap.add_argument("--bs", type=int, default=48)
ap.add_argument("--keys", default="lang,fam,q,neg,ord", help="condition fields for the summary")
a = ap.parse_args()

tok = C.get_tok()
model = C.get_model()
A, B = C.letter_ids(tok)
rows = []
for name in a.sets:
    items = C.load(name)
    if a.n:
        fams = defaultdict(set)
        for it in items:
            fams[it["fam"]].add(it["pair"])
        R = random.Random(0)
        keep = set()
        for f in sorted(fams, key=str):
            p = sorted(fams[f])
            keep |= set(R.sample(p, min(a.n, len(p))))
        items = [it for it in items if it["pair"] in keep]
    enc = C.encode(tok, items)
    order = sorted(range(len(items)), key=lambda i: len(enc[i][0]))
    t0 = time.time()
    for b in C.batches(len(order), a.bs):
        idx = [order[i] for i in b]
        ids, mask, _ = C.pad(tok, [enc[i][0] for i in idx])
        with torch.no_grad():
            lg = C.last_logits(model, ids, mask)
        pr = lg.softmax(-1)
        top = lg.argmax(-1)
        for j, i in enumerate(idx):
            it = items[i]
            ld = (lg[j, A] - lg[j, B]).item()
            r = {k: it[k] for k in it if k not in ("msg", "sp", "roles")}
            r.update(set=name, ld=ld, pAB=(pr[j, A] + pr[j, B]).item(), top=tok.decode(top[j]),
                     ok=int((ld > 0) == (it["ans"] == "A")))
            rows.append(r)
    print(f"{name}: {len(items)} items, {time.time() - t0:.0f} s, acc {sum(r['ok'] for r in rows[-len(items):]) / len(items):.3f}", flush=True)

keys = a.keys.split(",")


def summ(rs):
    pairs = defaultdict(list)
    for r in rs:
        pairs[(r["set"], r["pair"])].append(r["ok"])
    both = [min(v) for v in pairs.values() if len(v) == 2]
    sign = [r["ld"] if r["ans"] == "A" else -r["ld"] for r in rs]
    return {"n": len(rs), "acc": sum(r["ok"] for r in rs) / len(rs),
            "pair_acc": sum(both) / len(both) if both else None,
            "ld": sum(sign) / len(sign), "pAB": sum(r["pAB"] for r in rs) / len(rs),
            "sayA": sum(r["ld"] > 0 for r in rs) / len(rs)}


def table(fields):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r.get(f) for f in fields)].append(r)
    return [dict(zip(fields, k), **summ(v)) for k, v in sorted(g.items(), key=lambda x: str(x[0]))]


res = {"model": C.MODEL, "shots": C.SHOTS, "by_set_fam": table(["set", "fam"]), "by_cond": table(["set"] + keys)}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
json.dump(rows, open(a.out.replace(".json", "_items.json"), "w"))
print("set fam n acc pair_acc ld pAB sayA")
for r in res["by_set_fam"]:
    print(r["set"], r["fam"], r["n"], f"{r['acc']:.3f}", "-" if r["pair_acc"] is None else f"{r['pair_acc']:.3f}",
          f"{r['ld']:.2f} {r['pAB']:.3f} {r['sayA']:.2f}")
worst = sorted(res["by_cond"], key=lambda r: r["acc"])[:12]
print("lowest conditions:")
for r in worst:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
