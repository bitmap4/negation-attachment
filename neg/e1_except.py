"""E1, error analysis: the Hindi "nobody except X" items with other wordings of the exception.

The base items say "X को छोड़कर किसी को ... नहीं है". Each variant changes only the context sentence.
  base     X को छोड़कर          (the items of the quant set)
  alava    X के अलावा
  siva     X के सिवा
  only     केवल X को ... है      ("only X likes ...", no negation)
  split    two sentences: "किसी को ... नहीं है।" then the fact about X in the affirmative
  post     the exception phrase after the quantifier sentence: "किसी को ... नहीं है, X को छोड़कर।" (post_siva: के सिवा)
The same variants are applied to "everyone except X" (allx) where they exist.
With --lang fr: base ("..., sauf X.") and pre ("Sauf X, personne ..."), the exception before the quantifier.
Accuracy and the rate of "yes" are given for the question about the exception (exc) and about another person (rest).

Usage: python -m neg.e1_except <out.json> [--split test]
"""
import argparse, json, os
from collections import defaultdict
import torch
from . import common as C

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--split", default="test")
ap.add_argument("--lang", default="hi")
ap.add_argument("--bs", type=int, default=48)
a = ap.parse_args()

tok = C.get_tok()
model = C.get_model()
A, B = C.letter_ids(tok)
items = [it for it in C.load(f"{a.lang}_quant_{a.split}") if it["kind"] in ("nonex", "allx")]


def span(it, role):
    return "".join(it["msg"][s:e] for r, s, e in it["roles"] if r == role)


def variants(it):
    m = it["msg"]
    out = {"base": m}
    lines = m.split("\n")
    intro, x = span(it, "t"), span(it, "xn").strip()
    assert lines[2].startswith(intro), it["id"]
    sent = lines[2][len(intro):]
    if a.lang == "fr":
        tail = f", sauf {x}."
        assert sent.endswith(tail), it["id"]
        core = sent[: -len(tail)].strip()
        ln = list(lines); ln[2] = f"{intro} Sauf {x}, {core[0].lower() + core[1:]}."
        out["pre"] = "\n".join(ln)
        return out
    plain0 = sent.replace(f" {x} को छोड़कर", "")
    assert plain0 != sent and plain0.endswith("।"), it["id"]
    for name, w in (("post", "को छोड़कर"), ("post_siva", "के सिवा")):
        ln = list(lines); ln[2] = f"{intro}{plain0[:-1]}, {x} {w}।"
        out[name] = "\n".join(ln)
    for name, w in (("alava", " के अलावा"), ("siva", " के सिवा")):
        assert " को छोड़कर" in m
        out[name] = m.replace(" को छोड़कर", w)
    if it["kind"] == "nonex":
        lines = m.split("\n")
        intro = span(it, "t")
        x, qo = span(it, "xn").strip(), span(it, "qo")
        qn = span(it, "qn")
        fact = qo[len(qn):] if qo.startswith(qn) else qo          # " को दही पसंद" or " के पास किताब"
        assert lines[2].startswith(intro), it["id"]
        sent = lines[2][len(intro):]
        plain = sent.replace(f" {x} को छोड़कर", "")               # "किसी को ... नहीं है।"
        assert plain != sent, it["id"]
        lines2 = list(lines); lines2[2] = f"{intro} केवल {x}{fact} है।"
        out["only"] = "\n".join(lines2)
        lines3 = list(lines); lines3[2] = f"{intro}{plain} लेकिन {x}{fact} है।"
        out["split"] = "\n".join(lines3)
    return out


rows = []
jobs = [(it, name, msg) for it in items for name, msg in variants(it).items()]
# print(jobs[0][2]); print(jobs[3][2]); print(jobs[4][2])
jobs.sort(key=lambda j: len(j[2]))
for b in C.batches(len(jobs), a.bs):
    jb = [jobs[i] for i in b]
    enc = tok([C.chat(j[2], a.lang) for j in jb], add_special_tokens=False)["input_ids"]
    ids, mask, _ = C.pad(tok, enc)
    with torch.no_grad():
        lg = C.last_logits(model, ids, mask)
    for j, (it, name, msg) in enumerate(jb):
        ld = (lg[j, A] - lg[j, B]).item()
        yes = (ld > 0) == (it["ord"] == 0)
        rows.append({"kind": it["kind"], "who": "exc" if it["qi"] == 2 else "rest", "var": name, "f": it["f"], "ord": it["ord"],
                     "ok": int((ld > 0) == (it["ans"] == "A")), "yes": int(yes), "ld": ld if it["ans"] == "A" else -ld})
g = defaultdict(list)
for r in rows: g[(r["kind"], r["var"], r["who"])].append(r)
tab = []
for k, v in sorted(g.items()):
    tab.append({"kind": k[0], "var": k[1], "who": k[2], "n": len(v), "acc": sum(r["ok"] for r in v) / len(v),
                "yes": sum(r["yes"] for r in v) / len(v), "ld": sum(r["ld"] for r in v) / len(v)})
    print({x: (round(y, 3) if isinstance(y, float) else y) for x, y in tab[-1].items()}, flush=True)
ex = {name: msg for name, msg in variants(next(it for it in items if it["kind"] == "nonex")).items()}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump({"model": C.MODEL, "table": tab, "examples": ex, "rows": rows}, open(a.out, "w"), ensure_ascii=False, indent=1)
