"""Four fixed examples for each language, with names and objects that are in no split.
They are balanced: negated mention x question polarity, the answer letter and the answer mention.
Usage: python shots.py   (writes shots.json)"""
import json, os
import gen

n0 = len(gen.N)
gen.N += [("Deepak", "दीपक"), ("Maya", "माया"), ("Sunil", "सुनील"), ("Kiran", "किरण"),
          ("Alok", "आलोक"), ("Jyoti", "ज्योति"), ("Naveen", "नवीन"), ("Shreya", "श्रेया")]
o1, o2 = len(gen.O1), len(gen.O2)
gen.O1 += [("le fromage", "पनीर"), ("le miel", "शहद")]
gen.O2 += [("piano", "m", "पियानो"), ("tente", "f", "तंबू")]
# (family, first name, second name, object, negated mention, question polarity, option order)
S = [(1, n0, n0 + 1, o1, 0, 0, 0), (2, n0 + 2, n0 + 3, o2, 1, 1, 1),
     (1, n0 + 4, n0 + 5, o1 + 1, 0, 1, 0), (2, n0 + 6, n0 + 7, o2 + 1, 1, 0, 1)]
out = {}
for L in ("hi", "fr"):
    out[L] = []
    for f, a, b, o, ng, q, od in S:
        it = gen.item(L, f, a, b, o, ng, q, od)
        out[L].append({"msg": it["msg"], "ans": it["ans"]})
        print(L, it["ans"], it["msg"].split("\n\n", 1)[1].replace("\n", " / "))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots.json"), "w"), ensure_ascii=False, indent=1)
