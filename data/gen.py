import json, gzip, random, os, sys
from itertools import permutations

# (latin, devanagari); latin form is used in french prompts
N = [("Asha","आशा"),("Noor","नूर"),("Ravi","रवि"),("Meera","मीरा"),("Arjun","अर्जुन"),
     ("Kabir","कबीर"),("Priya","प्रिया"),("Rohan","रोहन"),("Sita","सीता"),("Vikram","विक्रम"),
     ("Anjali","अंजलि"),("Dev","देव"),("Neha","नेहा"),("Karan","करण"),("Leela","लीला"),
     ("Mohan","मोहन"),("Isha","ईशा"),("Aman","अमन"),("Tara","तारा"),("Nikhil","निखिल"),
     ("Pooja","पूजा"),("Sameer","समीर"),("Radha","राधा"),("Varun","वरुण"),("Kavya","काव्या"),
     ("Imran","इमरान"),("Zoya","ज़ोया"),("Farhan","फ़रहान"),("Gita","गीता"),("Hari","हरि"),
     ("Lata","लता"),("Manoj","मनोज"),("Nisha","निशा"),("Om","ओम"),("Rekha","रेखा"),
     ("Suresh","सुरेश"),("Uma","उमा"),("Vijay","विजय"),("Yash","यश"),("Rani","रानी")]

# family 1, preference: (fr with article, hi)
O1 = [("le thé","चाय"),("le café","कॉफ़ी"),("le lait","दूध"),("la musique","संगीत"),
      ("le cricket","क्रिकेट"),("la pluie","बारिश"),("le chocolat","चॉकलेट"),("la glace","आइसक्रीम"),
      ("le football","फ़ुटबॉल"),("les échecs","शतरंज"),("le riz","चावल"),("le cinéma","सिनेमा"),
      ("le yaourt","दही"),("la mer","समुद्र"),("la chaleur","गर्मी"),("les sucreries","मिठाई")]

# family 2, possession: (fr noun, gender, hi)
O2 = [("vélo","m","साइकिल"),("voiture","f","कार"),("livre","m","किताब"),("parapluie","m","छाता"),
      ("montre","f","घड़ी"),("chien","m","कुत्ता"),("chat","m","बिल्ली"),("téléphone","m","फ़ोन"),
      ("ordinateur","m","कंप्यूटर"),("billet","m","टिकट"),("clé","f","चाबी"),("sac","m","बैग"),
      ("stylo","m","पेन"),("scooter","m","स्कूटर"),("appareil photo","m","कैमरा"),("lampe de poche","f","टॉर्च")]

# family 3, ability, held out: (fr, hi)
O3 = [("nager","तैरना"),("chanter","गाना"),("danser","नाचना"),("cuisiner","खाना बनाना"),
      ("conduire","गाड़ी चलाना"),("faire du vélo","साइकिल चलाना"),("jouer aux échecs","शतरंज खेलना"),
      ("dessiner","चित्र बनाना"),("coudre","सिलाई करना"),("jouer de la guitare","गिटार बजाना")]

INS = {"hi": "केवल A या B लिखिए।", "fr": "Répondez seulement par A ou B."}
OBJ = {1: O1, 2: O2, 3: O3}

def vow(s): return s[0] in "aeiouéèêàâ"

def subj(L, f, x):
    # x is None for the question word
    if L == "fr": return [(x, "n")] if x else ["Qui"]
    if f == 2: return [(x, "n"), " के पास"] if x else ["किसके पास"]
    return [(x, "n"), " को"] if x else ["किसे"]

def pred(L, f, o, ng):
    if L == "hi":
        h = o[-1]
        if f == 3: return [f" {h} ", ("नहीं", "c"), " आता"] if ng else [f" {h} आता है"]
        p = f" {h} पसंद " if f == 1 else f" {h} "
        return [p, ("नहीं", "c"), " है"] if ng else [p + "है"]
    if f == 1: return [" ", ("n'", "c"), "aime ", ("pas", "c"), f" {o[0]}"] if ng else [f" aime {o[0]}"]
    if f == 2:
        if ng: return [" ", ("n'", "c"), "a ", ("pas", "c"), " " + ("d'" if vow(o[0]) else "de ") + o[0]]
        return [f" a {'un' if o[1] == 'm' else 'une'} {o[0]}"]
    return [" ", ("ne", "c"), " sait ", ("pas", "c"), f" {o[0]}"] if ng else [f" sait {o[0]}"]

def name(L, i): return N[i][0] if L == "fr" else N[i][1]

def build(L, segs, tag):
    s, sp = "", {}
    for g in segs:
        t, k = (g, None) if isinstance(g, str) else g
        if k:
            kk = tag + k
            a, b = sp.get(kk, (len(s), 0))
            sp[kk] = (a, len(s) + len(t))
        s += t
    return s, sp

def item(L, f, a, b, oi, ng, q, od, o2i=None):
    O = OBJ[1 if f == 0 else f]
    o = O[oi]
    if f == 0:
        # affirmative control: two objects, question asks about object q
        oo = O[o2i]
        c1 = subj(L, 1, name(L, a)) + pred(L, 1, o, 0)
        c2 = subj(L, 1, name(L, b)) + pred(L, 1, oo, 0)
        qs = subj(L, 1, None) + pred(L, 1, o if q == 0 else oo, 0)
        tgt = a if q == 0 else b
    else:
        c1 = subj(L, f, name(L, a)) + pred(L, f, o, ng == 0)
        c2 = subj(L, f, name(L, b)) + pred(L, f, o, ng == 1)
        qs = subj(L, f, None) + pred(L, f, o, q)
        neg = a if ng == 0 else b
        tgt = neg if q else (b if neg == a else a)
    j = ", लेकिन " if L == "hi" else ", mais "
    if f == 0: j = ", और " if L == "hi" else ", et "
    e = "।" if L == "hi" else "."
    ctx, sp = build(L, c1 + [j] + c2 + [e], "")
    # name spans by mention order
    s1, _ = build(L, c1, "")
    p1 = (0, len(name(L, a)))
    p2 = (len(s1) + len(j), len(s1) + len(j) + len(name(L, b)))
    qq, qsp = build(L, qs + [" ?" if L == "fr" else "?"], "q")
    o1, o2 = (a, b) if od == 0 else (b, a)
    sep = " : " if L == "fr" else ": "
    opts = f"A{sep}{name(L, o1)}\nB{sep}{name(L, o2)}"
    pre = INS[L] + "\n\n"
    msg = pre + ctx + "\n" + qq + "\n" + opts
    off, qoff = len(pre), len(pre) + len(ctx) + 1
    S = {"n1": [off + p1[0], off + p1[1]], "n2": [off + p2[0], off + p2[1]],
         "e1": [off + len(s1), off + len(s1) + 1], "e2": [off + len(ctx) - 1, off + len(ctx)]}
    if "c" in sp: S["cue"] = [off + sp["c"][0], off + sp["c"][1]]
    if "qc" in qsp: S["qcue"] = [qoff + qsp["qc"][0], qoff + qsp["qc"][1]]
    # option-name spans: a causal model can bind polarity here, not at the first mention
    x = qoff + len(qq) + 1 + len("A" + sep)
    S["oA"] = [x, x + len(name(L, o1))]
    x = S["oA"][1] + 1 + len("B" + sep)
    S["oB"] = [x, x + len(name(L, o2))]
    ans = "A" if tgt == o1 else "B"
    return {"lang": L, "fam": f, "a": N[a][0], "b": N[b][0], "obj": oi, "neg": ng, "q": q,
            "ord": od, "msg": msg, "ans": ans, "sp": S}

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(out, exist_ok=True)
    R = random.Random(13)
    ni = list(range(len(N))); R.shuffle(ni)
    NS = {"disc": ni[:20], "dev": ni[20:30], "test": ni[30:]}
    OS = {}
    for f in (1, 2):
        oi = list(range(len(OBJ[f]))); R.shuffle(oi)
        OS[f] = {"disc": oi[:8], "dev": oi[8:12], "test": oi[12:]}
    OS[3] = {"test": list(range(len(O3)))}
    NC = {"disc": 2000, "dev": 500, "test": 500}
    tot = {}
    for sp in ("disc", "dev", "test"):
        for L in ("hi", "fr"):
            fn = os.path.join(out, f"{L}_{sp}.jsonl.gz")
            k = 0
            with gzip.open(fn, "wt", encoding="utf-8") as w:
                for f in (1, 2, 3):
                    if sp not in OS[f]: continue
                    k0 = k
                    for a, b in permutations(NS[sp], 2):
                        for oi in OS[f][sp]:
                            for q in (0, 1):
                                for od in (0, 1):
                                    pid = f"{sp}-f{f}-{a}-{b}-{oi}-{q}-{od}"
                                    for ng in (0, 1):
                                        it = item(L, f, a, b, oi, ng, q, od)
                                        it["pair"], it["id"] = pid, f"{pid}-n{ng}"
                                        w.write(json.dumps(it, ensure_ascii=False) + "\n"); k += 1
                    tot[(sp, L, f)] = k - k0
                R2 = random.Random({"disc": 1, "dev": 2, "test": 3}[sp])
                P = list(permutations(NS[sp], 2))
                for t in range(NC[sp]):
                    a, b = P[R2.randrange(len(P))]
                    oi, o2i = R2.sample(OS[1][sp], 2)
                    q, od = R2.randrange(2), R2.randrange(2)
                    it = item(L, 0, a, b, oi, -1, q, od, o2i)
                    it["pair"], it["id"] = f"{sp}-f0-{t}", f"{sp}-f0-{t}"
                    w.write(json.dumps(it, ensure_ascii=False) + "\n"); k += 1
            # print(fn, k)
    for sp in ("disc", "dev", "test"):
        for f in (1, 2, 3):
            if (sp, "hi", f) in tot:
                print(sp, f"family{f}", "items/lang", tot[(sp, "hi", f)], "pairs/lang", tot[(sp, "hi", f)] // 2)
        print(sp, "controls/lang", NC[sp])
    full = len(N) * (len(N) - 1) * 4 * (len(O1) + len(O2) + len(O3))
    print("all minimal pairs per language, no split:", full)

if __name__ == "__main__":
    main()
