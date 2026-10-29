"""Extended data: negation forms, scope contrasts and explicit worlds, in Hindi and French.

Usage: python gen2.py [out_dir]      (also writes templates.md, one example for each template)

All prompts use the same layout as gen.py: instruction, context, question, options A and B.
Each item has "roles": character spans (role, start, end) that cover the user message.
Sets with a "who/what" question and a moved negation (pairs differ only in "neg"):
  hab       sentential negation with habitual verbs (baseline for the temporal forms)
  never     ne ... jamais / kabhi ... nahin
  nolonger  ne ... plus / ab ... nahin, with a time context
  nothing   ne ... rien / kuchh nahin (negative indefinite in the object position)
  lex       lexical negation (absent) against explicit negation (not present)
  focus     constituent negation: the negation targets one of two objects or one of two persons
Sets with one yes/no question for each entity (options are yes and no, in both orders):
  quant     everyone / nobody, with and without an exception, and "not everyone" with facts
  w1        two persons, one predicate, all four polarity states
  nest      w1 where the fact about the queried person has an outer "it is not true that"
  w2        two persons, two predicates, all sixteen polarity states
"""
import gzip, json, os, random, sys
from itertools import permutations
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen import N, O1, O2, INS, vow

FEM = {"Asha", "Noor", "Meera", "Priya", "Sita", "Anjali", "Neha", "Leela", "Isha", "Tara", "Pooja", "Radha",
       "Kavya", "Zoya", "Gita", "Lata", "Nisha", "Rekha", "Uma", "Rani"}

# habitual predicates: fr verb, fr imperfect plural, fr object (affirmative), fr object (negative), hi object, hi stem
HAB = [("boit", "buvaient", "du thé", "de thé", "चाय", "पी"), ("boit", "buvaient", "du café", "de café", "कॉफ़ी", "पी"),
       ("boit", "buvaient", "du lait", "de lait", "दूध", "पी"), ("boit", "buvaient", "du jus", "de jus", "जूस", "पी"),
       ("mange", "mangeaient", "du riz", "de riz", "चावल", "खा"), ("mange", "mangeaient", "du chocolat", "de chocolat", "चॉकलेट", "खा"),
       ("mange", "mangeaient", "du yaourt", "de yaourt", "दही", "खा"), ("mange", "mangeaient", "de la glace", "de glace", "आइसक्रीम", "खा"),
       ("mange", "mangeaient", "des sucreries", "de sucreries", "मिठाई", "खा"), ("mange", "mangeaient", "de la viande", "de viande", "मांस", "खा"),
       ("regarde", "regardaient", "la télévision", "la télévision", "टीवी", "देख"), ("écoute", "écoutaient", "la radio", "la radio", "रेडियो", "सुन"),
       ("lit", "lisaient", "le journal", "le journal", "अख़बार", "पढ़"),
       ("joue", "jouaient", "au cricket", "au cricket", "क्रिकेट", "खेल"), ("joue", "jouaient", "au football", "au football", "फ़ुटबॉल", "खेल"),
       ("joue", "jouaient", "aux échecs", "aux échecs", "शतरंज", "खेल")]
NO_RIEN = {13, 14, 15}   # "jouer" takes no direct object, so it is not used with "rien"

# lexical pairs: fr positive (m, f), fr negative (m, f), hi positive, hi negative
LEX = [(("présent", "présente"), ("absent", "absente"), "उपस्थित", "अनुपस्थित"),
       (("heureux", "heureuse"), ("malheureux", "malheureuse"), "खुश", "नाखुश"),
       (("honnête", "honnête"), ("malhonnête", "malhonnête"), "ईमानदार", "बेईमान"),
       (("marié", "mariée"), ("célibataire", "célibataire"), "विवाहित", "अविवाहित"),
       (("satisfait", "satisfaite"), ("insatisfait", "insatisfaite"), "संतुष्ट", "असंतुष्ट"),
       (("prudent", "prudente"), ("imprudent", "imprudente"), "सावधान", "असावधान"),
       (("poli", "polie"), ("impoli", "impolie"), "शिष्ट", "अशिष्ट"),
       (("capable", "capable"), ("incapable", "incapable"), "सक्षम", "अक्षम"),
       (("responsable", "responsable"), ("irresponsable", "irresponsable"), "ज़िम्मेदार", "ग़ैर-ज़िम्मेदार"),
       (("content", "contente"), ("mécontent", "mécontente"), "प्रसन्न", "अप्रसन्न"),
       (("en bonne santé", "en bonne santé"), ("malade", "malade"), "स्वस्थ", "अस्वस्थ")]

YES = {"fr": ("Oui", "Non"), "hi": ("हाँ", "नहीं")}
SEP = {"fr": " : ", "hi": ": "}
END = {"fr": ".", "hi": "।"}
QM = {"fr": " ?", "hi": "?"}
BUT = {"fr": ", mais ", "hi": ", लेकिन "}
HI = "hi"


def nm(L, i): return N[i][0] if L == "fr" else N[i][1]
def fem(i): return N[i][0] in FEM
def que(x): return "qu'" if vow(x.lower()) else "que "       # elision before a vowel
def ne(v): return "n'" if vow(v) else "ne "


def assemble(L, ctx, q, oa, ob):
    """ctx and q are lists of (text, role). Returns the message and the role spans."""
    segs = [(INS[L] + "\n\n", "ins")] + ctx + [("\n", ctx[-1][1])] + q + [("\n", "q")]
    segs += [("A" + SEP[L], "lA"), (oa, "oA"), ("\nB" + SEP[L], "lB"), (ob, "oB")]
    msg, roles = "", []
    for t, r in segs:
        if not t: continue
        if roles and roles[-1][0] == r: roles[-1][2] += len(t)
        else: roles.append([r, len(msg), len(msg) + len(t)])
        msg += t
    return msg, roles


# ---------- clauses for the "who" sets. Each returns segments for one clause after the name. ----------

def hab_clause(L, p, g, form):
    """form: aff, pas, jamais, plus, rien. g: the subject is feminine."""
    v, _, oa, on, ho, st = HAB[p]
    if L == "fr":
        if form == "aff": return [(f" {v} {oa}", "body")]
        w = {"pas": "pas", "jamais": "jamais", "plus": "plus", "rien": "rien"}[form]
        return [(f" {ne(v)}{v} {w}", "cue")] + ([] if form == "rien" else [(f" {on}", "body")])
    t = st + ("ती" if g else "ता")
    if form == "aff": return [(f" {ho} {t} है", "body")]
    if form == "jamais": return [(" कभी", "cue"), (f" {ho}", "body"), (" नहीं", "cue"), (f" {t}", "body")]
    if form == "rien": return [(" कुछ नहीं", "cue"), (f" {t}", "body")]
    return [(f" {ho}", "body"), (" नहीं", "cue"), (f" {t}", "body")]


def hab_q(L, p, form):
    """Question about the habit. The question word takes the masculine form in Hindi."""
    v, _, oa, on, ho, st = HAB[p]
    if L == "fr":
        if form == "aff": return [("Qui", "q"), (f" {v} {oa}", "q")]
        if form == "now": return [("Qui", "q"), (f" {v} {oa} maintenant", "q")]
        w = {"pas": "pas", "jamais": "jamais", "plus": "plus", "rien": "rien"}[form]
        return [("Qui", "q"), (f" {ne(v)}{v} {w}", "qcue")] + ([] if form == "rien" else [(f" {on}", "q")])
    t = st + "ता"
    if form == "aff": return [("कौन", "q"), (f" {ho} {t} है", "q")]
    if form == "now": return [("अब कौन", "q"), (f" {ho} {t} है", "q")]
    if form == "jamais": return [("कौन", "q"), (" कभी", "qcue"), (f" {ho}", "q"), (" नहीं", "qcue"), (f" {t}", "q")]
    if form == "plus": return [("अब कौन", "q"), (f" {ho}", "q"), (" नहीं", "qcue"), (f" {t}", "q")]
    if form == "rien": return [("कौन", "q"), (" कुछ नहीं", "qcue"), (f" {t}", "q")]
    return [("कौन", "q"), (f" {ho}", "q"), (" नहीं", "qcue"), (f" {t}", "q")]


def lex_clause(L, p, g, form):
    """form: aff (is present), lex (is absent), exp (is not present)."""
    pf, nf, hp, hn = LEX[p]
    if L == "fr":
        if form == "aff": return [(f" est {pf[g]}", "body")]
        if form == "lex": return [(" est", "body"), (f" {nf[g]}", "cue")]
        return [(" n'est pas", "cue"), (f" {pf[g]}", "body")]
    if form == "aff": return [(f" {hp} है", "body")]
    if form == "lex": return [(f" {hn}", "cue"), (" है", "body")]
    return [(f" {hp}", "body"), (" नहीं", "cue"), (" है", "body")]


def lex_q(L, p, form):
    pf, nf, hp, hn = LEX[p]
    if L == "fr":
        if form == "aff": return [(f"Qui est {pf[0]}", "q")]
        if form == "lex": return [("Qui est", "q"), (f" {nf[0]}", "qcue")]
        return [("Qui", "q"), (" n'est pas", "qcue"), (f" {pf[0]}", "q")]
    if form == "aff": return [(f"कौन {hp} है", "q")]
    if form == "lex": return [("कौन", "q"), (f" {hn}", "qcue"), (" है", "q")]
    return [(f"कौन {hp}", "q"), (" नहीं", "qcue"), (" है", "q")]


def tag(segs, i):
    """Add the clause number to body roles."""
    return [(t, r + str(i) if r == "body" else r) for t, r in segs]


def who_item(L, st, a, b, ng, q, od, c1, c2, qs, pre=None):
    """Two clauses about a and b; the clause of mention ng is negative. q: 0 positive, 1 negative question."""
    ctx = ([(pre, "t")] if pre else []) + [(nm(L, a), "n1")] + tag(c1, 1) + [(BUT[L][0], "e1"), (BUT[L][1:], "conj")]
    ctx += [(nm(L, b), "n2")] + tag(c2, 2) + [(END[L], "e2")]
    neg = a if ng == 0 else b
    tgt = neg if q else (b if neg == a else a)
    o1, o2 = (a, b) if od == 0 else (b, a)
    msg, roles = assemble(L, ctx, qs + [(QM[L], "q")], nm(L, o1), nm(L, o2))
    return {"lang": L, "set": st, "a": N[a][0], "b": N[b][0], "neg": ng, "q": q, "ord": od, "msg": msg,
            "ans": "A" if tgt == o1 else "B", "roles": roles}


def make_who(L, st, sp, a, b, p, q, od, ng, v):
    """v is the variant of the set (a string)."""
    ga, gb = fem(a), fem(b)
    if st in ("hab", "never", "nolonger", "nothing"):
        f = {"hab": "pas", "never": "jamais", "nolonger": "plus", "nothing": "rien"}[st]
        c1 = hab_clause(L, p, ga, f if ng == 0 else "aff")
        c2 = hab_clause(L, p, gb, f if ng == 1 else "aff")
        pre = None
        if st == "nolonger":
            _, imp, oa, _, ho, stem = HAB[p]
            if L == "fr": pre = f"Avant, {nm(L, a)} et {nm(L, b)} {imp} {oa}. Maintenant, "
            else: pre = f"पहले {nm(L, a)} और {nm(L, b)} दोनों {ho} {stem}{'ती थीं' if ga and gb else 'ते थे'}। अब "
            qs = hab_q(L, p, f if q else "now")
        else:
            qs = hab_q(L, p, f if q else "aff")
        it = who_item(L, st, a, b, ng, q, od, c1, c2, qs, pre)
        it["fam"] = st
    elif st == "lex":
        cf, qf = v.split("-")            # context form (lex, exp) and form of the negative question
        c1 = lex_clause(L, p, ga, cf if ng == 0 else "aff")
        c2 = lex_clause(L, p, gb, cf if ng == 1 else "aff")
        it = who_item(L, st, a, b, ng, q, od, c1, c2, lex_q(L, p, qf if q else "aff"))
        it["fam"] = "lex-" + (v if q else cf + "-pos")
    it["obj"] = p
    return it


def make_focus(L, sp, a, b, p, p2, q, od, ng, v):
    """Constituent negation. v=obj: one person, two objects. v=per: two persons, one object.
    ng is the negated constituent in mention order (0 first, 1 second)."""
    o, oo = O1[p], O1[p2]
    A, B = nm(L, a), nm(L, b)
    if v == "obj":
        x, y = (o[0], oo[0]) if L == "fr" else (o[1], oo[1])
        if L == "fr":
            if ng == 1: ctx = [(A, "s"), (" aime", "body1"), (f" {x}", "n1"), (",", "e1"), (" pas", "cue"), (f" {y}", "n2"), (".", "e2")]
            else: ctx = [(A, "s"), (" n'aime pas", "cue"), (f" {x}", "n1"), (",", "e1"), (" mais", "conj"), (f" {y}", "n2"), (".", "e2")]
            qs = [(f"Qu'est-ce {que(A)}{A}", "q")] + ([(" n'aime pas", "qcue")] if q else [(" aime", "q")]) + [(" ?", "q")]
        else:
            if ng == 1: ctx = [(A + " को", "s"), (f" {x}", "n1"), (" पसंद है", "body1"), (",", "e1"), (f" {y}", "n2"), (" नहीं", "cue"), ("।", "e2")]
            else: ctx = [(A + " को", "s"), (f" {x}", "n1"), (" नहीं", "cue"), (",", "e1"), (f" {y}", "n2"), (" पसंद है", "body2"), ("।", "e2")]
            qs = [(f"{A} को क्या पसंद", "q")] + ([(" नहीं", "qcue")] if q else []) + [(" है?", "q")]
        c = (x, y)
    else:
        x = o[0] if L == "fr" else o[1]
        if L == "fr":
            if ng == 1: ctx = [("C'est ", "t"), (A, "n1"), (",", "e1"), (" et non", "cue"), (f" {B}", "n2"), (f", qui aime {x}", "body2"), (".", "e2")]
            else: ctx = [("Ce n'est pas", "cue"), (f" {A}", "n1"), (",", "e1"), (" mais", "conj"), (f" {B}", "n2"), (f", qui aime {x}", "body2"), (".", "e2")]
            qs = [("Qui", "q")] + ([(" n'aime pas", "qcue")] if q else [(" aime", "q")]) + [(f" {x} ?", "q")]
        else:
            if ng == 1: ctx = [(f"{x} ", "t"), (A, "n1"), (" को पसंद है", "body1"), (",", "e1"), (f" {B}", "n2"), (" को", "body2"), (" नहीं", "cue"), ("।", "e2")]
            else: ctx = [(f"{x} ", "t"), (A, "n1"), (" को", "body1"), (" नहीं", "cue"), (",", "e1"), (f" {B}", "n2"), (" को पसंद है", "body2"), ("।", "e2")]
            qs = [(f"किसे {x} पसंद", "q")] + ([(" नहीं", "qcue")] if q else []) + [(" है?", "q")]
        c = (A, B)
    tgt = ng if q else 1 - ng
    o1, o2 = (0, 1) if od == 0 else (1, 0)
    msg, roles = assemble(L, ctx, qs, c[o1], c[o2])
    return {"lang": L, "set": "focus", "fam": "focus-" + v, "a": N[a][0], "b": N[b][0], "obj": p, "obj2": p2, "neg": ng,
            "q": q, "ord": od, "msg": msg, "ans": "A" if tgt == o1 else "B", "roles": roles}


# ---------- yes/no sets ----------

def fact(L, f, x, o, pol, i, outer=False):
    """One sentence: person x has polarity pol for object o. f: 1 preference, 2 possession.
    outer adds "it is not true that" around a clause of the opposite polarity."""
    inner = pol if not outer else 1 - pol
    k = f"f{i}"
    if L == "fr":
        if f == 1: body = [(" aime", k + "b")] if inner else [(" n'aime pas", k + "c")]
        else:
            a = "ait" if outer else "a"      # subjunctive after "il n'est pas vrai que"
            body = [(f" {a}", k + "b")] if inner else [(f" n'{a} pas", k + "c")]
        if f == 1: ob = " " + o[0]
        else: ob = (" un " if o[1] == "m" else " une ") + o[0] if inner else (" d'" if vow(o[0]) else " de ") + o[0]
        pre = [("Il n'est pas vrai " + que(x), k + "x")] if outer else []
        return pre + [(x, k + "n")] + body + [(ob, k + "b"), (".", k + "e")]
    h = o[-1]
    mid = " को" if f == 1 else " के पास"
    tail = " पसंद" if f == 1 else ""
    pre = [("यह सच नहीं है कि ", k + "x")] if outer else []
    return pre + [(x, k + "n"), (f"{mid} {h}{tail}", k + "b")] + ([] if inner else [(" नहीं", k + "c")]) + [(" है", k + "b"), ("।", k + "e")]


def ynq(L, f, x, o):
    """Yes/no question: does x like / have o?"""
    if L == "fr":
        v = " aime " + o[0] if f == 1 else " a " + ("un " if o[1] == "m" else "une ") + o[0]
        return [("Est-ce " + que(x), "q"), (x, "qn"), (v, "qo"), (" ?", "q")]
    return [("क्या ", "q"), (x, "qn"), ((" को " if f == 1 else " के पास ") + o[-1] + (" पसंद" if f == 1 else ""), "qo"), (" है?", "q")]


def yn_item(L, st, ctx, qs, truth, od):
    y, n = YES[L]
    msg, roles = assemble(L, ctx, qs, y if od == 0 else n, n if od == 0 else y)
    return {"lang": L, "set": st, "ord": od, "truth": truth, "msg": msg, "roles": roles,
            "ans": "A" if (truth == 1) == (od == 0) else "B"}


def join(L, facts):
    out = []
    for i, f in enumerate(facts):
        if i: f = [(" " + f[0][0], f[0][1])] + f[1:]
        out += f
    return out


def make_world(L, st, f, a, b, objs, pol, qe, qp, od, outer=0):
    """pol: polarity for each (person, object) in mention order. qe, qp: the queried person and object."""
    ents = (a, b)
    O = O1 if f == 1 else O2
    facts, i = [], 0
    for e in range(2):
        for p in range(len(objs)):
            i += 1
            facts.append(fact(L, f, nm(L, ents[e]), O[objs[p]], pol[e * len(objs) + p], i,
                              outer=bool(outer) and e == qe and p == qp))
    truth = pol[qe * len(objs) + qp]
    it = yn_item(L, st, join(L, facts), ynq(L, f, nm(L, ents[qe]), O[objs[qp]]), truth, od)
    it.update(fam=f"{st}-f{f}", a=N[a][0], b=N[b][0], objs=list(objs), pol=list(pol), qe=qe, qp=qp, outer=outer)
    return it


def make_quant(L, f, names, o, kind, qi, od):
    """Three persons. kind: all, none, allx, nonex (x: the third person in the list is the exception), notall.
    qi: index of the queried person."""
    O = O1 if f == 1 else O2
    ob = O[o]
    X = [nm(L, i) for i in names]
    if kind == "notall": exc = 2
    if L == "fr":
        intro = f"Il y a trois personnes : {X[0]}, {X[1]} et {X[2]}."
        pos = (" aime " + ob[0]) if f == 1 else (" a " + ("un " if ob[1] == "m" else "une ") + ob[0])
        neg = (" n'aime pas " + ob[0]) if f == 1 else (" n'a pas " + ("d'" if vow(ob[0]) else "de ") + ob[0])
        if kind in ("all", "allx"): s = [(" Tout le monde", "qt"), (pos, "f1b")]
        elif kind in ("none", "nonex"): s = [(" Personne", "qt"), (neg.replace(" pas", ""), "f1c")]
        else: s = [(" Tout le monde", "qt"), (neg, "f1c")]
        if kind in ("allx", "nonex"): s += [(", sauf", "x"), (" " + X[2], "xn")]
        s += [(".", "f1e")]
        if kind == "notall":
            pl = (" aiment " + ob[0]) if f == 1 else (" ont " + ("un " if ob[1] == "m" else "une ") + ob[0])
            s += [(f" {X[0]} et {X[1]}", "xn"), (pl, "f2b"), (".", "f2e")]
    else:
        intro = f"तीन लोग हैं: {X[0]}, {X[1]} और {X[2]}।"
        h = ob[-1]
        mid = " को" if f == 1 else " के पास"
        tail = (" पसंद" if f == 1 else "")
        allw, nonew = ("सबको", "किसी को") if f == 1 else ("सबके पास", "किसी के पास")
        s = []
        if kind in ("allx", "nonex"): s += [(" " + X[2], "xn"), (" को छोड़कर", "x")]
        if kind in ("all", "allx"): s += [(" " + allw, "qt"), (f" {h}{tail} है", "f1b")]
        elif kind in ("none", "nonex"): s += [(" " + nonew, "qt"), (f" {h}{tail}", "f1b"), (" नहीं", "f1c"), (" है", "f1b")]
        else: s += [(" " + allw, "qt"), (f" {h}{tail}", "f1b"), (" नहीं", "f1c"), (" है", "f1b")]
        s += [("।", "f1e")]
        if kind == "notall": s += [(f" {X[0]} और {X[1]}", "xn"), (f"{mid} {h}{tail} है", "f2b"), ("।", "f2e")]
    truth = {"all": 1, "none": 0, "allx": int(qi != 2), "nonex": int(qi == 2), "notall": int(qi != 2)}[kind]
    it = yn_item(L, "quant", [(intro, "t")] + s, ynq(L, f, X[qi], ob), truth, od)
    it.update(fam="quant-" + kind + ("" if kind in ("all", "none") else "-exc" if qi == 2 else "-rest"),
              a=N[names[0]][0], b=N[names[1]][0], c=N[names[2]][0], obj=o, qi=qi, kind=kind, f=f)
    return it


# ---------- splits and sampling ----------

def splits():
    """The same name and object splits as gen.py, and splits for the new predicates."""
    R = random.Random(13)
    ni = list(range(len(N))); R.shuffle(ni)
    NS = {"disc": ni[:20], "dev": ni[20:30], "test": ni[30:]}
    OS = {}
    for f in (1, 2):
        oi = list(range(len(O1 if f == 1 else O2))); R.shuffle(oi)
        OS[f] = {"disc": oi[:8], "dev": oi[8:12], "test": oi[12:]}
    R2 = random.Random(14)
    h = list(range(len(HAB))); R2.shuffle(h)
    OS["hab"] = {"disc": h[:8], "dev": h[8:12], "test": h[12:]}
    x = list(range(len(LEX))); R2.shuffle(x)
    OS["lex"] = {"disc": x[:5], "dev": x[5:8], "test": x[8:]}
    return NS, OS


SIZE = {"disc": 2000, "dev": 400, "test": 800}       # pairs for each "who" set
YSIZE = {"disc": 6000, "dev": 1000, "test": 2000}    # items for each yes/no set


def specs(st, sp, NS, OS):
    """Abstract items (the same for both languages). Each is (function, kwargs, pair id)."""
    R = random.Random(f"{st}-{sp}")
    P = list(permutations(NS[sp], 2))
    out = []
    if st in ("hab", "never", "nolonger", "nothing", "lex", "focus"):
        if st == "lex": vs = ["lex-lex", "lex-exp", "exp-lex", "exp-exp"]
        elif st == "focus": vs = ["obj", "per"]
        else: vs = [""]
        k = 0
        while k < SIZE[sp]:
            a, b = P[R.randrange(len(P))]
            v, q, od = vs[k % len(vs)], (k // len(vs)) % 2, (k // (2 * len(vs))) % 2
            if st == "focus":
                p, p2 = R.sample(OS[1][sp], 2)
                kw = dict(a=a, b=b, p=p, p2=p2, q=q, od=od, v=v)
            else:
                pool = OS["lex" if st == "lex" else "hab"][sp]
                if st == "nothing": pool = [x for x in pool if x not in NO_RIEN]
                kw = dict(st=st, a=a, b=b, p=R.choice(pool), q=q, od=od, v=v)
            for ng in (0, 1):
                out.append(("focus" if st == "focus" else "who", dict(kw, ng=ng), f"{st}-{sp}-{k}", f"{st}-{sp}-{k}-n{ng}"))
            k += 1
    elif st in ("w1", "nest", "w2"):
        npred = 2 if st == "w2" else 1
        for k in range(YSIZE[sp]):
            a, b = P[R.randrange(len(P))]
            f = 1 + k % 2
            objs = R.sample(OS[f][sp], npred)
            nf = 2 * npred
            c = k // 2
            pol = [(c >> i) & 1 for i in range(nf)]; c >>= nf
            qe = c % 2; c //= 2
            qp = c % npred; c //= npred
            od = c % 2; c //= 2
            outer = 1 if st == "nest" else 0
            out.append(("world", dict(st=st, f=f, a=a, b=b, objs=objs, pol=pol, qe=qe, qp=qp, od=od, outer=outer),
                        f"{st}-{sp}-{k}", f"{st}-{sp}-{k}"))
    elif st == "quant":
        kinds = ["all", "none", "allx", "nonex", "allx", "nonex", "notall", "notall"]
        T = list(permutations(NS[sp], 3))
        for k in range(YSIZE[sp] // 2):
            names = T[R.randrange(len(T))]
            f = 1 + k % 2
            kind = kinds[(k // 2) % len(kinds)]
            c = k // (2 * len(kinds))
            qi = 2 if (k // 2) % len(kinds) in (2, 3, 6) else c % 2 if kind in ("allx", "nonex", "notall") else c % 3
            od = (c // 3) % 2
            o = R.choice(OS[f][sp])
            for kd in ([kind] if kind == "notall" else [kind, {"all": "none", "none": "all", "allx": "nonex", "nonex": "allx"}[kind]]):
                out.append(("quant", dict(f=f, names=names, o=o, kind=kd, qi=qi, od=od), f"quant-{sp}-{k}", f"quant-{sp}-{k}-{kd}"))
    return out


def render(L, sp, s):
    fn, kw, pid, iid = s
    if fn == "who": it = make_who(L, kw["st"], sp, kw["a"], kw["b"], kw["p"], kw["q"], kw["od"], kw["ng"], kw["v"])
    elif fn == "focus": it = make_focus(L, sp, **kw)
    elif fn == "world": it = make_world(L, **kw)
    else: it = make_quant(L, **kw)
    it.update(split=sp, pair=pid, id=iid)
    assert it["roles"][-1][2] == len(it["msg"])
    return it


SETS = ["hab", "never", "nolonger", "nothing", "lex", "focus", "quant", "w1", "nest", "w2"]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    NS, OS = splits()
    doc = ["# Templates to validate", "",
           "One example for each template. A fluent reader must check the grammar and the answer.", ""]
    for st in SETS:
        for sp in ("disc", "dev", "test"):
            S = specs(st, sp, NS, OS)
            for L in ("hi", "fr"):
                items = [render(L, sp, s) for s in S]
                with gzip.open(os.path.join(out, f"{L}_{st}_{sp}.jsonl.gz"), "wt", encoding="utf-8") as w:
                    for it in items: w.write(json.dumps(it, ensure_ascii=False) + "\n")
                if sp == "dev":
                    seen = set()
                    for it in items:
                        key = (it["fam"], it.get("neg"), it.get("q"), it.get("outer"), tuple(it.get("pol", []))[:2] if st != "w2" else 0)
                        if key in seen or it["ord"] != 0: continue
                        seen.add(key)
                        body = it["msg"].split("\n\n", 1)[1].replace("\n", " / ")
                        doc.append(f"- `{L}` `{it['fam']}` neg={it.get('neg', '-')} q={it.get('q', '-')}, answer **{it['ans']}**: {body}")
            print(st, sp, "items/lang", len(S))
        doc.append("")
    open(os.path.join(out, "templates.md"), "w", encoding="utf-8").write("\n".join(doc))


if __name__ == "__main__":
    main()
