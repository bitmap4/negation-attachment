"""E4: can the released activation oracle read the attachment from the states of the model?

The oracle is a LoRA adapter on the same model (Karvonen et al.). It gets activations from
layer 25%, 50% or 75% of the depth, injected at placeholder tokens after layer 1, and answers a
question in English. Baselines: the same injection with the adapter off (Patchscopes style),
and a linear probe on the same activation.

Modes:
  readout  accuracy of the oracle against the known labels, for each site and question
  interv   we patch the model (as in E2) and ask the oracle about states after the patch.
           The answer of the oracle must change when, and only when, the answer of the model changes.

Usage: python -m neg.e4_oracle <out.json> --mode readout --sets fr_test hi_test [--n 300]
       python -m neg.e4_oracle <out.json> --mode interv --langs fr hi [--n 96]
"""
import argparse, json, os, random
import numpy as np
import torch
from . import common as C
from .pairs import build, sgn

ORACLES = {"Qwen/Qwen3-1.7B": "adamkarvonen/checkpoints_cls_latentqa_past_lens_Qwen3-1_7B",
           "Qwen/Qwen3-4B": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Qwen3-4B"}
ORACLE = os.environ.get("NEG_ORACLE", ORACLES[C.MODEL])
VERB = {1: "like", 2: "have", 3: "know how to do"}
# site name -> (roles, use all tokens of the roles; else the last token only)
SITES = {"e1": (["e1"], False), "e2": (["e2"], False), "cue": (["cue"], False), "oA": (["oA"], False),
         "oB": (["oB"], False), "end": (["end"], False),
         "win1": (["n1", "body1", "cue1", "e1"], True), "win2": (["n2", "body2", "cue2", "e2"], True),
         "ctx": (["n1", "body1", "cue", "e1", "conj", "n2", "body2", "e2"], True)}
PFX = "Answer with 'Yes' or 'No' only. "
C1, C2 = ["n1", "body1", "e1"], ["n2", "body2", "e2"]
# interventions: name -> (roles to patch from the partner item, layer)
INTERV = {"none": (None, 0), "c12@13": (C1 + ["conj"] + C2, 13), "c1@13": (C1, 13), "c2@13": (C2, 13),
          "opt@19": (["lA", "oA", "lB", "oB"], 19), "end@26": (["end"], 26)}


def oracle_config():
    """Layers that the oracle reads (25, 50 and 75% of the depth) and the layer where it injects."""
    from huggingface_hub import hf_hub_download
    try:
        cfg = json.load(open(hf_hub_download(ORACLE, "ao_config.json")))
        return [c[0] for c in cfg["act_layer_combinations"]], cfg["hook_onto_layer"]
    except Exception as e:                       # older checkpoints have no config file
        print("no ao_config.json:", e)
        from transformers import AutoConfig
        n = AutoConfig.from_pretrained(C.MODEL).num_hidden_layers
        return [int(n * p / 100) for p in (25, 50, 75)], 1


def questions(it):
    """(question name, site, text, label) for one item. Label 1 means Yes."""
    v = VERB[it["fam"]]
    neg_opt = it["ord"] if it["neg"] == 0 else 1 - it["ord"]      # option (0 = A) that holds the negated person
    out = []
    for s, lab in (("e1", it["neg"] != 0), ("win1", it["neg"] != 0), ("e2", it["neg"] != 1), ("win2", it["neg"] != 1),
                   ("oA", neg_opt != 0), ("oB", neg_opt != 1)):
        out.append(("pol", s, f"Does this person {v} it?", int(lab)))
    # attachment: the question names one of the two persons (names in Latin script)
    for k, nm in ((0, it["a"]), (1, it["b"])):
        for s in ("ctx", "e2", "end"):
            out.append((f"who{k + 1}", s, f"Does {nm} {v} it?", int(it["neg"] != k)))
    for s in ("cue", "e2", "ctx"):
        out.append(("first", s, "Is the negation about the first person that is mentioned?", int(it["neg"] == 0)))
    for s in ("oB", "end"):
        out.append(("ansA", s, "Is the correct answer option A?", int(it["ans"] == "A")))
    return out


def auc(score, lab):
    score, lab = np.asarray(score), np.asarray(lab)
    p, n = score[lab == 1], score[lab == 0]
    if not len(p) or not len(n): return float("nan")
    return float(((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean()))


def probe(X, y, seed=0):
    """Logistic regression on a random half, accuracy on the other half."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)
    tr, te = idx[: len(X) // 2], idx[len(X) // 2:]
    X = X.float(); y = torch.tensor(y, dtype=torch.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xn[tr] @ w + b, y[tr]) + 1e-2 * (w ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)
    with torch.no_grad():
        return float((((Xn[te] @ w + b) > 0).float() == y[te]).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--mode", default="readout")
    ap.add_argument("--sets", nargs="+", default=[])
    ap.add_argument("--langs", nargs="+", default=[])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--bs", type=int, default=24)
    a = ap.parse_args()
    from peft import PeftModel
    LAYERS, INJECT = oracle_config()
    tok = C.get_tok()
    peft = PeftModel.from_pretrained(C.get_model(), ORACLE).eval()
    net = C.Net(peft.get_base_model())
    A, B = C.letter_ids(tok)
    yes, no = tok.encode("Yes", add_special_tokens=False), tok.encode("No", add_special_tokens=False)
    q = tok.encode(" ?", add_special_tokens=False)
    assert len(yes) == len(no) == len(q) == 1
    yes, no, q = yes[0], no[0], q[0]
    print("oracle", ORACLE, "layers", LAYERS, "inject after layer", INJECT, flush=True)

    def site_pos(roles_of_item, site):
        roles, allt = SITES[site]
        r2 = C.tag_cue(roles_of_item)
        pos = [p for p in range(len(r2)) if roles_of_item[p] in roles or r2[p] in roles]
        return pos if allt else pos[-1:]

    def collect(items):
        """{(layer, site): list of tensors (k, d)} with the adapter off."""
        enc = C.encode(tok, items)
        acts = {(L, s): [None] * len(items) for L in LAYERS for s in SITES}
        with peft.disable_adapter():
            for b in C.batches(len(items), a.bs):
                ids, mask, padn = C.pad(tok, [enc[i][0] for i in b])
                _, st = net.run(ids, mask, read=[("resid", L + 1) for L in LAYERS])
                for x, i in enumerate(b):
                    for s in SITES:
                        pos = site_pos(enc[i][1], s)
                        if not pos: continue
                        for L in LAYERS:
                            acts[(L, s)][i] = st[("resid", L + 1)][x, [p + padn[x] for p in pos]].float().cpu()
        return acts

    def ask(vecs, L, texts, adapter=True):
        """Yes minus No logit of the oracle for each (activation, question)."""
        seqs, poss = [], []
        for v, t in zip(vecs, texts):
            s = C.PRE + f"Layer: {L}\n" + " ?" * len(v) + " \n" + t + C.SUF
            ids = tok.encode(s, add_special_tokens=False)
            p = [i for i, x in enumerate(ids) if x == q][: len(v)]
            assert len(p) == len(v) and p[-1] - p[0] == len(v) - 1, (len(p), len(v))
            seqs.append(ids); poss.append(p)
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        res = [0.0] * len(seqs)
        for b in C.batches(len(order), a.bs):
            idx = [order[i] for i in b]
            ids, mask, padn = C.pad(tok, [seqs[i] for i in idx])
            bi = torch.tensor([x for x, i in enumerate(idx) for _ in poss[i]], device="cuda")
            pp = torch.tensor([p + padn[x] for x, i in enumerate(idx) for p in poss[i]], device="cuda")
            vv = torch.nn.functional.normalize(torch.cat([vecs[i] for i in idx]).cuda(), dim=-1)

            def fn(x):
                x = x.clone()
                o = x[bi, pp]
                x[bi, pp] = o + (vv * o.norm(dim=-1, keepdim=True)).to(x.dtype)
                return x
            if adapter:
                lg, _ = net.run(ids, mask, write={("resid", INJECT + 1): fn})
            else:
                with peft.disable_adapter():
                    lg, _ = net.run(ids, mask, write={("resid", INJECT + 1): fn})
            d = (lg[:, yes] - lg[:, no]).tolist()
            for x, i in enumerate(idx): res[i] = d[x]
        return res

    out = {"oracle": ORACLE, "model": C.MODEL, "layers": LAYERS, "inject": INJECT, "mode": a.mode, "rows": []}

    def emit(r):
        out["rows"].append(r)
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)

    if a.mode == "readout":
        for name in a.sets:
            items = [it for it in C.load(name) if it.get("fam") in (1, 2, 3)]
            random.Random(0).shuffle(items)
            items = items[: a.n]
            acts = collect(items)
            qs = [questions(it) for it in items]
            for k in range(len(qs[0])):
                qn, site = qs[0][k][0], qs[0][k][1]
                lab = [x[k][3] for x in qs]
                texts = [PFX + x[k][2] for x in qs]
                for L in LAYERS:
                    vecs = acts[(L, site)]
                    keep = [i for i, v in enumerate(vecs) if v is not None]
                    y = [lab[i] for i in keep]
                    r = {"set": name, "q": qn, "site": site, "layer": L, "n": len(keep), "yes_rate": float(np.mean(y))}
                    for nm, ad in (("oracle", True), ("noadapter", False)):
                        s = ask([vecs[i] for i in keep], L, [texts[i] for i in keep], ad)
                        r[nm + "_acc"] = float(np.mean([(v > 0) == bool(t) for v, t in zip(s, y)]))
                        r[nm + "_auc"] = auc(s, y)
                        r[nm + "_says_yes"] = float(np.mean([v > 0 for v in s]))
                    r["probe_acc"] = probe(torch.stack([vecs[i][-1] for i in keep]), y)
                    emit(r)
        return

    # interv: patch R with states of its partner P, then ask the oracle about later states of R
    Lmid, Llate = LAYERS[1], LAYERS[2]
    for lang in a.langs:
        with peft.disable_adapter():
            trip, _, _ = build(tok, net, lang, "test", "neg", a.n)
        R = [t[0] for t in trip]
        encR, encD = C.encode(tok, R), C.encode(tok, [t[1] for t in trip])
        for cname, (roles, layer) in INTERV.items():
            ld, vend, voA, voB = [], [], [], []
            with peft.disable_adapter():
                for b in C.batches(len(R), a.bs):
                    idsR, maskR, padR = C.pad(tok, [encR[i][0] for i in b])
                    write = {}
                    if roles:
                        idsD, maskD, padD = C.pad(tok, [encD[i][0] for i in b])
                        _, cache = net.run(idsD, maskD, read=[("resid", layer)])
                        bi, pr, pd = [], [], []
                        for x, i in enumerate(b):
                            p, d = C.align(encR[i][1], encD[i][1], roles)
                            bi += [x] * len(p); pr += [u + padR[x] for u in p]; pd += [v + padD[x] for v in d]
                        t = [torch.tensor(v, device="cuda") for v in (bi, pr, pd)]

                        def fn(x, t=t, src=cache[("resid", layer)]):
                            x = x.clone()
                            x[t[0], t[1]] = src[t[0], t[2]].to(x.dtype)
                            return x
                        write = {("resid", layer): fn}
                    lg, st = net.run(idsR, maskR, read=[("resid", Lmid + 1), ("resid", Llate + 1)], write=write)
                    ld += (lg[:, A] - lg[:, B]).tolist()
                    for x, i in enumerate(b):
                        rl = encR[i][1]
                        vend.append(st[("resid", Llate + 1)][x, [site_pos(rl, "end")[0] + padR[x]]].float().cpu())
                        voA.append(st[("resid", Lmid + 1)][x, [site_pos(rl, "oA")[0] + padR[x]]].float().cpu())
                        voB.append(st[("resid", Lmid + 1)][x, [site_pos(rl, "oB")[0] + padR[x]]].float().cpu())
            ld = np.array(ld)
            sA = np.array(ask(vend, Llate, [PFX + "Is the correct answer option A?"] * len(R)))
            says_A = ld > 0                                             # the answer of the model after the patch
            orig_A = np.array([it["ans"] == "A" for it in R])
            r = {"lang": lang, "interv": cname, "n": len(R), "model_changed": float(np.mean(says_A != orig_A)),
                 "oracle_changed": float(np.mean((sA > 0) != orig_A)),
                 "oracle_agrees_with_model": float(np.mean((sA > 0) == says_A)), "oracle_auc_vs_model": auc(sA, says_A.astype(int))}
            # option names at the middle layer: does the oracle say that the person of the option has the property?
            for nm, vecs, k in (("oA", voA, 0), ("oB", voB, 1)):
                s = np.array(ask(vecs, Lmid, [PFX + f"Does this person {VERB[it['fam']]} it?" for it in R]))
                neg_opt = np.array([it["ord"] if it["neg"] == 0 else 1 - it["ord"] for it in R])
                pos_before = (neg_opt != k).astype(int)                 # label in the item as written
                r[f"{nm}_auc_vs_text"] = auc(s, pos_before)
                r[f"{nm}_yes"] = float(np.mean(s > 0))
            emit(r)


if __name__ == "__main__":
    main()
