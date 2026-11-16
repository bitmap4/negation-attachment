"""Figures for the report, from the files in results/. Usage: python -m neg.figs [results dir] [output dir]
With NEG_NOTITLE=1 the figures have no title line and a white background (for the paper)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

R = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(R, "fig")
NOTITLE = bool(os.environ.get("NEG_NOTITLE"))
os.makedirs(OUT, exist_ok=True)
INK, INK2, MUTED, SURF, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#ffffff" if NOTITLE else "#fcfcfb", "#e1e0d9", "#c3c2b7"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical slots 1 to 4, fixed order
DIV = LinearSegmentedColormap.from_list("div", ["#9c2a2a", "#e34948", "#f0efec", "#3987e5", "#0d366b"])
LANG = {"hi": "Hindi", "fr": "French"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.facecolor": SURF,
                     "figure.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})


def e2(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    s, sR, sP = z["s"], z["sR"], z["sP"]
    rec = (np.nanmean(s, 0) - sR.mean()) / (sP.mean() - sR.mean())
    return rec, meta["cols"], [u[0] for u in meta["units"]], len(sR)


def sup(fig, *args, **kw):
    if not NOTITLE: fig.suptitle(*args, **kw)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def style(ax, grid=True):
    if grid: ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_heat():
    """Residual patching, layer x role, for the two languages."""
    names = {"ins": "instruction", "n1": "name 1", "body1": "clause 1", "cue": "cue", "e1": "end 1", "conj": "but", "n2": "name 2",
             "body2": "clause 2", "e2": "end 2", "q": "question", "qcue": "question cue", "lA": "A:", "oA": "option A", "lB": "B:",
             "oB": "option B", "tail": "template", "end": "last position", "c1": "clause 1 (all)", "c2": "clause 2 (all)",
             "c12": "both clauses", "ctx": "context", "opt": "options", "qo": "question + options", "post": "all after context"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    for ax, lang in zip(axes, ("hi", "fr")):
        rec, cols, L, n = e2(f"{R}/e2_4b/resid_neg_{lang}.npz")
        keep = [i for i, c in enumerate(cols) if c not in ("ins", "c1", "c2", "qo")]
        cut = sum(1 for i in keep if cols[i] in names and i < cols.index("c1"))
        im = ax.imshow(rec[:, keep], aspect="auto", origin="lower", cmap=DIV, norm=TwoSlopeNorm(0, -1, 1))
        ax.axvline(cut - 0.5, color=SURF, lw=5)
        ax.set_xticks(range(len(keep)), [names[cols[i]] for i in keep], rotation=55, ha="right", fontsize=8, color=INK2)
        ax.set_yticks(range(0, len(L), 4), [str(l) for l in L[::4]])
        ax.set_title(f"{LANG[lang]} (n = {n} pairs)", loc="left", fontsize=10)
        ax.set_xlabel("patched tokens: one role (left), groups of roles (right)")
        ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
    axes[0].set_ylabel("layer")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
    cb.set_label("recovery of the logit difference"); cb.outline.set_visible(False)
    sup(fig, "Where is the attachment? Residual stream patched from the minimal partner (Qwen3-4B)", x=0.125, ha="left", fontsize=11)
    save(fig, "e2_heat.png")


def fig_flow():
    """Recovery by layer for four token groups, for each donor type."""
    series = [("ctx", "context"), ("opt", "options"), ("lB", "B: label"), ("end", "last position")]
    modes = [("neg", "donor: minimal partner"), ("other", "donor: other names and object"),
             ("ord", "donor: option order changed"), ("q", "donor: other question polarity"), ("name", "donor: names exchanged")]
    fig, axes = plt.subplots(2, len(modes), figsize=(16, 5.6), sharex=True, sharey=True)
    for r, lang in enumerate(("hi", "fr")):
        for c, (mode, title) in enumerate(modes):
            ax = axes[r, c]
            p = f"{R}/e2_4b/resid_{mode}_{lang}.npz"
            style(ax)
            if not os.path.exists(p):
                ax.text(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", color=MUTED); continue
            rec, cols, L, n = e2(p)
            for k, (col, lab) in enumerate(series):
                y = rec[:, cols.index(col)]
                ax.plot(L, y, color=S[k], lw=2, label=lab)
            ax.axhline(0, color=AXIS, lw=0.8)
            ax.set_ylim(-0.08, 1.12)
            if r == 0: ax.set_title(title, loc="left", fontsize=9)
            if r == 1: ax.set_xlabel("layer")
            if c == 0: ax.set_ylabel(f"{LANG[lang]}\nrecovery")
    axes[0, -1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    sup(fig, "The attachment moves from the context to the options, then to the last position", x=0.07, ha="left", fontsize=11)
    save(fig, "e2_flow.png")


def das_rows(files):
    rows = []
    for f in files:
        if os.path.exists(f): rows += json.load(open(f))["runs"]
    return rows


def fig_das():
    """Interchange accuracy of each variable at each site."""
    sites = [("c12", l) for l in (10, 12, 14, 16)] + [("opt", l) for l in (18, 19, 20)] + [("end", l) for l in (25, 27, 30)]
    lab = {"c12": "both clauses", "opt": "option names", "end": "last position"}
    vars_ = [("W", "W: negated mention"), ("Z", "Z: option A names the negated person"), ("ANS", "answer letter")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    for ax, lang in zip(axes, ("hi", "fr")):
        rows = das_rows([f"{R}/e3/das_{lang}_a.json", f"{R}/e3/das_{lang}_b.json", f"{R}/e3/das_{lang}.json"])
        style(ax)
        x = np.arange(len(sites))
        for k, (v, vl) in enumerate(vars_):
            y = []
            for role, l in sites:
                m = [r for r in rows if r["role"] == role and r["layer"] == l and r["var"] == v]
                y.append(m[0]["eval"][f"{lang}_test"]["das"]["iia"] if m else np.nan)
            ax.bar(x + (k - 1) * 0.27, y, 0.25, color=S[k], label=vl)
        for b in (3.5, 6.5): ax.axvline(b, color=GRID, lw=1)
        ax.set_xticks(x, [f"L{l}" for _, l in sites])
        for i, name in ((1.5, "c12"), (5, "opt"), (8, "end")):
            ax.text(i, 1.09, lab[name], ha="center", fontsize=8.5, color=INK2)
        ax.set_ylim(0, 1.16)
        ax.set_title(f"{LANG[lang]}: train and test in {LANG[lang]}", loc="left", fontsize=10, pad=14)
    axes[0].set_ylabel("interchange accuracy (answer changes as predicted)")
    axes[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    sup(fig, "DAS, 8 dimensions: the variable at each site", x=0.07, ha="left", fontsize=11, y=1.06)
    save(fig, "e3_das.png")


def fig_das_transfer():
    """Same language, other language, and other-language source, for the best variable of three sites."""
    picks = [("c12", 14, "W", "both clauses L14, W"), ("opt", 18, "Z", "options L18, Z"), ("opt", 19, "ANS", "options L19, answer"), ("end", 27, "ANS", "last position L27, answer")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6), sharey=True)
    for ax, lang in zip(axes, ("hi", "fr")):
        o = "fr" if lang == "hi" else "hi"
        rows = das_rows([f"{R}/e3/das_{lang}_a.json", f"{R}/e3/das_{lang}_b.json", f"{R}/e3/das_{lang}.json"])
        style(ax)
        conds = [(f"{lang}_test", f"test in {LANG[lang]}"), (f"{o}_test", f"test in {LANG[o]}"), (f"{lang}_test<{o}_test", f"base {LANG[lang]}, source {LANG[o]}")]
        x = np.arange(len(picks))
        for k, (e, el) in enumerate(conds):
            y = []
            for role, l, v, _ in picks:
                m = [r for r in rows if r["role"] == role and r["layer"] == l and r["var"] == v]
                y.append(m[0]["eval"][e]["das"]["iia"] if m and e in m[0]["eval"] else np.nan)
            bars = ax.bar(x + (k - 1) * 0.27, y, 0.25, color=S[k], label=el)
            for b, v in zip(bars, y):
                if not np.isnan(v): ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK2)
        ax.set_xticks(x, [p[3] for p in picks], fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_title(f"subspace trained on {LANG[lang]}", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
    axes[0].set_ylabel("interchange accuracy")
    sup(fig, "Do the subspaces transfer between Hindi and French?", x=0.07, ha="left", fontsize=11, y=1.03)
    save(fig, "e3_transfer.png")


def fig_heads():
    stages = [("opt", "to the option tokens (layers 16-19, 128 heads)"), ("lbt", "to B: and template (layers 19-24, 192 heads)"),
              ("end", "to the last position (layers 23-26, 128 heads)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=True)
    for ax, (st, title) in zip(axes, stages):
        style(ax)
        for k, lang in enumerate(("hi", "fr")):
            p = f"{R}/e2_4b/heads_{st}_{lang}.json"
            if not os.path.exists(p): continue
            c = json.load(open(p))["curve"]
            ks = [x["k"] for x in c]
            ax.plot(ks, [x["only"] for x in c], color=S[k], lw=2, marker="o", ms=5, label=f"{LANG[lang]}: only the top k heads")
            ax.plot(ks, [x["except"] for x in c], color=S[k], lw=2, ls=(0, (4, 2)), marker="s", ms=4, label=f"{LANG[lang]}: all heads except the top k")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128], ["1", "2", "4", "8", "16", "32", "64", "128"])
        ax.set_xlabel("k (heads ranked on other pairs)")
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(title, loc="left", fontsize=9)
    axes[0].set_ylabel("recovery")
    axes[2].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    sup(fig, "A few heads move the attachment at each step; one head alone does little", x=0.07, ha="left", fontsize=11, y=1.03)
    save(fig, "e2_heads.png")


def fig_forms():
    """Frozen subspaces from the base task, tested on each negation form."""
    forms = [("hab", "habitual\nne ... pas"), ("never", "never"), ("nolonger", "no longer"), ("nothing", "nothing"), ("lex", "lexical"), ("focus", "focus")]
    rows = [("c12", "c12", 14, "W", "both clauses L14: W"), ("optZ", "opt", 18, "Z", "options L18: Z"),
            ("ans", "opt", 19, "ANS", "options L19: answer"), ("ans", "end", 27, "ANS", "last position L27: answer")]
    seq = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.2), sharey=True)
    for ax, lang in zip(axes, ("hi", "fr")):
        M = np.full((len(rows), len(forms)), np.nan)
        for i, (f, role, layer, var, _) in enumerate(rows):
            p = f"{R}/e3/forms_{f}_{lang}.json"
            if not os.path.exists(p): continue
            for r in json.load(open(p))["runs"]:
                if (r["role"], r["layer"], r["var"]) != (role, layer, var): continue
                for j, (st, _) in enumerate(forms):
                    e = r["eval"].get(f"{lang}_{st}_test")
                    if e: M[i, j] = e["das"]["iia"]
        im = ax.imshow(M, cmap=seq, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, "n/a" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color=MUTED if np.isnan(M[i, j]) else ("#ffffff" if M[i, j] > 0.6 else INK))
        ax.set_xticks(range(len(forms)), [f[1] for f in forms], fontsize=8, color=INK2)
        ax.set_yticks(range(len(rows)), [r[4] for r in rows], fontsize=8.5, color=INK2)
        ax.set_title(f"{LANG[lang]}", loc="left", fontsize=10)
        ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
    sup(fig, "Subspaces from sentential negation, applied to other negation forms (interchange accuracy)", x=0.02, ha="left", fontsize=11, y=1.05)
    save(fig, "e3_forms.png")


def fig_worlds():
    """Explicit worlds, two predicates: effect of a patch of the target fact and of other facts."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, lang in zip(axes, ("hi", "fr")):
        style(ax)
        rec, cols, L, n = e2(f"{R}/e2w/w2_target_{lang}.npz")
        ax.plot(L, rec[:, cols.index("ctx")], color=S[0], lw=2, label="target fact changed: context tokens")
        ax.plot(L, rec[:, cols.index("post")], color=S[1], lw=2, label="target fact changed: tokens after the context")
        for k, (mode, lab) in enumerate((("sib", "same person, other object changed"), ("cross", "other person, same object changed"))):
            p = f"{R}/e2w/w2_{mode}_{lang}.npz"
            if not os.path.exists(p): continue
            r2, c2, L2, _ = e2(p)
            ax.plot(L2, np.nanmax(np.abs(r2), 1), color=S[2 + k], lw=2, label=f"{lab}: largest effect of any group")
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xlabel("layer"); ax.set_title(f"{LANG[lang]} (n = {n})", loc="left", fontsize=10)
    axes[0].set_ylabel("effect on the answer (1 = answer changes)")
    axes[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    sup(fig, "Two persons, two predicates: only the fact in the question controls the answer", x=0.07, ha="left", fontsize=11, y=1.03)
    save(fig, "e2_worlds.png")


def fig_oracle():
    """Oracle readout accuracy against the adapter-off baseline and a linear probe."""
    d = json.load(open(f"{R}/e4/readout.json"))
    picks = [("pol", "win1", 18, "clause 1:\nperson\nlikes it?"), ("who2", "ctx", 18, "context:\n<name 2>\nlikes it?"),
             ("pol", "oA", 18, "option A:\nperson\nlikes it?"), ("first", "ctx", 18, "context:\nnegation on\nperson 1?"),
             ("ansA", "end", 27, "last position\n(layer 27):\nanswer is A?")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), sharey=True)
    for ax, lang in zip(axes, ("hi", "fr")):
        style(ax)
        x = np.arange(len(picks))
        for k, (key, lab) in enumerate((("oracle_acc", "activation oracle"), ("noadapter_acc", "same injection, adapter off"), ("probe_acc", "linear probe (last token)"))):
            y = []
            for q, site, layer, _ in picks:
                m = [r for r in d["rows"] if r["set"] == f"{lang}_test" and r["q"] == q and r["site"] == site and r["layer"] == layer]
                y.append(m[0][key] if m else np.nan)
            ax.bar(x + (k - 1) * 0.27, y, 0.25, color=S[k], label=lab)
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.set_xticks(x, [p[3] for p in picks], fontsize=8, color=INK2)
        ax.set_ylim(0, 1.08); ax.set_title(LANG[lang], loc="left", fontsize=10)
    axes[0].set_ylabel("accuracy (chance = 0.5)")
    axes[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    sup(fig, "What can the activation oracle read? (states from layer 18 unless noted)", x=0.07, ha="left", fontsize=11, y=1.03)
    save(fig, "e4_oracle.png")


for f in (fig_heat, fig_flow, fig_das, fig_das_transfer, fig_heads, fig_forms, fig_worlds, fig_oracle):
    try:
        f()
    except Exception as e:      # a missing result file must not stop the other figures
        print("skipped", f.__name__, repr(e))
