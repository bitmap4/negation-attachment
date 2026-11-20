"""Plot transfer of the saved base-task subspaces to extended negation sets."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent.parent
forms = [("hab", "Habitual"), ("never", "Never"), ("nolonger", "No longer"),
         ("nothing", "Nothing"), ("lex", "Lexical"), ("focus", "Focus")]
rows = [("c12", "c12", 14, "W", "Clauses L14: W"),
        ("optZ", "opt", 18, "Z", "Options L18: Z"),
        ("ans", "opt", 19, "ANS", "Options L19: answer"),
        ("ans", "end", 27, "ANS", "Final L27: answer")]
fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.25), sharex=True, layout="constrained")
for ax, (lang, title) in zip(axes, [("hi", "Hindi"), ("fr", "French")]):
    values = np.full((len(rows), len(forms)), np.nan)
    for i, (filename, role, layer, var, _) in enumerate(rows):
        path = root / "results" / "e3" / f"forms_{filename}_{lang}.json"
        if not path.exists():
            continue
        for run in json.loads(path.read_text())["runs"]:
            if (run["role"], run["layer"], run["var"]) != (role, layer, var):
                continue
            for j, (form, _) in enumerate(forms):
                item = run["eval"].get(f"{lang}_{form}_test")
                if item:
                    values[i, j] = item["das"]["iia"]
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#ededed")
    ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, "n/a" if np.isnan(value) else f"{value:.2f}",
                ha="center", va="center", fontsize=8.5,
                color="white" if not np.isnan(value) and value > .6 else "#222222")
    ax.set_yticks(range(len(rows)), [row[4] for row in rows], fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=10, pad=3)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
axes[-1].set_xticks(range(len(forms)), [label for _, label in forms], fontsize=8.5)
fig.savefig(root / "report" / "fig" / "e3_forms_readable.pdf", bbox_inches="tight")
