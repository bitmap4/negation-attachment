"""Plot partner-patch recovery by layer from the saved E2 results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent.parent
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), sharey=True, layout="constrained")
series = [("c12", "Both clauses", "#0072B2", "-"),
          ("ctx", "Context + cue", "#009E73", "--"),
          ("opt", "Options", "#D55E00", "-"),
          ("lB", "Label", "#E69F00", ":"),
          ("end", "Final position", "#CC79A7", "-")]
for ax, (lang, title) in zip(axes, [("hi", "Hindi"), ("fr", "French")]):
    with np.load(root / "results" / "e2_4b" / f"resid_neg_{lang}.npz", allow_pickle=True) as data:
        meta = json.loads(str(data["meta"]))
        scores = (np.nanmean(data["s"], axis=0) - data["sR"].mean()) / (data["sP"].mean() - data["sR"].mean())
        layers = [u[0] for u in meta["units"]]
        for key, label, color, linestyle in series:
            ax.plot(layers, scores[:, meta["cols"].index(key)], label=label,
                    color=color, linestyle=linestyle, linewidth=1.6)
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel("Layer", fontsize=9)
    ax.set_xticks([0, 10, 20, 30, 36])
    ax.axhline(0, color="#777777", linewidth=.5)
    ax.grid(axis="y", color="#dddddd", linewidth=.5)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Logit-difference recovery", fontsize=9)
axes[0].set_ylim(-.25, 1.25)
axes[1].legend(fontsize=7.6, loc="center right", frameon=False, ncol=1)
fig.savefig(root / "report" / "fig" / "e2_route.pdf", bbox_inches="tight")
