"""Heat maps of patching recovery (layer x role) from the files of e2_patch.

Usage: python -m neg.plot_e2 <out.png> <file.npz> [<file.npz> ...]
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

INK, MUTED, SURF = "#0b0b0b", "#898781", "#fcfcfb"
# diverging: red (negative) - neutral grey - blue (positive)
CMAP = LinearSegmentedColormap.from_list("div", ["#9c2a2a", "#e34948", "#f0efec", "#3987e5", "#0d366b"])


def load(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    s, sR, sP = z["s"], z["sR"], z["sP"]
    rec = (np.nanmean(s, 0) - sR.mean()) / (sP.mean() - sR.mean())
    return rec, meta, len(sR)


def main():
    out, files = sys.argv[1], sys.argv[2:]
    fig, axes = plt.subplots(1, len(files), figsize=(1.2 + 5.6 * len(files), 6.4), squeeze=False, facecolor=SURF)
    for ax, f in zip(axes[0], files):
        rec, meta, n = load(f)
        cols = meta["cols"]
        keep = [i for i in range(len(cols)) if not np.all(np.isnan(rec[:, i]))]
        rec = rec[:, keep]
        if meta["kind"] == "head":
            L = max(u[0] for u in meta["units"]) + 1
            rec = rec.reshape(L, -1, rec.shape[1]).max(1)     # the best head of each layer
            ylab = [str(l) for l in range(L)]
            title = f"{meta['lang']} {meta['mode']}: best head of the layer (n={n})"
        else:
            ylab = [str(u[0]) for u in meta["units"]]
            title = f"{meta['lang']} {meta['mode']} {meta['kind']} (n={n})"
        v = max(1.0, float(np.nanmax(np.abs(rec))))
        im = ax.imshow(rec, aspect="auto", origin="lower", cmap=CMAP, norm=TwoSlopeNorm(0, -v, v))
        ax.set_facecolor(SURF)
        ax.set_xticks(range(len(keep)), [cols[i] for i in keep], rotation=60, ha="right", fontsize=8, color=INK)
        ax.set_yticks(range(0, len(ylab), 2), ylab[::2], fontsize=8, color=INK)
        ax.set_xlabel("role of the patched tokens", color=INK, fontsize=9)
        ax.set_ylabel("layer", color=INK, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        for sp in ax.spines.values(): sp.set_color("#c3c2b7")
        ax.tick_params(color="#c3c2b7")
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("recovery", color=INK, fontsize=9); cb.ax.tick_params(labelsize=8, color="#c3c2b7")
        cb.outline.set_edgecolor("#c3c2b7")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
