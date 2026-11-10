"""Print DAS results. Usage: python -m neg.show_e3 <file.json> [<file.json> ...]"""
import json, sys

for f in sys.argv[1:]:
    d = json.load(open(f))
    ev = d["args"]["eval"]
    print(f"== {f}  train {d['args']['train']} k={d['args']['k']}   cells: das / full vector / random / same-value / control unchanged")
    print(f"{'site':>10} {'var':>4} " + " ".join(f"{e:>30}" for e in ev))
    for r in d["runs"]:
        cells = []
        for e in ev:
            x = r["eval"].get(e)
            if not x: cells.append(" " * 30); continue
            c = x.get("ctrl_das", {}).get("iia", float("nan"))
            cells.append(f"{x['das']['iia']:.2f} /{x['full']['iia']:.2f} /{x['rand']['iia']:.2f} /{x['das_same']['iia']:.2f} /{c:.2f}".rjust(30))
        print(f"{r['role'] + ':' + str(r['layer']):>10} {r['var']:>4} " + " ".join(cells))
