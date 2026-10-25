import json, gzip, os, sys, statistics
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

D = os.path.join(os.path.dirname(__file__), "out")
tk = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3-1.7B", "tokenizer.json"))

def chat(m):
    # qwen3 chat format with thinking disabled
    return "<|im_start|>user\n" + m + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

for c in ("A", "B"):
    print("answer", c, "ids", tk.encode(c, add_special_tokens=False).ids)

for L in ("hi", "fr"):
    for sp in ("disc", "test"):
        ln, nt, bad, ex = {}, [], 0, {}
        with gzip.open(os.path.join(D, f"{L}_{sp}.jsonl.gz"), "rt", encoding="utf-8") as r:
            for i, s in enumerate(r):
                if i % 7: continue
                it = json.loads(s)
                p = chat(it["msg"])
                e = tk.encode(p, add_special_tokens=False)
                ln.setdefault(it["fam"], []).append(len(e.ids))
                pre = len("<|im_start|>user\n")
                for k in ("n1", "n2"):
                    a, b = it["sp"][k]
                    t = [j for j, (x, y) in enumerate(e.offsets) if x < b + pre and y > a + pre]
                    if not t: bad += 1
                    nt.append(len(t))
                if it["fam"] not in ex: ex[it["fam"]] = it["msg"]
        for f in sorted(ln):
            v = ln[f]
            print(L, sp, f"family{f}", "n", len(v), "tokens mean", round(statistics.mean(v), 1), "max", max(v))
        print(L, sp, "name tokens mean", round(statistics.mean(nt), 2), "max", max(nt), "unmapped spans", bad)
        if sp == "disc":
            for f in sorted(ex): print("----", L, f"family{f}\n" + ex[f])
