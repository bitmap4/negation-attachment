"""Compare float16 on one GPU with float32 divided between two GPUs (the 4B model does not fit one
GPU in float32). Prints the difference of the logit difference A - B on the same items.
Usage: python -m neg.check_half hi_dev fr_dev   (the job must have two GPUs)"""
import sys
import torch
from transformers import AutoModelForCausalLM
from . import common as C

tok = C.get_tok()
A, B = C.letter_ids(tok)
sets = sys.argv[1:]
data = {}
for name in sets:
    its = C.load(name)[::max(1, len(C.load(name)) // 160)][:160]
    data[name] = (its, C.encode(tok, its))


def lds(model, dev):
    out = {}
    for name, (its, enc) in data.items():
        v = []
        for b in C.batches(len(its), 8):
            ids, mask, _ = C.pad(tok, [enc[i][0] for i in b], device=dev)
            with torch.no_grad():
                lg = C.last_logits(model, ids, mask).cpu()
            v += (lg[:, A] - lg[:, B]).tolist()
        out[name] = torch.tensor(v)
    return out


m = AutoModelForCausalLM.from_pretrained(C.MODEL, dtype=torch.float16).to("cuda:0").eval()
h = lds(m, "cuda:0")
del m; torch.cuda.empty_cache()
m = AutoModelForCausalLM.from_pretrained(C.MODEL, dtype=torch.float32, device_map="balanced").eval()
f = lds(m, "cuda:0")
for name in sets:
    d = (h[name] - f[name]).abs()
    print(name, "n", len(d), "mean |ld| float32", f"{f[name].abs().mean():.2f}", "max diff", f"{d.max():.3f}",
          "mean diff", f"{d.mean():.3f}", "sign agreement", f"{((h[name] > 0) == (f[name] > 0)).float().mean():.3f}",
          "finite", bool(torch.isfinite(h[name]).all()))
