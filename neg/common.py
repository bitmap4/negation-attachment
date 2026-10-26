"""Shared code: data, prompt format, role labels for tokens, model and batches."""
import gzip, json, os
import torch

MODEL = os.environ.get("NEG_MODEL", "Qwen/Qwen3-1.7B")
DATA = os.environ.get("NEG_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))
PRE = "<|im_start|>user\n"
# Qwen3 chat format with thinking disabled
SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


SHOTS = int(os.environ.get("NEG_SHOTS", "0"))     # number of fixed examples before the item (0 to 4)
_shots = {}


def prefix(lang):
    """Text before the user message: the fixed examples as earlier turns, then the start of the turn."""
    if lang not in _shots:
        s = ""
        if SHOTS:
            ex = json.load(open(os.path.join(DATA, "shots.json"), encoding="utf-8"))[lang][:SHOTS]
            s = "".join(f"{PRE}{e['msg']}<|im_end|>\n<|im_start|>assistant\n{e['ans']}<|im_end|>\n" for e in ex)
        _shots[lang] = s + PRE
    return _shots[lang]


def chat(msg, lang):
    return prefix(lang) + msg + SUF


def load(name, n=None, keep=None):
    """Read data/<name>.jsonl.gz. keep is a filter on items."""
    out = []
    with gzip.open(os.path.join(DATA, name + ".jsonl.gz"), "rt", encoding="utf-8") as r:
        for s in r:
            it = json.loads(s)
            if keep is None or keep(it):
                out.append(it)
                if n and len(out) >= n:
                    break
    return out


def role_spans(it):
    """Character spans (role, start, end) that cover the user message, in order."""
    if "roles" in it:
        return [tuple(x) for x in it["roles"]]
    m, s = it["msg"], it["sp"]
    ctx_end = s["e2"][1] + 1                      # newline after the context goes with e2
    q_end = m.index("\n", ctx_end) + 1            # question line and its newline
    base = [("ins", 0, s["n1"][0]), ("n1", *s["n1"]), ("body1", s["n1"][1], s["e1"][0]),
            ("e1", *s["e1"]), ("conj", s["e1"][1], s["n2"][0]), ("n2", *s["n2"]),
            ("body2", s["n2"][1], s["e2"][0]), ("e2", s["e2"][0], ctx_end),
            ("q", ctx_end, q_end), ("lA", q_end, s["oA"][0]), ("oA", *s["oA"]),
            ("lB", s["oA"][1], s["oB"][0]), ("oB", *s["oB"])]
    over = [(k, *s[k]) for k in ("cue", "qcue") if k in s]
    return base, over


def token_roles(it, offsets):
    """One role for each token of chat(msg). A token gets the role of its last character."""
    m = it["msg"]
    sp = role_spans(it)
    base, over = sp if isinstance(sp, tuple) else (sp, [])
    ch = [None] * len(m)
    for k, a, b in list(base) + list(over):
        for i in range(a, b):
            ch[i] = k
    assert all(c is not None for c in ch), it["id"]
    p = len(prefix(it["lang"]))
    out = []
    for a, b in offsets:
        e = b - 1 - p
        out.append("pre" if e < 0 else ch[e] if e < len(m) else "tail")
    out[-1] = "end"
    return out


def get_tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    return tok


DTYPES = {"float32": torch.float32, "half": torch.float16, "bf16": torch.bfloat16}


def get_model(dtype=None, eager=False):
    """eager: use the attention code that can return the attention weights."""
    dtype = dtype or DTYPES[os.environ.get("NEG_DTYPE", "float32")]
    from transformers import AutoModelForCausalLM
    kw = {"attn_implementation": "eager"} if eager else {}
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, **kw)
    model = model.to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def encode(tok, items):
    """Token ids and token roles for each item."""
    if not items: return []
    enc = tok([chat(it["msg"], it["lang"]) for it in items], add_special_tokens=False, return_offsets_mapping=True)
    return [(ids, token_roles(it, off)) for it, ids, off in zip(items, enc["input_ids"], enc["offset_mapping"])]


def pad(tok, seqs, device="cuda"):
    """Left-padded batch. Returns ids, mask, and the pad length of each row."""
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s):] = torch.tensor(s)
        mask[i, n - len(s):] = 1
    return ids.to(device), mask.to(device), [n - len(s) for s in seqs]


def last_logits(model, ids, mask):
    """Logits at the last position only (logits for all positions do not fit in memory)."""
    h = model.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, -1]
    return model.lm_head(h).float()


def letter_ids(tok):
    a, b = tok.encode("A", add_special_tokens=False), tok.encode("B", add_special_tokens=False)
    assert len(a) == 1 and len(b) == 1
    return a[0], b[0]


def batches(n, bs):
    for i in range(0, n, bs):
        yield range(i, min(n, i + bs))


# ---------- hooks ----------

class Net:
    """Read and write activations of the model.

    Sites: ("resid", l) is the input of layer l, and l = n_layers is the input of the final norm;
    ("attn", l) and ("mlp", l) are the sublayer outputs; ("attn_in", l) is what the attention of layer l
    reads (a patch here changes only this attention, not the residual stream); ("z", l) is the input of o_proj
    (the head outputs side by side, n_heads x d_head)."""

    def __init__(self, model):
        self.m = model
        self.layers = model.model.layers
        self.L = len(self.layers)
        self.H = model.config.num_attention_heads
        self.dh = self.layers[0].self_attn.o_proj.in_features // self.H     # not hidden_size / heads in Qwen3-4B

    def _mod(self, kind, l):
        if kind == "resid": return (self.layers[l] if l < self.L else self.m.model.norm), True
        if kind == "attn": return self.layers[l].self_attn, False
        if kind == "attn_in": return self.layers[l].self_attn, True      # input of the attention (after the norm)
        if kind == "mlp": return self.layers[l].mlp, False
        if kind == "z": return self.layers[l].self_attn.o_proj, True
        raise ValueError(kind)

    def hook(self, site, fn):
        """fn(x) returns the new activation, or None to leave it."""
        mod, pre = self._mod(*site)
        if pre:
            def h(m, args, kwargs):
                if args:
                    y = fn(args[0])
                    return None if y is None else ((y,) + tuple(args[1:]), kwargs)
                y = fn(kwargs["hidden_states"])
                return None if y is None else (args, dict(kwargs, hidden_states=y))
            return mod.register_forward_pre_hook(h, with_kwargs=True)

        def h(m, args, out):
            x = out[0] if isinstance(out, tuple) else out
            y = fn(x)
            if y is None: return None
            return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y
        return mod.register_forward_hook(h)

    def run(self, ids, mask, read=(), write=None, grad=False):
        """Forward pass. read: sites to store. write: {site: fn}. Returns last-position logits and the store."""
        store, hs = {}, []
        for s in read:
            hs.append(self.hook(s, lambda x, s=s: store.__setitem__(s, x.detach()) if not grad else store.__setitem__(s, x)))
        for s, fn in (write or {}).items():
            hs.append(self.hook(s, fn))
        try:
            with torch.set_grad_enabled(grad):
                lg = last_logits(self.m, ids, mask)
        finally:
            for h in hs: h.remove()
        return lg, store


def tag_cue(roles):
    """Rename the cue to cue1 or cue2 (the number of its clause)."""
    out, c = [], 1
    for r in roles:
        if r in ("conj", "n2"): c = 2
        out.append(r + str(c) if r == "cue" else r)
    return out


def align(rr, rd, group):
    """Token positions of the roles in `group` in the recipient (rr) and the donor (rd).
    For each role the last tokens are aligned and the shorter count is used."""
    pr, pd = [], []
    for g in group:
        a = [i for i, r in enumerate(rr) if r == g]
        b = [i for i, r in enumerate(rd) if r == g]
        k = min(len(a), len(b))
        if k:
            pr += a[len(a) - k:]; pd += b[len(b) - k:]
    return pr, pd
