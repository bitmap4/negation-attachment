#!/bin/bash
# Build the environment on the node's /scratch (idempotent). Sets PY and the cache paths.
set -euo pipefail
export SCRATCH_BASE="/scratch/$USER"; mkdir -p "$SCRATCH_BASE"
export HF_HOME="$SCRATCH_BASE/huggingface" UV_CACHE_DIR="$SCRATCH_BASE/uv-cache" TORCH_HOME="$SCRATCH_BASE/torch"
export PATH="$HOME/.local/bin:$PATH" TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
VENV="$SCRATCH_BASE/venvs/negation"; export PY="$VENV/bin/python"
if [ ! -e "$VENV/.ready" ]; then
  T0=$(date +%s)
  [ -x "$PY" ] || uv venv -q --python 3.12 "$VENV"
  export UV_HTTP_TIMEOUT=300
  echo "torch==2.7.1" > "$SCRATCH_BASE/neg-constraints.txt"
  OK=0
  for try in 1 2 3 4 5; do     # the network on the nodes sometimes times out; uv keeps finished downloads
    if uv pip install -q --python "$PY" --index-url https://download.pytorch.org/whl/cu126 "torch==2.7.1" \
       && uv pip install -q --python "$PY" -c "$SCRATCH_BASE/neg-constraints.txt" \
          "transformers>=4.53,<5" "accelerate>=1.0" "peft>=0.15" "numpy>=2,<3" safetensors; then OK=1; break; fi
    echo "install attempt $try failed"; sleep 20
  done
  [ "$OK" = 1 ] || { echo "ENV BUILD FAILED"; exit 4; }
  touch "$VENV/.ready"
  echo "env built in $(( $(date +%s) - T0 )) s"
fi
# Refuse old GPUs: every visible GPU must be an RTX 2080 Ti or newer.
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qiE "1080|TITAN X|980"; then echo "OLD GPU, STOP"; exit 3; fi
"$PY" - <<'PYEOF'
import torch, transformers
assert torch.cuda.is_available(), "CUDA is unavailable"
x = torch.randn(256, 256, device="cuda")
print("torch", torch.__version__, "transformers", transformers.__version__, "gpus", torch.cuda.device_count(),
      torch.cuda.get_device_name(0), "matmul ok", bool(torch.isfinite(x @ x).all()))
PYEOF
