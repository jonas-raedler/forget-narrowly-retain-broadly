#!/usr/bin/env bash
# Bootstrap the SUITE proximity-relearning fork on a fresh RunPod pod.
#
# pip-venv port of the upstream conda setup.sh (same pinned versions), for the
# runpod/pytorch:*-cu1281-* images (Python 3.12, driver >= CUDA 12.8). The
# venv installs its own torch 2.9.1 cu128 wheel; the image only supplies the
# driver.
#
# Usage on the pod:
#   export HF_TOKEN=hf_xxx          # meta-llama/Llama-3.2-3B-Instruct is gated
#   export GITHUB_TOKEN=ghp_xxx     # push access to the fork (results publish)
#   bash runpod/setup_pod.sh
#
# Re-running is safe: pulls latest, reuses venv + HF cache.
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
PROJ="$WORKDIR/forget-narrowly-retain-broadly"
REPO_SLUG="jonas-raedler/forget-narrowly-retain-broadly"

export HF_HOME="${HF_HOME:-$WORKDIR/.cache/huggingface}"
mkdir -p "$HF_HOME"

# --- clone or update (token passed per-command, never persisted) ---
CLEAN_URL="https://github.com/${REPO_SLUG}.git"
GIT_AUTH=()
if [ -n "${GITHUB_TOKEN:-}" ]; then
  B64=$(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 | tr -d '\n')
  GIT_AUTH=(-c "http.extraHeader=Authorization: Basic ${B64}")
fi
if [ -d "$PROJ/.git" ]; then
  echo "==> Updating existing checkout"
  git "${GIT_AUTH[@]}" -C "$PROJ" pull --ff-only
else
  echo "==> Cloning $REPO_SLUG"
  git "${GIT_AUTH[@]}" clone "$CLEAN_URL" "$PROJ"
fi
cd "$PROJ"

# --- venv (upstream uses conda; pip works with the same pins) ---
if [ ! -d ".venv" ]; then
  echo "==> Creating venv (Python: $(python3 --version))"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

echo "==> Installing pinned stack (torch 2.9.1 cu128 wheel ~3GB, one-time per volume)"
pip install "numpy<=2"
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.3.0
pip install datasets accelerate peft deepspeed bitsandbytes \
            tqdm rich pydantic omegaconf nltk \
            hydra-core==1.3.2 hydra-colorlog==1.2.0 tensorboard \
            matplotlib pandas scipy psutil ninja packaging

# --- CUDA extensions ---
# flash-attn: required by the model configs (attn_implementation flash_attention_2).
# causal-conv1d + flash-linear-attention: required by the Qwen3.5 JUDGE (hybrid
# linear attention) — do NOT skip them even though Llama doesn't need them.
# Try prebuilt wheels first; fall back to source build (upstream's FORCE_BUILD
# path) only if the prebuilt wheel is missing/ABI-incompatible.
MAX_JOBS="${MAX_JOBS:-4}"
pip install flash-attn==2.8.3 --no-build-isolation \
  || FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="$MAX_JOBS" pip install flash-attn==2.8.3 --no-build-isolation
pip install causal-conv1d==1.6.0 --no-build-isolation \
  || CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="$MAX_JOBS" pip install causal-conv1d==1.6.0 --no-build-isolation
pip install flash-linear-attention==0.4.1

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN not set — gated Llama downloads will fail." >&2
fi

# --- verify BEFORE spending GPU-hours ---
echo "==> Sanity: torch sees the GPU?"
python -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
echo "==> Sanity: imports"
python -c "import transformers, datasets, hydra, peft, flash_attn; print('transformers', transformers.__version__)"
echo "==> Sanity: band configs vs local dataset copy"
python scripts/verify_band_configs.py

cat <<EOF

==> Setup complete.

    M1a — pretrained baseline eval (needed as reference row):
      TOPIC=challenger_disaster MODELS="meta-llama/Llama-3.2-3B-Instruct" TRAIN_GPUS=0 JUDGE_N_GPUS=1 \\
        bash scripts/suite_evaluation_optimized.sh
    M1b — unlearn (JensUn++, paper defaults):
      METHOD=JensUnPP MODEL=llama_3b TOPIC=challenger_disaster TRAIN_GPUS=0 JUDGE_N_GPUS=1 \\
        bash scripts/suite_unlearn.sh
    M1c — C-Full reproduction + M2 tier sweep (NB: JensUnPP checkpoints land
    under the method folder "jensen" — see scripts/trainer_method_map.txt):
      MODEL_PATH="./saves/unlearn/challenger_disaster/Llama-3.2-3B-Instruct/jensen/<exp>" \\
        TRAIN_GPUS=0 JUDGE_N_GPUS=1 bash scripts/proximity_relearn_sweep.sh

    Publish result artifacts (committed to fork main):
      bash runpod/publish_results.sh
EOF
