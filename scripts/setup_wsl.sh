#!/usr/bin/env bash
# Phase 1 bring-up for the WSL2 + CUDA environment.
#
#   bash scripts/setup_wsl.sh
#
# Idempotent: safe to re-run. Stops at the first genuine failure.
#
# What it does NOT do: build cv_reader (that needs its own repo and is prompted for at the end),
# and it does not touch the Windows-side .venv.
set -euo pipefail

CUDA_CHANNEL="${CUDA_CHANNEL:-cu124}"      # cu124 suits an Ada GPU on a recent driver
VENV_DIR="${VENV_DIR:-.venv-wsl}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[1;33mWARN: %s\033[0m\n' "$1"; }

cd "$REPO_ROOT"

# Pick an interpreter the distro actually ships, newest first but stopping short of 3.13, whose
# native-extension wheel coverage is still patchy for this dependency set. Ubuntu 24.04 has 3.12,
# 22.04 has 3.10; asking for a version the distro lacks fails at apt rather than here.
if [ -z "${PYTHON_BIN:-}" ]; then
    for candidate in python3.12 python3.11 python3.10; do
        if apt-cache show "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
echo "Using interpreter: ${PYTHON_BIN}"

# --------------------------------------------------------------------------------------------
step "System packages"
# ffmpeg + the libav* headers cv_reader links against; default-jre for pycocoevalcap's
# PTBTokenizer and METEOR (without it every caption metric fails at evaluation time).
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake pkg-config git aria2 \
    ffmpeg \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev \
    default-jre \
    "${PYTHON_BIN}" "${PYTHON_BIN}-venv" "${PYTHON_BIN}-dev"

# --------------------------------------------------------------------------------------------
step "GPU visibility"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
    warn "nvidia-smi not found. WSL CUDA needs a recent Windows driver; do NOT install a Linux
      NVIDIA driver inside WSL. Training will fall back to CPU until this resolves."
fi

# --------------------------------------------------------------------------------------------
step "Python environment (${VENV_DIR}, ${PYTHON_BIN})"
[ -d "$VENV_DIR" ] || "$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip wheel setuptools

# --------------------------------------------------------------------------------------------
step "PyTorch (${CUDA_CHANNEL})"
# Must precede requirements-wsl.txt: installing torch from PyPI first pulls the CPU-only build
# and later CUDA installs get skipped as "already satisfied".
python -m pip install --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}" torch torchvision

step "Project requirements"
python -m pip install -r requirements-wsl.txt
python -m pip install -e .

# --------------------------------------------------------------------------------------------
step "Verifying the environment"
python - <<'PY'
import sys

def check(label, fn):
    try:
        print(f"  {label:22s} {fn()}")
        return True
    except Exception as e:
        print(f"  {label:22s} FAIL - {type(e).__name__}: {e}")
        return False

results = []
import torch
results.append(check("torch", lambda: f"{torch.__version__} cuda={torch.cuda.is_available()}"))
if not torch.cuda.is_available():
    print("  !! torch has no CUDA. Re-run with a different CUDA_CHANNEL (cu121/cu126/cu128).")
    results.append(False)

results.append(check("decord", lambda: __import__("decord").__version__))
results.append(check("transformers", lambda: __import__("transformers").__version__))
results.append(check("pycocoevalcap", lambda: "ok" and __import__(
    "pycocoevalcap.tokenizer.ptbtokenizer", fromlist=["PTBTokenizer"]) and "ok"))

try:
    import cv_reader  # noqa: F401
    print("  cv_reader              ok")
except ImportError:
    print("  cv_reader              NOT INSTALLED - see the note below")

sys.exit(0 if all(results) else 1)
PY

# --------------------------------------------------------------------------------------------
step "Java (pycocoevalcap)"
java -version 2>&1 | head -1 || warn "java missing - every caption metric will fail"

# --------------------------------------------------------------------------------------------
step "CLIP weights (baseline run only)"
# Only ViT-B/16 is fetched; see model_zoo/urls.txt. The SigLIP2 + GPT-2 model needs no CLIP.
bash model_zoo/download_model.sh

# --------------------------------------------------------------------------------------------
cat <<'EOF'

============================================================================================
Remaining manual step: build cv_reader (the compressed-domain parser). It is not on PyPI.

    git clone https://github.com/yaojie-shen/Compressed-Video-Reader.git ~/Compressed-Video-Reader
    cd ~/Compressed-Video-Reader
    # follow its README (CMake, links the libav* headers installed above)
    pip install .
    python -c "import cv_reader; print('cv_reader OK')"

If CMake cannot find a compatible FFmpeg, build FFmpeg from source at the version its README
names rather than fighting the distro packages.

Then, in this repo:

    export VATEX_SUBSET_ROOT=/mnt/c/Research-Personal/Datasetextract5000train1000validfromhuggingface
    python tools/verify_motion_channels.py --video_dir "$VATEX_SUBSET_ROOT/train"   # checks motion_channels=2
    python tools/prepare_vatex_subset.py \
        --annotations "$VATEX_SUBSET_ROOT/VATEX_Caption.json" \
        --train_dir   "$VATEX_SUBSET_ROOT/train" \
        --val_dir     "$VATEX_SUBSET_ROOT/val" \
        --output_dir  ./dataset/vatex_subset
    pytest test/ -q
    python tools/train_net.py --config-name=exp/train/vatex_subset_baseline
============================================================================================
EOF
