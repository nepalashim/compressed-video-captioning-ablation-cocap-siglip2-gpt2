# Running on a rented GPU

Everything in this repo runs at the paper's full settings (`num_gop: 8`, `num_mv`/`num_res: 59`)
— which needs **more than the 8 GB an RTX 4070 Laptop has**. On that card VRAM sits at
7710/8188 MiB and PyTorch's allocator thrashes: `nvidia-smi` shows 100% utilisation at 29 W of
an 89 W budget, meaning the GPU is stalling rather than computing.

This guide covers renting a bigger card. It assumes SSH access to an Ubuntu box.

---

## 1. Before you rent — what to check

The GPU is not the only thing that matters here. **Every training sample decodes a whole video
on the CPU** through `cv_reader`, so an instance with a fast GPU and few vCPUs will sit idle
while the CPU works.

| Check | Want | Why |
|---|---|---|
| **VRAM** | **≥ 24 GB** | 8 GB thrashes at these settings. 24 GB fits comfortably. |
| **Architecture** | **Ampere or newer** (RTX 3090/4090, A100, L40S, A6000) | The configs use `precision: bf16-mixed`. **V100 is Volta and has no bf16** — it would force fp16, which needs loss scaling and can destabilise training. Avoid V100 even though it has 32 GB. |
| **vCPUs** | **≥ 16** | The dataloader is CPU-bound on video decoding. See §5. |
| **Disk** | **≥ 40 GB** | ~3 GB dataset, ~15 GB for the environment and FFmpeg build, the rest for checkpoints. |
| **CUDA** | 12.x | Matches the `cu124` torch wheels. |

At **$0.30/hr** you are looking at community-cloud RTX 3090 or 4090 — both are 24 GB, both
support bf16, both are a good fit.

---

## 2. Rent one hour first, and measure

**Do not commit to a long booking before measuring.** The per-epoch time on this workload is
hard to predict because it is bound by memory bandwidth and video decoding, not raw FLOPs — the
same reason the 4070 behaved so badly. One hour at $0.30 buys you the number you need.

Plan: spend the first hour on setup (§3–4), start training, read the `it/s` at ~300 iterations,
then decide whether to book the rest.

---

## 3. Setup on the instance

`scripts/setup_wsl.sh` works on any Ubuntu box, not just WSL. It installs the system packages,
creates a venv on the newest available Python, installs **torch from the CUDA wheel index first**
(installing it from PyPI first silently pins the CPU-only build), then the project.

```bash
sudo apt update && sudo apt install -y git
git clone <your-repo-url> ~/CoCap        # or scp the repo up
cd ~/CoCap
git checkout siglip2-gpt2

bash scripts/setup_wsl.sh
source .venv-wsl/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Build `cv_reader` — with both patches

This is the slow part (~20–50 min) and it needs two patches on any modern Ubuntu. Both are
explained in [runbook.md](runbook.md) §6; the short version:

```bash
git clone https://github.com/yaojie-shen/Compressed-Video-Reader.git ~/Compressed-Video-Reader
cd ~/Compressed-Video-Reader

# cap parallelism - the script's bare `make -j` can exhaust RAM
sed -i 's/^make -j || exit 1/make -j$(nproc) || exit 1/' ffmpeg/install_ffmpeg.sh

# Patch B: NumPy 2 tightened PyArray_GETPTR3 to require PyArrayObject*
python - <<'PY'
import pathlib
p = pathlib.Path("src/cv_reader/api.cpp")
s = p.read_text()
old, new = "PyArray_GETPTR3(mv_arr,", "PyArray_GETPTR3((PyArrayObject *) mv_arr,"
print(f"replaced {s.count(old)} occurrences")   # expect 4
p.write_text(s.replace(old, new))
PY

bash install.sh 2>&1 | tee ~/cv_reader_install.log   # will fail at the FFmpeg compile

# Patch A: FFmpeg 5.1 does not assemble under binutils >= 2.41
cd ~/Compressed-Video-Reader/ffmpeg/ffmpeg_source
sed -i 's/"ic" ((uint8_t)(-s))/"ic" ((uint8_t)(-s \& 0x1F))/g' libavcodec/x86/mathops.h
grep -c '0x1F' libavcodec/x86/mathops.h          # expect 2
make -j$(nproc) && make install

cd ~/Compressed-Video-Reader && pip install .
python -c "import cv_reader; print('cv_reader OK')"
```

---

## 4. Get the dataset there

2.6 GB, ~6,000 files. Two options — **re-downloading is usually much faster than uploading**,
because home upload bandwidth is the bottleneck.

**Option A — re-download on the instance** (preferred): pull the same VATEX subset from
HuggingFace directly. Datacentre bandwidth will beat your uplink by an order of magnitude.

**Option B — upload from your machine:**
```bash
rsync -avz --progress \
    ~/data/vatex_subset/ \
    user@host:~/data/vatex_subset/
```
`rsync` resumes on interruption, which `scp` does not — worth it for 2.6 GB.

Then, on the instance:
```bash
echo 'export VATEX_SUBSET_ROOT=$HOME/data/vatex_subset' >> ~/.bashrc
source ~/.bashrc

cd ~/CoCap
python tools/verify_motion_channels.py --video_dir "$VATEX_SUBSET_ROOT/train" -n 20
python tools/prepare_vatex_subset.py \
    --annotations "$VATEX_SUBSET_ROOT/VATEX_Caption.json" \
    --train_dir   "$VATEX_SUBSET_ROOT/train" \
    --val_dir     "$VATEX_SUBSET_ROOT/val" \
    --output_dir  ./dataset/vatex_subset
python tools/validate_data_pipeline.py --variant baseline -n 200 --build-model
```

The validator must report **`reader failures (all-zero I-frames) : 0`**. Anything higher means
clips are training as black video.

---

## 5. Tune for the instance before the long run

Two settings depend on the hardware, and both are worth a minute of attention because they
multiply across every run you pay for.

**`num_workers`** — the dataloader decodes one video per sample, measured at **~1.3 s per video
per worker**. With an epoch of 49,990 samples:

| workers | data-only time per epoch |
|---|---|
| 8 | ~2.3 h |
| 16 | ~1.1 h |
| 32 | ~0.6 h |

Set it to roughly your vCPU count. If `nvidia-smi` shows the GPU idling, this is why.

**`batch_size`** — the configs use 2 with `accumulate_grad_batches: 6` for an effective batch of
12. On 24 GB you can raise the real batch and lower accumulation, keeping the **product at 12**:

```bash
python tools/train_net.py --config-name=exp/train/vatex_subset_baseline \
    train_dataloader.batch_size=6 trainer.accumulate_grad_batches=2 \
    train_dataloader.num_workers=16 val_dataloader.num_workers=16
```

Larger real batches use the GPU far more efficiently on this workload. Push `batch_size` up
until VRAM is around 80% — leave headroom, since sitting at 95% is what caused the thrashing on
the 4070.

---

## 6. Run it so an SSH drop cannot kill it

**Always run training inside `tmux`.** Closing your laptop or losing wifi otherwise kills the
process and everything you have paid for since the last checkpoint.

```bash
tmux new -s train
# inside tmux:
cd ~/CoCap && source .venv-wsl/bin/activate
python tools/train_net.py --config-name=exp/train/vatex_subset_baseline

# detach: Ctrl+B then D
# reattach later, from a fresh SSH session:
tmux attach -t train
```

Checkpoints land in `logs/vatex_subset_baseline/` every epoch (`save_top_k: -1` keeps them all).
To resume after a preemption:

```bash
python tools/train_net.py --config-name=exp/train/vatex_subset_baseline \
    +trainer.ckpt_path=logs/vatex_subset_baseline/checkpoints/last.ckpt
```

**On spot/interruptible instances** (which is what $0.30/hr usually buys) assume you *will* be
preempted. Checkpoint resume is not optional there.

Monitor from a second SSH session:
```bash
watch -n 2 nvidia-smi                      # want VRAM ~80%, power near cap
tail -f logs/vatex_subset_baseline/*.log
tensorboard --logdir logs/ --host 0.0.0.0  # then SSH-forward port 6006
```

---

## 7. Cost

Epoch is **24,995 iterations** at batch 2 (49,990 samples). The honest position is that
per-epoch time must be **measured**, not predicted — this workload is bound by memory bandwidth
and video decoding, so FLOPs comparisons mislead. Read the `it/s` at ~300 iterations and find
your row:

| observed it/s | h/epoch | 10 epochs | 20 epochs | 30 epochs |
|---|---|---|---|---|
| 1.0 | 6.9 | 69 h | 139 h | 208 h |
| 2.0 | 3.5 | 35 h | 69 h | 104 h |
| 3.5 | 2.0 | 20 h | 40 h | 60 h |
| 5.0 | 1.4 | 14 h | 28 h | 42 h |
| 8.0 | 0.9 | 9 h | 17 h | 26 h |

**At $0.30/hr, for one run:**

| h/epoch | 10 epochs | 20 epochs | 30 epochs |
|---|---|---|---|
| 1 | $3 | $6 | $9 |
| 2 | $6 | $12 | $18 |
| 3 | $9 | $18 | $27 |
| 4 | $12 | $24 | $36 |

**But you need more than one run.** The full experiment matrix is three variants (baseline,
siglip, siglip_gpt2) × 2 seeds = **6 runs**, because 4,999 training videos is small enough that
single-run differences sit inside the noise:

| scenario | GPU-hours | cost @ $0.30/hr |
|---|---|---|
| 1 run, 20 epochs @ 2 h/epoch | 40 h | **$12** |
| 4 runs (2 variants × 2 seeds), 20 epochs @ 2 h/epoch | 160 h | **$48** |
| 6 runs (3 variants × 2 seeds), 20 epochs @ 2 h/epoch | 240 h | **$72** |
| 6 runs, 30 epochs @ 3 h/epoch | 540 h | **$162** |

Three ways to cut that, none of which weaken the paper:

1. **Fewer epochs.** 30 on 4,999 videos is likely more than needed — watch validation CIDEr and
   stop when it plateaus. This is usually the biggest saving.
2. **Drop the `siglip` variant.** It is the encoder-only ablation. Useful for attributing the
   gain, but not required to claim one. Takes 6 runs to 4.
3. **`unfold_sentences: False`** — cuts the epoch from 49,990 samples to 4,999 (one pass per
   video with a randomly chosen caption) — a straight **10×** on decoding. You would raise the
   epoch count to compensate, but total video decodes still drop sharply.

---

## 8. Get results off before you destroy the instance

Checkpoints are large; the results usually are not.

```bash
# from your machine
rsync -avz user@host:~/CoCap/logs/ ./logs-remote/

# just the metrics and predictions, if bandwidth is tight
rsync -avz --include='*/' --include='*.json' --include='events.out.*' --exclude='*' \
    user@host:~/CoCap/logs/ ./logs-remote/
```

### Shutdown checklist

- [ ] `logs/` copied back, and the caption prediction JSONs verified non-empty
- [ ] final metrics recorded in `RESULTS.md` with the exact config and commit hash
- [ ] any checkpoint you want to keep downloaded (they are ~700 MB each and there is one per epoch)
- [ ] **instance destroyed** — hourly billing continues while it is merely stopped on some
      providers

---

## 9. What to bring back to the discussion

- the measured `it/s` and h/epoch
- `nvidia-smi` at steady state: VRAM used and power draw
- validation CIDEr per epoch, so we can see where it plateaus
- `python tools/benchmark_latency.py --variant <v> --device cuda -n 50` for all three variants —
  those are the inference-speed numbers the paper's central claim rests on, and they should be
  measured on the same hardware as the accuracy numbers
