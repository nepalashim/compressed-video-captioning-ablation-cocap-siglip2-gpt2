# Runbook — from here to a trained baseline

Follow these in order. Each step has a **check** — do not move on until it passes, because a
failure here is much cheaper to fix than the same failure discovered thirty epochs into a run.

Estimated effort: steps 1–4 are about an hour of mostly waiting. Step 5 (`cv_reader`) is the one
that can eat a day. Steps 6–9 are minutes. Step 10 is the first real training run.

---

## 1. Install an Ubuntu distro

`wsl --list --verbose` currently shows only `docker-desktop`, which is Docker's internal distro
and cannot be used for this. In **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot if prompted, then launch Ubuntu from the Start menu and set a username and password.

**Check** — inside Ubuntu:
```bash
lsb_release -a          # Ubuntu 24.04
```

---

## 2. Confirm the GPU is visible

Do this *before* investing time in the build. WSL CUDA works through the Windows driver; you
must **not** install a Linux NVIDIA driver inside WSL.

```bash
nvidia-smi
```

**Check** — the table lists `NVIDIA GeForce RTX 4070 Laptop GPU`. If it does not, stop and fix
this first: update the Windows NVIDIA driver, then `wsl --shutdown` in PowerShell and relaunch.

---

## 3. Get the code onto the Linux filesystem

All work is committed to the `siglip2-gpt2` branch, so you can clone straight from the Windows
checkout. Work on `~`, not `/mnt/c` — native builds and many-small-file I/O are far slower over
the Windows bridge.

```bash
git clone -b siglip2-gpt2 /mnt/c/Research-Personal/CoCap ~/CoCap
cd ~/CoCap
git log --oneline -3
```

**Check** — you see `Validate the data path without cv_reader...` at the top.

---

## 4. Copy the dataset onto the Linux filesystem

2.6 GB. Reading it over `/mnt/c` during training is slow enough to become the bottleneck.

```bash
mkdir -p ~/data
cp -r /mnt/c/Research-Personal/Datasetextract5000train1000validfromhuggingface ~/data/vatex_subset
echo 'export VATEX_SUBSET_ROOT=$HOME/data/vatex_subset' >> ~/.bashrc
source ~/.bashrc
```

**Check**
```bash
ls "$VATEX_SUBSET_ROOT/train" | wc -l    # 5000
ls "$VATEX_SUBSET_ROOT/val"   | wc -l    # 1000
```

---

## 5. Environment

```bash
cd ~/CoCap
bash scripts/setup_wsl.sh
```

This installs the system packages (ffmpeg, the libav* headers `cv_reader` links against, a JRE
for the caption metrics), creates `.venv-wsl` on Python 3.11, installs **torch from the CUDA
wheel index first** (installing it from PyPI first would silently pin the CPU-only build), then
the rest of `requirements-wsl.txt`, then fetches CLIP ViT-B/16 for the baseline run.

Activate it in every new shell:
```bash
source ~/CoCap/.venv-wsl/bin/activate
```

**Check** — the script's verification block prints `torch ... cuda=True`. If it prints
`cuda=False`, re-run with a different channel, e.g. `CUDA_CHANNEL=cu126 bash scripts/setup_wsl.sh`.

---

## 6. Build `cv_reader` — the hard step

This is the native H.264 parser. It is not on PyPI and it is the only remaining unknown in the
whole pipeline.

Its `install.sh` builds a patched FFmpeg 5.1 into `ffmpeg/ffmpeg_install` — not system-wide, so
your distro FFmpeg is untouched — then runs `pip3 install .`.

**Two patches are required on Ubuntu 24.04 / Python 3.12** (verified: without them the build
fails twice over). Both are age-related incompatibilities upstream, not misconfiguration.

| | Symptom | Cause |
|---|---|---|
| **A** | `mathops.h:125: Error: operand type mismatch for 'shr'` | FFmpeg 5.1 passes an unclipped shift constant to inline asm; binutils ≥ 2.41 rejects it. Fixed upstream in FFmpeg commit `effadce6c7` by masking with `& 0x1F`; 5.1 predates it. |
| **B** | `cannot convert '_object*' to 'const PyArrayObject*'` | NumPy 2 tightened `PyArray_GETPTR3` to require `PyArrayObject *`, but `api.cpp` declares `mv_arr` as `PyObject *`. |

Also cap the build parallelism — the script's `make -j` has **no job limit** and can spawn
enough compilers to exhaust RAM and freeze the machine.

```bash
git clone https://github.com/yaojie-shen/Compressed-Video-Reader.git ~/Compressed-Video-Reader
cd ~/Compressed-Video-Reader

# cap parallelism
sed -i 's/^make -j || exit 1/make -j$(nproc) || exit 1/' ffmpeg/install_ffmpeg.sh

# Patch B - use python, not sed; the nested parens are easy to mangle
python - <<'PY'
import pathlib
p = pathlib.Path("src/cv_reader/api.cpp")
s = p.read_text()
old, new = "PyArray_GETPTR3(mv_arr,", "PyArray_GETPTR3((PyArrayObject *) mv_arr,"
print(f"replaced {s.count(old)} occurrences")   # expect 4
p.write_text(s.replace(old, new))
PY

bash install.sh 2>&1 | tee ~/cv_reader_install.log
```

`install.sh` downloads FFmpeg fresh, so **Patch A can only be applied after that download**. Let
the run above fail at the FFmpeg compile, then patch and resume by hand — `configure` has
already run by that point, so only `make` is needed:

```bash
cd ~/Compressed-Video-Reader/ffmpeg/ffmpeg_source
sed -i 's/"ic" ((uint8_t)(-s))/"ic" ((uint8_t)(-s \& 0x1F))/g' libavcodec/x86/mathops.h
grep -c '0x1F' libavcodec/x86/mathops.h          # expect 2
make -j$(nproc) && make install

cd ~/Compressed-Video-Reader
pip install .
```

Do **not** re-run `install.sh` after patching — it re-downloads FFmpeg and discards Patch A.

**Timebox this.** If it is not building after a few hours, stop and tell me; the fallback is to
port `read_frames_compressed_domain` onto a Python motion-vector extractor, which is real work
but bounded, and the rest of the pipeline is already validated against a stub of exactly that
interface.

**Check** — the Python import is the one that matters; the CLI can work while the module landed
outside the venv.
```bash
python -c "import cv_reader; print('cv_reader OK')"
cv_reader ./test_data/h264_sample.mp4 ./test_output    # parses a real H.264 file
```

---

## 7. Verify the motion-channel assumption

The configs use `motion_channels: 2`, on the assumption that `cv_reader` lays AVC motion vectors
out as `[dx_L0, dy_L0, dx_L1, dy_L1]` and that the L1 pair is dead on a B-frame-free stream.
Your clips have no B-frames, but the *layout* has never been confirmed against the real module.

```bash
cd ~/CoCap
python tools/verify_motion_channels.py --video_dir "$VATEX_SUBSET_ROOT/train" -n 20
```

**Check** — prints `PASS: no B-frames, and channels 2:4 are identically zero.`
If it prints FAIL, set `motion_channels: 4` in `configs/dataset/vatex_subset.yaml` and tell me
what it reported — the slice is wrong and we re-derive the layout.

---

## 8. Build the dataset metadata

```bash
python tools/prepare_vatex_subset.py \
    --annotations "$VATEX_SUBSET_ROOT/VATEX_Caption.json" \
    --train_dir   "$VATEX_SUBSET_ROOT/train" \
    --val_dir     "$VATEX_SUBSET_ROOT/val" \
    --output_dir  ./dataset/vatex_subset
```

**Check** — `train ids : 4999`, `test ids : 1000`, `captions : 59990`.
(4999 not 5000: two clip ids share one video file and the correct segment is not recoverable, so
both are dropped rather than trained on a known-wrong pairing.)

---

## 9. Validate the real data path

Same validator you have already seen pass on synthetic motion/residual values — now with the
real reader. This is the step that catches a bad `cv_reader` build before it costs you a run.

```bash
pytest test/ -q
python tools/validate_data_pipeline.py --variant baseline    -n 200 --build-model
python tools/validate_data_pipeline.py --variant siglip_gpt2 -n 200 --build-model
```

**Check** — both print `PASSED`, and in particular:
- `reader failures (all-zero I-frames) : 0` — **anything above 0 means some clips are training
  as black video**; investigate before running.
- shapes match `iframe (8,3,224,224) motion (8,59,2,56,56) residual (8,59,3,224,224)`
- a finite loss

---

## 10. Get the GPU latency numbers

The CPU numbers in `docs/implementation-plan.md` §11 are only good for relative comparison.
These are the ones that go in the paper.

```bash
for v in baseline siglip siglip_gpt2; do
    python tools/benchmark_latency.py --variant $v --device cuda -n 50
done
```

Record them. This tells you how much the GPT-2 decoder actually costs on real hardware, which
decides whether the KV-cache work is worth doing.

---

## 11. First training run — the baseline

This is the number everything else is measured against, so run it first and run it properly.

```bash
python tools/train_net.py --config-name=exp/train/vatex_subset_baseline
```

Watch the first few minutes for:
- **OOM** → lower `train_dataloader.batch_size` to 1 and raise `trainer.accumulate_grad_batches`
  to 12, keeping the product at 12
- **dataloader starvation** (GPU idle) → raise `num_workers`
- loss decreasing

Then:
```bash
tensorboard --logdir logs/
```

**Check** — validation CIDEr is reported at the end of each epoch and is rising.

---

## 12. Then the variants

```bash
python tools/train_net.py --config-name=exp/train/vatex_subset_siglip        # encoder swap only
python tools/train_net.py --config-name=exp/train/vatex_subset_siglip_gpt2   # full change
```

Run each with at least two seeds — 4,999 clips is a small training set and single-run
differences will be within noise otherwise.

---

## Where to tell me you are

Come back at whichever of these happens first:

- **step 6 fails after a few hours** → we switch to porting the reader
- **step 7 says FAIL** → the motion-channel layout is different from assumed
- **step 9 reports reader failures > 0** → some clips are silently black
- **step 10 numbers are in** → we decide on the KV cache with real data
- **step 11 finishes** → we have a baseline and can start comparing

The open decisions waiting on those results are in `docs/implementation-plan.md`: the `num_gop`
5-vs-8 ablation (§0.5 L3), and the GPT-2 KV cache (§11).
