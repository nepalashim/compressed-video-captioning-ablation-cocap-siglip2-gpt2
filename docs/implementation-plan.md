# CoCap → SigLIP2 + GPT-2 : Implementation Plan

**Goal.** Take the CoCap baseline (compressed-domain video captioning: I-frame + motion-vector + residual → caption) and replace two components:

1. **Vision encoder:** CLIP ViT-B/16 → **SigLIP2** (`google/siglip2-base-patch16-224`)
2. **Caption decoder:** custom BERT-style self-attention decoder → **GPT-2** (`gpt2`)

**Success criterion.** On a fixed VATEX subset (5000 train / 1000 val), the new model beats the reproduced CoCap baseline on CIDEr (primary) and BLEU-4 / METEOR / ROUGE-L (secondary), at comparable or better inference latency. Each swap is evaluated in isolation so we can attribute the gain.

**Runtime decision (locked):** WSL2 + CUDA. `cv_reader` will be built natively (not ported).

---

## 0. Current state (as found)

### Codebase
- Revised CoCap release (Hydra + PyTorch Lightning). Entry point [tools/train_net.py](../tools/train_net.py).
- Model: [cocap/modules/compressed_video/compressed_video_transformer.py](../cocap/modules/compressed_video/compressed_video_transformer.py) (encoders + action encoder), [cocap/modules/compressed_video/compressed_video_captioner.py](../cocap/modules/compressed_video/compressed_video_captioner.py) (`CaptionHead` + `CompressedVideoCaptioner`).
- Lightning glue: [cocap/modeling/lm_cocap.py](../cocap/modeling/lm_cocap.py) (`CoCapLM`).
- Data: [cocap/data/datasets/compressed_video/video_readers.py](../cocap/data/datasets/compressed_video/video_readers.py) `read_frames_compressed_domain` (needs `cv_reader`), dataset classes per benchmark.
- Config: [configs/exp/train/](../configs/exp/train/) + [configs/dataset/](../configs/dataset/).

### Known defects to fix before anything runs
| # | Location | Problem | Fix |
|---|---|---|---|
| D1 | [compressed_video_captioner.py:83](../cocap/modules/compressed_video/compressed_video_captioner.py#L83) | `from cocap.trainer.cocap_trainer import convert_ids_to_sentence` — module does not exist | import from `cocap.modeling.lm_cocap` |
| D2 | [configs/exp/train/base.yaml:69](../configs/exp/train/base.yaml#L69) | `multiprocessing_context: "fork"` | keep `fork` on WSL (fine); document |
| D3 | [video_readers.py:126](../cocap/data/datasets/compressed_video/video_readers.py#L126) | `/usr/bin/ffprobe` hard-coded in `get_frame_type` | not on the training path; leave or parametrize |
| D4 | [cocap/utils/video.py:45](../cocap/utils/video.py#L45) | `/usr/bin/ffmpeg` default in `convert_video` | pass `--ffmpeg_exec $(which ffmpeg)` |
| D5 | [lm_cocap.py:185](../cocap/modeling/lm_cocap.py#L185) | greedy decode runs `max_t_len=77` steps every val | lower `max_t_len` (see §2.3) |
| D6 | base.yaml trainer | `strategy: ddp_find_unused_parameters_true` | single GPU → set `devices: 1`, `strategy: auto` |

### Environment
- Existing `.venv` is **Windows, Python 3.13.1, torch 2.14.0+cpu** — will be abandoned; WSL gets its own.
- All `requirements.txt` deps already resolve to newer releases (audit in §1.3). No package is unavailable.
- Missing everywhere: `cv_reader`, system `ffmpeg`/`ffprobe`, Java JRE, `transformers`.
- GPU: RTX 4070 Laptop, **8 GB VRAM**, Ada (sm_89), driver 596.08 → CUDA 12.4–12.8 OK.

### Dataset (`C:\Research-Personal\Datasetextract5000train1000validfromhuggingface`, in WSL: `/mnt/c/Research-Personal/Datasetextract5000train1000validfromhuggingface`)
- `train/` 5000 `.mp4`, `val/` 1000 `.mp4`. VATEX clips, ~10 s.
- **Measured with PyAV (not assumed):**
  - codec **H.264**, short side **240** ✓
  - **`keyint` is already exactly 60** — measured GOP lengths are 60, 60, 60, … ✓
  - **zero B-frames already** — every sampled video is `I` + `P` only ✓
  - → **no re-encoding step is required.** `tools/video_convert.py` is not used for this dataset.
- **GOP census** (800 videos sampled, demux-only):

  | usable GOPs/video | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8+ |
  |---|---|---|---|---|---|---|---|---|
  | train (400) | 1 | 8 | 28 | 41 | **259** | 49 | 8 | 6 |
  | val (400) | 2 | 15 | 31 | 40 | **247** | 52 | 10 | 3 |

  Median **5**. Only **1.5% train / 0.8% val** have ≥8. See decision L3.
- Files named by **`videoID`** (`--33Lscn6sk.mp4`); captions in `VATEX_Caption.json` keyed by **`video_id`** (`6_4kjPiQr7w_000191_000201`).
- `VATEX_Caption.json` is **COCO-style**: `{"videos":[{video_id,videoID,filename,split}], "annotations":[{id,video_id,caption,split}]}`, 10 captions/video, splits `train`(5001)/`val`(1000). CoCap's [dataset_vatex.py](../cocap/data/datasets/compressed_video/dataset_vatex.py) expects a **different** schema (`metadata["train"]`, `metadata["test"]`, `metadata["metadata"]=[{video_id,sentence}]`).

---

## 0.5 Locked decisions

| # | Decision | Consequence |
|---|---|---|
| **L1** | **No pretraining for the motion and residual encoders** — they stay randomly initialized | Already true in the baseline ([compressed_video_transformer.py:272-285](../cocap/modules/compressed_video/compressed_video_transformer.py#L272)); only the I-frame encoder is pretrained. No change needed; do **not** add pretrained init for these. |
| **L2** | **`keyint=60`, no I-frame preservation step, no re-encoding** | Verified already satisfied by the source data. `M = keyint - 1 = 59` → `num_mv = num_res = 59`. |
| **L3** | **`num_gop = 8`** (as specified) | **Measured**: `validate_data_pipeline.py` reports a **37.5% duplicated-GOP fraction** on the real clips — only ~5.0 of the 8 sampled GOPs are distinct, the rest byte-identical (GOP sampling happens after per-GOP B/P sampling, so duplicate indices yield identical tensors). Three of every eight I-frame/motion/residual encoder passes therefore produce features the model already has. `num_gop: 5` recovers that compute at no information cost; recommended before the runs that matter. |
| **L4** | **No B-frames → motion vector `4×56×56` becomes `2×56×56`** — ✅ **VERIFIED** | Source already has `B=0`, so no re-encode. `cv_reader` returns 4 channels for AVC (`[dx_L0, dy_L0, dx_L1, dy_L1]`); channels `2:4` are dead. Slice to the first 2 in the reader; `motion_encoder.in_channels = 2`; the `== 4` assertion now reads the encoder via `motion_channels`. `bp_type_ids` are then all `0` (P).<br><br>**Confirmed against the real reader** (`tools/verify_motion_channels.py`, 20 videos / 5,605 motion-carrying frames): picture types `{I: 105, P: 5500}` — no B-frames; per-channel `\|max\|` = `[253.0, 125.0, 0.0, 0.0]` — channels 2:4 identically zero. The slice discards nothing. |
| **L5** | **No AGDTR module** | Out of scope for this iteration. Not implemented, not stubbed. Revisit later. |
| **L6** | **Training batch size = 12** | Interpreted as the **effective/optimization** batch. On 8 GB, per-device batch must be 1–2 (the residual tensor alone is `12×8×59×3×224×224` = 852 MB uint8 → 3.4 GB fp32). Use `batch_size: 2` + `accumulate_grad_batches: 6` (= 12). On a stronger GPU, raise `batch_size` and lower accumulation to keep the product at 12. |
| **L7** | **SigLIP2 = `SiglipVisionModel`, FixRes `google/siglip2-base-patch16-224`** | That checkpoint reports `model_type: "siglip"`, so it loads with `SiglipVisionModel` (clean `forward(pixel_values)`), **not** `Siglip2VisionModel` (NaFlex-only; requires `pixel_attention_mask` + `spatial_shapes`). Use **`pooler_output`** (MAP attention-pooling head, learnable probe) as the CLS replacement for `F_ctx`, and **`last_hidden_state`** (196×768, already post-layernormed) as the action-encoder cross-attention memory. Normalization becomes **mean=std=(0.5,0.5,0.5)**. Width 768 throughout (was 512) — which also matches GPT-2's `n_embd=768`. |
| **L8** | **CLIP is needed only for the baseline, not for the proposed model** | `vatex_subset_siglip_gpt2` uses **zero** CLIP weights. `ViT-B-16.pt` (~350 MB) is required by `vatex_subset_baseline` (vision tower + the BERT decoder's word embeddings) and by `vatex_subset_siglip` (word embeddings only). The baseline **cannot be replaced by the paper's published VATEX numbers**: those were trained on full VATEX (~26k clips) while this project trains on 4,999, so the comparison would be invalid. `ViT-L-14.pt` is not needed and is commented out in `model_zoo/urls.txt`. |
| **L9** | **The VATEX subset is the only dataset** | `$VATEX_SUBSET_ROOT` (default `C:/Research-Personal/Datasetextract5000train1000validfromhuggingface`, set it to the `/mnt/c/...` path in WSL) is the single source of video. The upstream `msrvtt` / `msvd` / full-`vatex` configs remain in the repo untouched but are not part of any experiment here. |

---

## 1. Phase 1 — WSL2 + CUDA environment

### 1.1 System (WSL2 Ubuntu 22.04/24.04)
```bash
sudo apt update && sudo apt install -y \
  build-essential cmake pkg-config git aria2 \
  ffmpeg \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev \
  default-jre                                   # pycocoevalcap (METEOR/PTBTokenizer)
```
Verify NVIDIA GPU is visible in WSL: `nvidia-smi` (needs a recent Windows driver + WSL CUDA; no separate Linux driver).

### 1.2 Python env
```bash
cd ~/CoCap                       # clone or bind-mount the repo (see note)
python3.11 -m venv .venv-wsl     # 3.11 or 3.12; avoid 3.13 for native-build friendliness
source .venv-wsl/bin/activate
pip install -U pip wheel setuptools
```
**Repo location:** work on a native-Linux clone (`git clone` into `~/CoCap`), *not* the `/mnt/c` path — I/O over the 9p mount is slow for many-small-file datasets and native builds. Keep the dataset on `/mnt/c` (read-only, large) or copy to `~` if disk allows (~a few GB).

### 1.3 PyTorch (CUDA) + requirements

Install torch first, from the CUDA index:
```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
```

Then the rest. **Recommended versions** (all verified available; deltas from `requirements.txt` pins called out):

| Package | requirements.txt | Use | Why |
|---|---|---|---|
| torch / torchvision | 2.5.1 / 0.20.1 | **latest cu124** (2.6+) | GPU build; Ada fully supported |
| pytorch-lightning | 2.5.1.post0 | latest 2.x | API stable for `Trainer.fit` usage here |
| numpy | 1.26.4 | **2.x OK** | decord 0.6.0 + fvcore import fine under numpy 2 |
| pandas | 2.2.3 | 2.2.x (pin `<3`) | pandas 3.0 has breaking changes; CoCap use is trivial, but pin for safety |
| opencv-python | 4.10 | 4.11.x (pin `<5`) | opencv 5 is new; stay on 4.x |
| hydra-core / hydra-zen | 1.3.2 / 0.15.0 | 1.3.x / 0.16.x | config API compatible |
| omegaconf | 2.3.0 | 2.3.x | — |
| decord | 0.6.0 | try `0.6.0`; fallback **`eva-decord`** | 0.6.0 Linux wheels stop at py3.10; `eva-decord` is the maintained fork for 3.11+ |
| av (PyAV) | 14.4.0 | any 14–18 | CoCap does **not** use `av` for motion vectors; only some readers |
| pycocoevalcap | 1.2 | 1.2 | needs the JRE from §1.1 |
| timm | 1.0.15 | latest 1.0.x | only used if we keep timm paths; `transformers` covers SigLIP2 |
| einops, easydict, ftfy, regex, lz4, fvcore, h5py, flow_vis, colorlog, terminaltables, tabulate, joblib, tqdm, tensorboard, pyyaml, requests, pillow | — | latest | no concerns |
| **transformers** | *(not listed)* | **latest 5.x** (SigLIP2 + GPT-2) | new dependency |
| **sentencepiece** | *(not listed)* | latest | SigLIP2 / Gemma tokenizer backend |
| **accelerate** | *(not listed)* | latest | optional, helps `from_pretrained` device placement |

Produce a `requirements-wsl.txt` from the frozen env after install (`pip freeze`), commit it alongside the original.

### 1.4 Build `cv_reader`
```bash
git clone https://github.com/yaojie-shen/Compressed-Video-Reader.git ~/Compressed-Video-Reader
cd ~/Compressed-Video-Reader
# follow its README: typically a CMake build linking the system libav* from §1.1,
# then `pip install .` (it ships a pyproject / setup that builds the extension)
python -c "import cv_reader; print('cv_reader OK')"
```
**Risk:** the extension targets a specific FFmpeg major (H.264 MV/residual extraction API). If the Ubuntu `libavcodec-dev` version mismatches, either (a) `apt` a matching FFmpeg from a PPA, or (b) build FFmpeg from source and point CMake at it. Budget a half-day here. This is a hard blocker for the whole data path — do it before touching model code.

### 1.5 CLIP weights (needed for the baseline reproduction in Phase 3)
```bash
bash model_zoo/download_model.sh          # → model_zoo/clip_model/{ViT-B-16.pt, ViT-L-14.pt}
```
Note the path inconsistency: [compressed_video_transformer.py:84](../cocap/modules/compressed_video/compressed_video_transformer.py#L84) resolves via `~/.cache/clip`, [compressed_video_captioner.py:130](../cocap/modules/compressed_video/compressed_video_captioner.py#L130) via `model_zoo/clip_model`. Symlink one to the other, or pass an absolute path in config.

### 1.6 Gate: environment is ready when
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → `True`
- [ ] `python -c "import cv_reader, decord, transformers"` → no error
- [ ] `pytest test/` — expect the `cv_reader`-dependent tests to pass, others already do
- [ ] `java -version` works

---

## 2. Phase 2 — VATEX subset data integration

### 2.1 Re-encoding — NOT REQUIRED

Verified: the source clips are already H.264, `keyint=60`, **zero B-frames**, 240p short side (see §0). `tools/video_convert.py` is **not** run for this dataset. Videos are consumed in place.

Consequences that still need code changes:
- `M = keyint − 1 = 59` → `num_mv = num_res = 59` (matches the paper).
- **No B-frames** → `bp_type_ids` are uniformly `0` (P-frame). The `sample: "pad"` path would emit `type_ids_mv = 2` which overflows `ActionEncoder`'s `bp_type_embedding` (`n_bp_type=2`, [compressed_video_transformer.py:120](../cocap/modules/compressed_video/compressed_video_transformer.py#L120)). **Fix: `n_bp_type=3`** (P / B / pad) so either sampling mode is safe.
- **MV channel reduction (L4)**: `cv_reader` returns 4 channels for AVC; with `B=0` the L1 (backward) pair is dead → slice to `[:2]`, giving `2×56×56`.

### 2.2 Metadata + filename adapter

Add `tools/prepare_vatex_subset.py` that reads the COCO-style `VATEX_Caption.json` and the `train/` `val/` dirs, and writes:

`dataset/vatex_subset/vatex_subset_caption.json`:
```json
{
  "train": ["<video_id>", ...],           // 5000 ids present on disk
  "test":  ["<video_id>", ...],           // the 1000 val ids  (code uses split key "test")
  "metadata": [ {"video_id": "<video_id>", "sentence": "<caption>"}, ... ]   // 60000 rows
}
```
plus `dataset/vatex_subset/id_to_file.json` mapping `video_id → <videoID>.mp4`.

Then add a dataset subclass `cocap/data/datasets/compressed_video/dataset_vatex_subset.py`:
```python
class VATEXSubsetCaptioningDataset(VATEXCaptioningDataset):
    def __init__(self, *a, id_to_file: str, **kw):
        self._id_to_file = load_json(id_to_file)
        super().__init__(*a, **kw)
    def _get_video_path(self, video_id):
        return os.path.join(self.video_root, self._id_to_file[video_id])
```
(`VATEXCaptioningDataset` builds `_get_video_path` inline in `_get_video`; refactor that one line to call `self._get_video_path(video_id)` so the subclass hook works — small change to [dataset_vatex.py:98](../cocap/data/datasets/compressed_video/dataset_vatex.py#L98).)

`json_ref` for eval: base class builds it from `metadata[split]` + all sentences → gives all 10 refs/video for the 1000-val. Good, no change.

### 2.3 Configs

`configs/dataset/vatex_subset.yaml`:
```yaml
_target_: cocap.data.datasets.compressed_video.dataset_vatex_subset.VATEXSubsetCaptioningDataset
video_root: <abs path>/train        # overridden per split
metadata: ./dataset/vatex_subset/vatex_subset_caption.json
id_to_file: ./dataset/vatex_subset/id_to_file.json
video_reader: read_frames_compressed_domain
max_frames: 8
video_size: [224, 224]
max_words: 32           # was 77; VATEX captions ~15 words; big greedy-val speedup
unfold_sentences: true
cv_config:
  num_gop: 8            # L3 — data supplies ~5; 5 is the information-preserving value
  num_mv: 59            # L2 — keyint 60 => M = 59
  num_res: 59
  motion_channels: 2    # L4 — no B-frames, slice off the dead L1 pair
  with_residual: true
  use_pre_extract: false
  sample: rand
```

`configs/exp/train/vatex_subset_*.yaml` — trainer overrides:
```yaml
trainer:
  default_root_dir: ./logs/vatex_subset_baseline
  devices: 1
  strategy: auto
  precision: bf16-mixed        # Ada; halves activation memory
  accumulate_grad_batches: 6   # L6: 2 x 6 = effective batch 12
  max_epochs: 30
train_dataloader: { batch_size: 2, num_workers: 6 }
val_dataloader:   { batch_size: 2, num_workers: 6 }
model: { lr: 1e-4, clip_lr: 1e-6, warmup_ratio: 0.1 }
```
**L6 memory note.** The residual tensor is `batch × num_gop × num_res × 3 × 224 × 224`. At `batch=12, gop=8, res=59` that is 852 MB as uint8 and **3.4 GB once cast to fp32** — infeasible on 8 GB before a single activation. Keep the *product* `batch_size × accumulate_grad_batches = 12`; on a stronger GPU raise `batch_size` and lower accumulation.

Also set `max_v_len = num_gop * 2` (= 16 visual tokens) and `max_t_len = max_words` on the caption head so the decoder's expected visual length tracks `num_gop`.

### 2.4 Gate
- [x] `prepare_vatex_subset.py` runs; counts reconcile (4999/1000, 59990 rows — 2 ids dropped, §0)
- [x] shapes verified end to end via `tools/validate_data_pipeline.py --fake-reader`
- [ ] the same command **without** `--fake-reader`, over the whole train split, once `cv_reader` exists

### 2.5 Validating without `cv_reader`

`tools/validate_data_pipeline.py` walks real video files through the dataset, the transforms,
collation and (with `--build-model`) the model and loss. `--fake-reader` swaps in
`cocap/data/datasets/compressed_video/fake_cv_reader.py`, which takes **picture types and frame
indices from the real files via PyAV** and synthesises only the motion-vector and residual
*values*. That exercises every piece of logic that does not live inside the native module: GOP
grouping, the `len(g) > 2` filter, B/P sampling, GOP sampling, the motion-channel slice, padding
and masks, the dict transforms, and the model's dimension asserts.

Result on the real clips (both variants, `--build-model`):

```
expected  : iframe (8,3,224,224)  motion (8,59,2,56,56)  residual (8,59,3,224,224)
reader failures (all-zero I-frames) : 0
duplicated GOP fraction             : 37.5%   -> only ~5.0 of 8 GOPs are distinct
no shape/dtype/label problems found
baseline    prediction_scores (2,32,49408)  loss 292.2
siglip_gpt2 prediction_scores (2,32,50257)  loss 187.2
```

Any number produced with `--fake-reader` is meaningless; the stub logs a warning and sets
`cv_reader.IS_FAKE`. It exists to de-risk the pipeline, not to produce results.

### 2.6 The reader's silent failure mode

`read_frames_compressed_domain` catches every exception and returns **all-zero tensors** with a
`False` success flag — and `get_video` used to discard that flag, so an unreadable video became
a black clip that trained silently. For a paper this is a correctness landmine: a few percent of
silently-black clips would degrade every metric with no visible cause.

Now: the flag is propagated, failures are logged (`logger.warning`, not `print`) and recorded in
`video_text_base.GET_VIDEO_FAILURES`, `get_video(strict=True)` raises instead, and
`validate_data_pipeline.py` counts all-zero I-frames explicitly. **Check that count is 0 before
trusting any run.**

---

## 3. Phase 3 — Reproduce the CoCap baseline on the subset

This is the number every later experiment is measured against. **Do not skip.**

1. Apply fixes D1, D5, D6.
2. Train stock CoCap (CLIP ViT-B/16 + BERT decoder) with the Phase 2 config, seed fixed, ~30 epochs.
3. Record on the 1000-val: CIDEr, BLEU-4, METEOR, ROUGE-L, and **inference latency** (ms/clip, batch 1, greedy) — reuse the timing style from paper Table 3; add a small `tools/benchmark_latency.py` if needed.
4. Repeat with **2 seeds** (small data → variance). Report mean ± spread.

Expected: below the paper's full-VATEX 52.7 CIDEr (ViT-B/16) because the train set is ~1/6 the size — that's fine, it's a controlled internal baseline.

**Deliverable:** `logs/vatex_subset_baseline/RESULTS.md` with the table + exact config hash + commit.

---

## 4. Phase 4 — Swap 1: SigLIP2 vision encoder (decoder unchanged)

### 4.1 New module — verified interface (L7)

`google/siglip2-base-patch16-224` reports `model_type: "siglip"`, so it loads as **`SiglipVisionModel`**, whose forward is a clean `forward(pixel_values)`. `Siglip2VisionModel` is for the **NaFlex** checkpoints only and requires `pixel_attention_mask` + `spatial_shapes` — not used here.

| | CLIP ViT-B/16 (baseline) | SigLIP2-base-patch16-224 |
|---|---|---|
| class | local `IFrameEncoder(VisionTransformer)` | `transformers.SiglipVisionModel` |
| patch tokens | 196 + 1 CLS | **196, no CLS** |
| context vector | `ln_post(x[:,0]) @ proj` → 512 | **`pooler_output`** → 768 (MAP head: learnable probe attends over patches) |
| cross-attn memory | `ln_post_hidden(x[:,1:]) @ proj_hidden` → 196×512 | **`last_hidden_state`** → 196×768 (already post-layernormed) |
| normalization | ImageNet mean/std | **(0.5,0.5,0.5) / (0.5,0.5,0.5)** |
| layers / heads / intermediate | 12 / 12 / 3072 | 12 / 12 / 3072 |

`cocap/modules/siglip/siglip_iframe_encoder.py` returns the same tuple shape as `IFrameEncoder.forward` so it is drop-in for `CompressedVideoTransformer`:
```python
out = self.model(pixel_values=x)
outputs = (out.pooler_output,)                     # -> feature_context
if output_all_features:  outputs += (out.last_hidden_state,)   # -> cross-attn memory
if output_attention_map: outputs += (None,)        # HF doesn't expose per-layer maps
```

Propagation of the 512 → **768** width change: `CompressedVideoTransformer.output_dim`, `ActionEncoder(width=768)`, motion/residual `VisionTransformer(output_dim=768)` (randomly initialized per **L1** — only the output projection resizes; motion width stays `768//4=192`), `CaptionHead(visual_feature_size=768, hidden_size=768)`. 768 also equals GPT-2's `n_embd`, so the Phase-5 visual→decoder projection is dimension-preserving.

Token count is **identical to CLIP's 196**, so the action encoder's cross-attention is structurally unchanged.

Resolution stays 224 (matches the existing centre crop). `-256` / `-384` variants cost more memory; skip on 8 GB.

### 4.2 Wiring changes
- `CompressedVideoTransformer.from_pretrained` and its hydra `builds(...)` config: add a branch / new builder that constructs `SiglipIFrameEncoder` instead of `IFrameEncoder.from_pretrained`. New config `iframe_encoder=siglip`.
- `CompressedVideoTransformer.forward` ([:206](../cocap/modules/compressed_video/compressed_video_transformer.py#L206)): the `iframe_attn` output becomes `None`; guard the downstream `einops.rearrange` on it (it's only returned for visualization, never consumed by the caption head).
- **Optimizer param groups** [lm_cocap.py:83-93](../cocap/modeling/lm_cocap.py#L83): the `pretrained_modules` prefix list hard-codes CLIP submodule names (`...rgb_encoder.conv1`, `.transformer`, ...). Replace the rgb-encoder entries with `compressed_video_transformer.rgb_encoder.model` (the whole HF module) so SigLIP2 weights get `clip_lr=1e-6` (or `freeze=True` initially).
- CLS-token assumptions: none downstream beyond the encoder — `feature_context`/`feature_action` are the only things the caption head sees.

### 4.3 Experiment
SigLIP2 + **BERT decoder**, everything else identical to Phase 3 (data, schedule, seeds). Compare to baseline → isolates the encoder contribution.

Sub-ablation (cheap, informative): `feature_context` = `pooler_output` vs mean-pooled `last_hidden_state` vs both concatenated.

---

## 5. Phase 5 — Swap 2: GPT-2 decoder

### 5.1 New caption head
`cocap/modules/gpt2/gpt2_caption_head.py`, **prefix-LM** design (mirrors CoCap's "concat visual + text, causal mask"):
```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class GPT2CaptionHead(nn.Module):
    def __init__(self, name="gpt2", visual_dim=768, max_v_len=8, max_t_len=32,
                 n_visual_types=2, label_smoothing=0.1):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(name)      # 12L, 768d, vocab 50257
        d = self.gpt2.config.n_embd
        self.visual_proj = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, d))
        self.visual_type_emb = nn.Embedding(n_visual_types, d)  # context vs action
        self.visual_pos_emb  = nn.Embedding(max_v_len, d)

    def forward(self, visual_output, input_ids, input_mask):
        ctx, act = visual_output["feature_context"], visual_output["feature_action"]  # (B,N,dv)
        v = torch.cat([ctx, act], dim=1)                       # (B, 2N, dv)
        v = self.visual_proj(v) + <type/pos emb>
        t = self.gpt2.transformer.wte(input_ids)               # (B, T, d)
        inp = torch.cat([v, t], dim=1)
        attn = torch.cat([ones(B, v.size(1)), input_mask], dim=1)
        out = self.gpt2(inputs_embeds=inp, attention_mask=attn)
        logits = out.logits[:, -input_ids.size(1):]            # text positions only
        return logits
```
- **Causality.** GPT-2 is causal by default → visual prefix is fully visible to every text step; text is left-to-right. Same information flow as `make_pad_shifted_mask` ([bert.py:12](../cocap/modules/bert.py#L12)).
- **Loss.** Keep `LabelSmoothingLoss` but `target_vocab_size=50257`. Switch masking from `ignore_index=0` to `-100` labels (GPT-2 pad = eos = 50256 would collide with a real token). Build `input_labels` as `input_ids` shifted left, `-100` on pad and on the prompt's non-supervised positions. Update [loss.py:31](../cocap/modeling/loss.py#L31) + the datasets.
- **Generation.** Two options: (a) keep the manual greedy loop in [`validation_step`](../cocap/modeling/lm_cocap.py#L182) (works unchanged, just slower), or (b) call `self.gpt2.generate(inputs_embeds=v_prefix, ...)` with KV-cache — faster, recommended once correctness is confirmed.

### 5.2 Tokenizer swap (touches data)
- Datasets currently use `clip.tokenize` + `clip._tokenizer` ([dataset_vatex.py:111](../cocap/data/datasets/compressed_video/dataset_vatex.py#L111)). Add a tokenizer abstraction or a config flag `text_tokenizer: {clip|gpt2}`.
- GPT-2: `GPT2TokenizerFast.from_pretrained("gpt2")`, `pad_token = eos_token`. `input_ids` = `bos? + bpe(caption) + eos`; GPT-2 has no BOS — use `eos` (50256) as the decode-start symbol (standard). `max_words=32`.
- `convert_ids_to_sentence` ([lm_cocap.py:30](../cocap/modeling/lm_cocap.py#L30)) → `gpt2_tokenizer.decode(ids, skip_special_tokens=True)`.
- `CaptionHead.cap_config.{BOS_id,EOS_id,PAD_id}` consumers in `validation_step` → point at GPT-2 ids.

### 5.3 Optimizer
Add `caption_head.gpt2` to the low-LR / no-decay pretrained group; remove the BERT-specific prefixes (`caption_head.cap_sa_decoder.*`, `caption_head.prediction_head.decoder`) from [lm_cocap.py:83](../cocap/modeling/lm_cocap.py#L83). The `whitelist/blacklist_weight_modules` split still applies to the new projection layers.

### 5.4 Experiments
- SigLIP2 + GPT-2 (full innovation) vs baseline.
- CLIP + GPT-2 (isolates decoder) — optional but completes the 2×2.

---

## 6. Evaluation protocol & final matrix

| Vision \ Decoder | BERT (CoCap) | GPT-2 |
|---|---|---|
| CLIP ViT-B/16 | **baseline (Phase 3)** | Phase 5 optional |
| SigLIP2-base | Phase 4 | **Phase 5 main** |

- **Same** 5000/1000 split, schedule, `num_gop=4`, augmentation, ≥2 seeds each.
- Metrics: CIDEr (primary), BLEU-4, METEOR, ROUGE-L via `pycocoevalcap` ([eval_captioning.py](../cocap/modeling/eval_captioning.py)) — unchanged, just needs the JRE.
- **Latency**: ms/clip at batch 1, greedy, on the 4070 — CoCap's headline claim is speed, so a decoder swap that regresses latency needs justifying. GPT-2 (12 layers) is heavier than CoCap's 2-layer BERT decoder; measure and report. Consider `gpt2` (small) only; not `gpt2-medium`.
- Optional paper-style ablations if time: input modalities (I / I+MV / I+MV+Res, paper Table 4), `num_gop` (Table 5), encoder depth.
- Log everything to TensorBoard + a committed `RESULTS.md` per run.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `cv_reader` won't build against distro FFmpeg | build FFmpeg from source pinned to the version its README names; document in `docs/cv_reader-build.md` |
| 8 GB VRAM: residual encoder processes `num_gop×num_res` = 4×29 = 116 images/clip at 224² | `num_gop=4`, `num_res≤29`, `bf16-mixed`, `accumulate_grad_batches`, gradient checkpointing on the ViT encoders, freeze SigLIP2 initially |
| 5k train set → SigLIP2/GPT-2 overfit | freeze backbones or very low LR, label smoothing, early stop on val CIDEr, strong hflip/crop aug |
| SigLIP2 MAP-pooled single vector loses spatial context vs CLIP CLS | ablation in §4.3; fall back to mean-pooled patch tokens |
| GPT-2 BPE ≠ CLIP BPE → all text plumbing changes | isolate behind `text_tokenizer` config flag; add unit test for round-trip encode/decode |
| Short VATEX clips → few GOPs, `sample:"pad"` path bug | `n_bp_type=3`, keep `sample:"rand"`, `num_mv/res ≤ min GOP len` |
| Greedy val at `max_t_len=77` is very slow on 1000 clips ×2 | `max_words=32`; later switch to `generate()` with KV cache |
| WSL `/mnt/c` dataset I/O slow | copy dataset into ext4, or pre-extract compressed features once (`use_pre_extract`) |

---

## 8. File change map

**New**
- `tools/prepare_vatex_subset.py` — COCO-json → CoCap-json + `id_to_file.json`
- `tools/benchmark_latency.py` — ms/clip timing
- `cocap/data/datasets/compressed_video/dataset_vatex_subset.py`
- `cocap/modules/siglip/siglip_iframe_encoder.py` (+ hydra `builds` config)
- `cocap/modules/gpt2/gpt2_caption_head.py` (+ hydra `builds` config)
- `configs/dataset/vatex_subset.yaml`
- `configs/exp/train/vatex_subset_{baseline,siglip,siglip_gpt2}.yaml`
- `docs/cv_reader-build.md`, `requirements-wsl.txt`

**Modified**
- [compressed_video_captioner.py:83](../cocap/modules/compressed_video/compressed_video_captioner.py#L83) — fix import (D1)
- [dataset_vatex.py:98](../cocap/data/datasets/compressed_video/dataset_vatex.py#L98) — extract `_get_video_path` hook
- [compressed_video_transformer.py](../cocap/modules/compressed_video/compressed_video_transformer.py) — SigLIP2 encoder branch, `n_bp_type=3`, dim propagation, guard `None` attn maps
- [lm_cocap.py](../cocap/modeling/lm_cocap.py) — optimizer param-group prefixes, `convert_ids_to_sentence` tokenizer, val decode ids, `max_t_len`
- [loss.py:31](../cocap/modeling/loss.py#L31) — `-100` ignore index, vocab size
- [transforms.py](../cocap/data/datasets/compressed_video/transforms.py) / dataset configs — configurable iframe normalization stats, tokenizer flag
- [configs/exp/train/base.yaml](../configs/exp/train/base.yaml) — `devices: 1`, `strategy: auto`, precision

---

## 9. Sequencing checklist

- [ ] **P1** WSL2 + CUDA torch + requirements + **cv_reader builds** + JRE + pytest green
- [x] **P2a** `prepare_vatex_subset.py` → 4999 train / 1000 val / 59990 captions (re-encoding not needed, §2.1)
- [x] **P2b** dataset class, tokenizer abstraction, motion-channel plumbing, configs — all three compose
- [ ] **P2c** full-set `__getitem__` sweep (needs `cv_reader`) + `verify_motion_channels.py`
- [ ] **P3** reproduce CoCap baseline (CLIP+BERT), 2 seeds → `RESULTS.md` (the number to beat)
- [ ] **P4** SigLIP2 + BERT → compare
- [ ] **P5** SigLIP2 + GPT-2 (+ optional CLIP + GPT-2) → compare
- [ ] **P6** fill the 2×2 matrix + latency + ablations → writeup

---

## 10. Implementation status

Everything below is written, composes under Hydra, and is covered by tests that run without
`cv_reader` (8 new tests in `test/cocap/modules/compressed_video/test_siglip_gpt2_variant.py`,
all passing; the 4 pre-existing CLIP-baseline model tests still pass).

**New**
| File | Purpose |
|---|---|
| `cocap/data/tokenizers.py` | `TextTokenizer` interface + `ClipTextTokenizer` / `GPT2TextTokenizer` |
| `cocap/data/datasets/compressed_video/dataset_vatex_subset.py` | subset dataset, `video_id -> videoID.mp4` indirection |
| `cocap/modules/siglip/siglip_iframe_encoder.py` | `SiglipIFrameEncoder`, drop-in for `IFrameEncoder` |
| `cocap/modules/gpt2/gpt2_caption_head.py` | `GPT2CaptionHead`, prefix-LM, drop-in for `CaptionHead` |
| `tools/prepare_vatex_subset.py` | COCO-style json → CoCap schema + id map |
| `tools/verify_motion_channels.py` | **run in WSL** to confirm the L4 slice before trusting it |
| `tools/benchmark_latency.py` | per-clip visual/decode latency from an experiment config |
| `cocap/modeling/config_store.py` | shared hydra registration, so tools compose the same configs |
| `scripts/setup_wsl.sh` | one-command Phase 1 bring-up |
| `requirements-wsl.txt` | curated dependency set (see §1.3) |
| `configs/dataset/vatex_subset.yaml` | subset dataset config |
| `configs/exp/train/vatex_subset_{base,baseline,siglip,siglip_gpt2}.yaml` | the three runs |
| `test/.../test_siglip_gpt2_variant.py` | wiring tests for the new path |

**Modified**
| File | Change |
|---|---|
| `compressed_video_captioner.py` | fixed the dead `cocap.trainer.cocap_trainer` import (D1); `visual_feature_size` override on `CaptionHead.from_pretrained`; two new captioner configs |
| `compressed_video_transformer.py` | `from_siglip_pretrained` + shared `_assemble`; `motion_channels` and `n_bp_type` configurable; assertion reads the encoder instead of hard-coding 4; `None` attention maps tolerated |
| `lm_cocap.py` | tokenizer-driven decode; separate `decoder_lr` group; configurable pretrained prefixes; rank-based fallback for unclassified params; frozen params excluded from the optimizer; `removeprefix("video")` instead of `split("video")` |
| `loss.py` | `.reshape` instead of `.view` — GPT-2's sliced logits are non-contiguous |
| `dataset_vatex.py` | tokenizer abstraction, configurable normalization stats |
| `video_readers.py`, `video_text_base.py` | `motion_channels` plumbed through `CVConfig` and the reader |
| `clip/clip.py` | `pkg_resources` → `packaging` (removed in setuptools 84) |
| `tools/train_net.py` | `model` is a selectable config group; `MISSING` default so variants merge |
| `configs/exp/train/{msrvtt,msvd,vatex}_captioning.yaml` | explicit `- /model@model: cocap` |

**Known open items**
- `motion_channels: 2` rests on the `[dx_L0, dy_L0, dx_L1, dy_L1]` layout — **unverified until `cv_reader` exists**; `tools/verify_motion_channels.py` settles it and fails loudly if wrong.
- `num_gop: 8` duplicates GOPs for ~98.5% of clips (L3). Compare against `num_gop: 5`.
- **GPT-2 decoding costs 4.6× the baseline decoder** — see §11.

---

## 11. Measured latency (before any training)

`tools/benchmark_latency.py`, synthetic inputs matching each experiment config, batch 1,
`max_t_len=32`, CPU (relative comparison only — re-measure on the 4070 and report those):

| variant | params | visual ms | decode ms | total ms |
|---|---|---|---|---|
| `baseline` (CLIP + BERT) | 171.0M | 1442 | **313** | 1730 |
| `siglip` (SigLIP2 + BERT) | 182.9M | 1523 | 285 | 1816 |
| `siglip_gpt2` (proposed) | 252.7M | 1554 | **1426** | 2970 |

**Reading.** Swapping CLIP → SigLIP2 costs ~6% on the visual path, which is the cheap half of the
change. The decoder swap is where the cost is: GPT-2 is 12 layers against the baseline head's 2,
and the greedy loop re-runs the full 16-visual + 32-text sequence at *every* one of the 32 steps,
so the work is O(T²).

This matters because speed is CoCap's headline claim — a CIDEr win that costs 1.7× total latency
is a weaker result. Two mitigations, in order of value:

1. **KV cache.** `GPT2CaptionHead` should expose a `generate()` path using `past_key_values`, so
   decoding becomes ~32 single-token steps instead of 32 full-sequence passes. This is the bulk
   of the 1426 ms. **Caveat for honest reporting:** the baseline BERT head has no KV cache
   either, so the internal 2×2 comparison must either give both heads one or neither. Report
   "GPT-2 (no cache)" and "GPT-2 (cached)" separately; only the cached number belongs in an
   external comparison against SwinBERT et al.
2. **Fewer decoder layers.** `gpt2` distilled to 6 layers, or `distilgpt2`, trades capacity for
   latency — worth an ablation row if the cached number is still uncompetitive.

Decide this before Phase 5 concludes, not after.
