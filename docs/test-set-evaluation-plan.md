# Test-set evaluation: implementation plan

The study so far reports only the 1,000-clip validation split, because the subset downloaded from
HuggingFace carried `train` and `val` only. This plan adds a genuine held-out test split of 2,000
clips, evaluates all three ablation variants on it, and folds the numbers into the paper.

Everything below was checked against the live HuggingFace repository and the local data before
being written; the "Verified" lines are measurements, not assumptions.

---

## 0. One finding that changes the plan

`Blazewild/Vatex_reencoded` contains **two different encodings** of VATEX, and the directory the
request names (`test_videos/`) is the one that does *not* match our training data.

| Directory | Files | Resolution | `KeyInt` | GOPs/clip | B-frames | Encoder tag |
|---|---|---|---|---|---|---|
| `train/`, `train1/`, `train2/`, `val/`, `test/`, `test1/` | 37,134 | short edge **240**, native aspect (426x240, 320x240, ...) | **60** | ~5 | 0 | `Lavf61.1.100` |
| `selected_8k/`, `test_videos/` | 12,000 | **224x224 square** | **30** | ~9-10 | 0 | `Lavf61.1.100` |
| *our local `train/` + `val/`* | 6,000 | short edge 240, native aspect | 60 | median 5 | 0 | `Lavf61.1.100` |

Our train and val clips are family A. `test_videos/` is family B. Using it directly would break the
evaluation in two independent ways:

1. **Geometry.** The pipeline applies `DictCenterCrop((224, 224))` to a 240p frame of native aspect
   ratio. A 224x224 source makes that crop a no-op, so the model would see horizontally squashed
   content it never saw in training. The squash is not reversible: the 240p original cannot be
   recovered from the square re-encode.
2. **GOP structure.** `num_mv = num_res = 16` and `num_gop = 5` were both chosen for `KeyInt = 60`
   (see `configs/budget/laptop_8gb.yaml` and paper Section 4.2). At `KeyInt = 30` a clip holds 9-10
   GOPs of half the temporal span, so `num_gop = 5` would cover only half the clip. Test scores would
   drop for reasons that have nothing to do with the SigLIP2 or GPT-2 substitution.

Re-encoding family B to look like family A is not a fix - it would add transcode generation loss on
top of an already-destroyed aspect ratio.

**The resolution is better than a compromise.** The 2,000 clip IDs curated in `test_videos/` are
*all* also published in the matched encoding:

```
test_videos/ curated ids                      : 2000
  with exactly 10 captions in VATEX_CAPTIONS.json : 2000
  also present in test/   (family A)          : 1907
  also present in test1/  (family A)          : 174
  present in test/ OR test1/                  : 2000   <- full coverage
```

So we download **the same 2,000 videoIDs the request asks for**, sourced from `test/` and `test1/`
instead of `test_videos/`. Identical clip selection, identical captions, and an encoding that matches
train and val exactly - which is also the direct answer to "use the same video encoding mechanism as
train and val": that mechanism was *no local re-encoding at all* (`tools/video_convert.py` has never
been run), and picking family A preserves that.

### Also verified about the selection

| Property | Result |
|---|---|
| Captions per clip | exactly 10 for all 2,000 |
| videoID -> clip-id collisions | **0** (the train/val subset had some; these do not) |
| Overlap with our 4,999 train clips | **0** files, **0** clip ids |
| Overlap with our 1,000 val clips | **0** files, **0** clip ids |
| VATEX split label on all 2,000 | `test` |
| Download size | ~0.86 GiB (free space on `C:` is 146 GiB) |

Because the download is exactly the 2,000 wanted clips, the "keep 2,000 and delete the rest" step
becomes a verification rather than a deletion. **Nothing gets deleted.** Step 3 asserts set equality
and stops if it does not hold.

### `VATEXCAPTIONS.json`

The repository already publishes it at the root: `VATEX_CAPTIONS.json`, 74.6 MB, 41,269 videos /
349,910 annotations, split counts `train 25,991 / test 12,278 / val 3,000`. Same COCO-style schema as
the local `VATEX_Caption.json` that `tools/prepare_vatex_subset.py` already consumes. There is also a
pre-filtered `test_videos/VATEX_CAPTIONS_test.json` holding exactly our 2,000 clips and their 20,000
captions.

This means the plan is **not blocked** waiting for the file. Step 2 pulls it from the repository. When
you supply your copy we diff the two and stop if the captions for any of the 2,000 clips differ, so
your file still gets the final say.

---

## 1. Download the 2,000 test clips

New tool: `tools/fetch_vatex_testset.py`

- reads the curated ID list from `test_videos/VATEX_CAPTIONS_test.json` (2,000 videoIDs)
- resolves each ID to `test/<id>.mp4`, falling back to `test1/<id>.mp4`
- refuses to fall back to `test_videos/` or `selected_8k/`, so family B can never leak in
- downloads into a fresh directory, 8 parallel connections
- writes a manifest with each file's source directory, size and sha256
- exits non-zero unless exactly 2,000 files landed

```bash
cd /home/seethos/CoCap && . .venv-wsl/bin/activate
export VATEX_SUBSET_ROOT=/mnt/c/Research-Personal/Datasetextract5000train1000validfromhuggingface
python tools/fetch_vatex_testset.py --out "$VATEX_SUBSET_ROOT/test"
```

Target layout, matching the existing two splits:

```
C:\Research-Personal\Datasetextract5000train1000validfromhuggingface\
  VATEX_Caption.json          (existing, 6,001 clips - untouched)
  VATEX_CAPTIONS.json         (new, full 41,269-clip annotation file)
  train\   5,000 mp4          (untouched)
  val\     1,000 mp4          (untouched)
  test\    2,000 mp4          (new)
```

Runtime: a few minutes. Nothing existing is written to.

## 2. Captions

```bash
python tools/fetch_vatex_testset.py --captions-only --out "$VATEX_SUBSET_ROOT"
```

Pulls `VATEX_CAPTIONS.json` and `VATEX_CAPTIONS_test.json`. **Hold point:** when you hand over your
`VATEXCAPTIONS.json`, the tool diffs it against the downloaded copy over the 2,000 clip IDs and
reports any disagreement in ID set or caption text. If they agree, nothing changes; if they do not,
yours wins and we re-run step 5.

## 3. Verify the 2,000 (no deletion)

Assertions, all of which must pass:

- the 2,000 files on disk == the 2,000 videoIDs in the caption JSON, exactly
- every clip has exactly 10 captions
- no videoID maps to more than one clip id (would make the clip's timestamps ambiguous, which is why
  `prepare_vatex_subset.py` has an `--on_collision` flag; measured 0 here)
- zero intersection with the train and val file lists and clip-id lists
- every file opens and decodes

Should any extra file appear, the tool lists it and **stops** rather than deleting; deletion only
happens after you look at that list and approve it.

## 4. Encoding parity sweep

`tools/check_video_integrity.py`, extended with a `--compare-against <dir>` mode, over all 2,000 test
clips plus a 200-clip random sample of train for the reference distribution. Per-clip assertions:

| Property | Required | Why it matters |
|---|---|---|
| codec / profile / pix_fmt | `h264` / `High` / `yuv420p` | `cv_reader` parses H.264 bitstreams |
| short edge | 240 | `DictCenterCrop((224,224))` must crop, not no-op |
| aspect ratio | native (not 1:1) | same |
| B-frame count | 0 | justifies `motion_channels: 2` (paper Section 4.3) |
| largest I-frame gap | <= 60 | `num_mv = num_res = 16` sized for `KeyInt = 60` |
| encoder tag | `Lavf61.1.100` | same extraction pass as train/val |

Then the GOP histogram, against the train reference measured earlier
(`{1:3, 2:7, 3:26, 4:16, 5:105, 6:22, 7:9, 9:3}`, median 5, 74% with >= 5 GOPs), and the duplicate-GOP
fraction at `num_gop = 5` via `tools/validate_data_pipeline.py` pointed at the test split. Paper
Section 4.2.1 reports 4.0% on train; the test split should land near it. If it does not, the number
goes in the paper as a separate measurement rather than being glossed over.

**No re-encoding step.** Parity is achieved by sourcing from family A, which is exactly how train and
val were handled.

## 5. Build the test metadata

New tool: `tools/prepare_vatex_testset.py`. Same logic as `prepare_vatex_subset.py`, but writing a
separate pair of files so the existing ones stay byte-identical and the published results remain
reproducible:

```
dataset/vatex_subset/vatex_testset_caption.json   {"train": [], "test": [2000 ids], "metadata": [20000]}
dataset/vatex_subset/id_to_file_test.json         {clip_id: "<videoID>.mp4"}
```

`VATEXSubsetCaptioningDataset` then needs no code change: it is instantiated with `split: "test"`,
which is already how the val dataloader is configured (`configs/exp/train/vatex_subset_base.yaml`).

Note on sample count: with `unfold_sentences: True` and `split == "test"`, `dataset_vatex.py` keeps
one sample per clip and puts all ten captions into `json_ref` for scoring. So the test dataloader is
**2,000 samples scored against 20,000 references** - not 20,000 samples. This is the same convention
the 1,000-clip val numbers already use.

## 6. Evaluation entry point

There is currently no eval-only entry point - `tools/train_net.py` calls `trainer.fit` and nothing
else. New `tools/eval_net.py`:

- registers the same hydra config store, so the model is composed exactly as training composed it
  (this is what `cocap/modeling/config_store.py` was factored for, and what
  `tools/benchmark_latency.py` already does)
- calls `trainer.validate(model, dataloaders=..., ckpt_path=...)`
- the existing `on_validation_epoch_start` / `validation_step` / `on_validation_epoch_end` hooks run
  unchanged, so decoding, prediction dumping and `pycocoevalcap` scoring are identical to validation
  during training - no second scoring path to keep in sync

Verified prerequisites in the WSL environment: `cv_reader` imports, CUDA available (RTX 4070 Laptop),
`torch 2.6.0+cu124`, `pytorch_lightning 2.6.5`, and `java 21` present so METEOR and the PTB tokenizer
work.

### 6a. Sanity gate - reproduce the published val numbers first

Training used `reader=pre_extract` against a 20.2 GB feature cache that **no longer exists** (0
`.pict_type` files remain), so test evaluation has to use `reader=decode`. That is the same parse,
just not cached - but "should be identical" is not a measurement. So before trusting any test number,
re-run each best checkpoint over the **val** split with `reader=decode` and require it to reproduce
the table in the paper:

| Variant | Checkpoint | Must reproduce |
|---|---|---|
| CLIP + BERT | `logs/vatex_subset_baseline_seed42/.../best-epoch11-cider0.5485.ckpt` | C 54.85, B4 29.67, M 23.40, R 48.92 |
| SigLIP2 + BERT | `logs/vatex_subset_siglip_seed42/version_1/.../best-epoch09-cider0.5933.ckpt` | C 59.33, B4 31.01, M 24.06, R 49.43 |
| SigLIP2 + GPT-2 | `logs/vatex_subset_siglip_gpt2_seed42/version_1/.../best-epoch02-cider0.5656.ckpt` | C 56.56, B4 29.71, M 23.63, R 48.75 |

All three checkpoints were confirmed present, and their epoch and `global_step` counters match the
paper (4,166 optimiser steps per epoch = 49,990 samples / batch 2 / 6 accumulation steps, i.e. the
documented effective batch of 12).

If a val number moves, we find out why before reporting anything on test.

### 6b. Run on the test split

```bash
python tools/eval_net.py --config-name=exp/train/vatex_subset_siglip \
  budget=laptop_8gb reader=decode \
  ckpt_path=logs/vatex_subset_siglip_seed42/lightning_logs/version_1/checkpoints/best-epoch09-cider0.5933.ckpt \
  val_dataloader.dataset.video_root="$VATEX_SUBSET_ROOT/test" \
  val_dataloader.dataset.metadata=./dataset/vatex_subset/vatex_testset_caption.json \
  val_dataloader.dataset.id_to_file=./dataset/vatex_subset/id_to_file_test.json \
  trainer.default_root_dir=./logs/testset_siglip \
  'trainer.callbacks=[]'
```

and the same for `vatex_subset_baseline` and `vatex_subset_siglip_gpt2` with their own checkpoints and
`default_root_dir`. `budget=laptop_8gb` is mandatory: it is the configuration the checkpoints were
trained under (`num_gop 5`, `num_mv/num_res 16`, `n_bp 16`, `max_v_len 10`). The per-variant
`normalize_mean/std` of 0.5 and the `gpt2` tokenizer come from the variant configs, so they apply to
the overridden val dataloader automatically - the only overrides needed are the three paths.
`trainer.callbacks=[]` drops the checkpoint and LR-monitor callbacks, which have no role in a
validate-only pass. Everything else - `precision: bf16-mixed`, `max_words: 32`, uniform frame
sampling, 32-step greedy decode - is inherited unchanged.

Each run writes `logs/testset_<variant>/caption_greedy_pred_validation_<ts>.json`, which keeps the
2,000 generated captions for the qualitative table and lets `tools/compute_cider.py` re-score offline
without a GPU.

### Measured cost

Compressed-domain parse is 254 ms/clip median on the `/mnt/c` mount, so 2,000 clips is ~8.5 min
single-threaded and ~1.4 min across the 6 dataloader workers - fully hidden behind the GPU. From the
paper's latency table (visual 50-55 ms, decode 173-400 ms per clip), expect roughly:

- val sanity gate (3 x 1,000 clips): ~30 min total
- test runs (3 x 2,000 clips): ~60 min total

Call it 2 hours end to end including the encoding sweep.

## 7. Paper updates

New results subsection after `\subsection{Attribution}`, plus a test-set table in the same shape as
`tab:main` (B4 / M / R / C for the three variants). Then:

| Location | Change |
|---|---|
| `tab:deviations` (line 249) | `Evaluation clips` 1,000 -> `1,000 val / 2,000 test` |
| Section 4.1 Data (line 227) | describe the 4,999 / 1,000 / 2,000 three-way split and its provenance |
| `tab:repro` caption | the "evaluation splits differ" hedge can now point at a real held-out split |
| Section 5.2 Attribution | report the test-set delta alongside the val delta |
| Section 6 Limitations (line 505) | "evaluate on a 1,000-clip split" no longer stands as written |
| Abstract, Introduction, Conclusion | wherever `+4.48 CIDEr` is quoted, add the test-set figure |

Framing stays as it is throughout: every delta is **against our own reproduction**, never a claim to
beat CoCap. Two honest caveats to state explicitly:

- VATEX normally withholds its test captions; this mirror publishes 10 captions for 6,000 of the
  12,278 test clips, and our 2,000 are a subset of those. So this is a held-out split drawn from
  VATEX's test partition, **not** the official VATEX leaderboard protocol, and absolute numbers are
  still not comparable to published VATEX test results.
- Still a single seed per variant. A test split removes the "tuned on the reported split" concern; it
  does not address run-to-run spread, and Section 6 should keep saying so.

Then: rebuild `paper/arxiv-submit.zip` and `paper/overleaf-upload.zip`, re-upload to Overleaf,
recompile, and re-check the page count for the arXiv Comments field.

## 8. Commit

New files (`tools/fetch_vatex_testset.py`, `tools/prepare_vatex_testset.py`, `tools/eval_net.py`, the
two test metadata JSONs, this plan) plus the `check_video_integrity.py` extension and the paper edits.
`README.md` gains the test-set results table and the evaluation commands. The video files themselves
live outside the repository, as train and val do.

---

## Decision points

Three things worth your explicit sign-off before step 1 runs:

1. **Source the 2,000 clips from `test/` + `test1/` rather than `test_videos/`.** Same 2,000 videoIDs,
   same captions, but the encoding that matches train and val. This is the whole of Section 0 and the
   only real departure from the request as written.
2. **Nothing is deleted.** The download is exactly the 2,000 wanted clips, so step 3 verifies instead
   of deleting. If a stray file does turn up, you see the list before anything is removed.
3. **`VATEX_CAPTIONS.json` comes from the repository root** so work can start now; your copy is
   diffed against it when you hand it over, and yours wins on any disagreement.
