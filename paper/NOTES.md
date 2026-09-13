# Running notes for the preprint

Things that are easy to forget between running an experiment and writing it up. Add to this
**as things happen**, not afterwards.

---

## Must appear in the paper

### Seeding (Limitations)
- Baseline (CoCap reproduction): **epochs 0-7 unseeded, epochs 8-11 seed 42**. The run was
  interrupted by a WSL shutdown after epoch 7 and resumed from `epoch07.ckpt`; seeding support
  was added to the codebase in between. Not exactly reproducible.
- `siglip` and `siglip_gpt2`: **seed 42 throughout**, fully reproducible.
- **All results are single-seed.** This is the headline limitation, and it applies to every
  variant. Run-to-run spread on 4,999 clips is plausibly +-1-2 CIDEr, so any gap below ~2 points
  cannot be distinguished from noise.

### Deviations from the published CoCap setup
State these in Experimental Setup, not a footnote. They are why absolute numbers are **not**
comparable with the published paper.

| | this work | CoCap paper |
|---|---|---|
| training clips | **4,999** | ~25,991 (full VATEX train) |
| test clips | **1,000** (subset) | full VATEX public test |
| GOPs per clip (N) | **5** | 8 |
| motion vectors / residuals per GOP (M) | **16** | 59 |
| motion vector channels | **2** | 4 |
| epochs | **12** | 20 |
| batch size | **12** (2 x 6 accumulation) | 64 |
| lr decay gamma | **0.91** (retuned for 12 epochs) | 0.95 |

### Justifications to give for each deviation
- **N=5 not 8**: measured, not arbitrary. The clips contain ~5 GOPs (median); at N=8 the sampler
  drew **35% byte-identical duplicate GOPs**, at N=5 it draws **0%**. So 8 added compute and
  duplicated visual tokens without adding information.
- **M=16 not 59**: forced by 8 GB VRAM. At N=8, M=59 the residual tensor alone is 944 images of
  224^2 per batch; VRAM sat at 7710/8188 MiB and the allocator thrashed (100% GPU utilisation at
  29 W of an 89 W budget), giving 0.04 it/s or ~6.5 days per epoch. Sampling is re-drawn every
  epoch (`sample: "rand"`), so the model still sees the whole GOP across training.
- **2 motion channels not 4**: verified empirically. The source clips contain **no B-frames**
  (measured: {I: 105, P: 5500} over 20 clips / 5,605 motion-carrying frames), so the L1 pair of
  `[dx_L0, dy_L0, dx_L1, dy_L1]` is identically zero (per-channel |max| = [253, 125, 0, 0]).
  Dropping it discards nothing.
- **Applied identically to all three variants**, so the internal comparison remains valid.

### FINAL baseline numbers (epoch 11, run complete)
CIDEr **54.85**, BLEU-4 29.67, METEOR **23.40**, ROUGE-L 48.92.

Per-metric peaks differ: BLEU-4 peaked at epoch 8 (30.28) and ROUGE-L at epoch 9 (49.05).
We monitor and report CIDEr, so epoch 11 is the reported checkpoint. Say so explicitly.

**Note the direction of the gap.** Our 54.85 sits *above* the published 52.7 despite one
fifth the training data. Do not read this as an improvement - it almost certainly reflects
the 1,000-clip evaluation subset differing from the full VATEX test split. State that
attribution in the paper rather than leaving the higher number to speak for itself.

### Reproduction fidelity (a strength - report it)
The CoCap reproduction reached **CIDEr 52.68 / BLEU-4 30.03 / METEOR 22.90 / ROUGE-L 48.93** at
epoch 7, against the paper's published VATEX numbers of 52.7 / 31.4 / 23.2 / 49.4. Different
test split, so not like-for-like - but landing this close is evidence the reproduction is
faithful, which is what licenses every comparison built on it.

### Training curve - RESOLVED: the baseline converged
Earlier concern that 12 epochs might be too few is **settled**. Per-epoch CIDEr gains:

| epoch | 7 | 8 | 9 | 10 |
|---|---|---|---|---|
| CIDEr | 52.68 | 53.89 | 54.55 | 54.65 |
| gain | +2.5 | +1.2 | +0.7 | +0.1 |

The curve flattens cleanly, so 12 epochs is sufficient and the models are **not** under-trained.
State this in the paper - it removes an obvious reviewer question. The retuned
`lr_decay_gamma=0.91` annealed the schedule properly over the shortened run.

Note also that BLEU-4 peaked at epoch 8 (30.28) and METEOR/ROUGE at epoch 9, while CIDEr rose
until epoch 10. Best-checkpoint selection therefore depends on the monitored metric; we monitor
CIDEr. Worth one sentence in the paper so the choice is explicit.

Full curve recorded in `docs/checking-results.md`.

### Training loss is NOT comparable across variants
The GPT-2 run shows loss ~141 where the baseline showed ~250-300 at the same iteration.
**This says nothing about caption quality.** Three reasons:

1. Different vocabularies (50257 vs 49408), so label smoothing spreads differently.
2. The loss sums over valid tokens (reduction="sum"), and the two tokenizers split
   the same caption into different numbers of tokens.
3. GPT-2 begins as a trained language model, so its LM loss starts far lower whether or
   not it is using the visual input at all.

Only CIDEr / BLEU-4 / METEOR / ROUGE-L are comparable. Do not plot the loss curves together
in the paper, and do not cite the loss as evidence of anything.

### Lightning eval-mode warning is benign
`Found 323 module(s) in eval mode at the start of training` appears for the HuggingFace
variants. `from_pretrained` returns models in eval mode and Lightning inspects before calling
`.train()`. Verified: afterwards zero modules remain in eval and all 37 GPT-2 dropout layers
are active at p=0.1. No action needed, but have the answer ready.

### Latency
CoCap's central claim is speed. Report `tools/benchmark_latency.py` numbers for all three
variants **on the same GPU as the accuracy numbers**. Preliminary CPU measurement suggested
GPT-2 decoding costs ~4.6x the baseline decoder (12 layers vs 2, and the greedy loop has no KV
cache, so decoding is O(T^2)). **If the GPU numbers confirm a regression, report it.** An
accuracy gain that hides a speed cost is not a defensible result for a paper whose baseline is
about speed.

---

## Environment reproducibility (Appendix)

- WSL2 / Ubuntu 24.04, Python 3.12, PyTorch (cu124), RTX 4070 Laptop 8 GB
- `cv_reader` needs two patches on any modern toolchain; both are in `docs/runbook.md` section 6:
  - FFmpeg 5.1 does not assemble under binutils >= 2.41 (upstream fix `effadce6c7`, masking the
    shift constant with `& 0x1F`)
  - `api.cpp` predates NumPy 2's tighter `PyArray_GETPTR3` signature
- Features are pre-extracted once (`tools/pre_extract.py`): 3.4 MB/clip, ~20 GB for 6,000 clips,
  ~8 minutes. Training reads the cache rather than re-parsing H.264 per sample.

---

## Checklist before submitting

- [ ] all three runs complete, metrics table filled
- [ ] latency numbers measured on GPU for all three
- [ ] qualitative examples: a few predicted vs ground-truth captions per variant
- [ ] limitations section written (single seed, subset, reduced sampling budget)
- [ ] deviations table in Experimental Setup
- [ ] framing checked: "against our reproduction" everywhere, never "beats CoCap"
- [ ] author name/affiliation/ORCID correct
- [ ] code + config released, with the exact commit hash of each run
