# Does a stronger vision encoder or a pretrained decoder help compressed-domain video captioning?

A controlled three-way ablation built on **CoCap** (*Accurate and Fast Compressed Video Captioning*, ICCV 2023), asking which half of the architecture is actually worth modernising: the I-frame encoder, or the caption decoder.

The short answer: **upgrade the encoder, leave the decoder alone.**

> **This is a derivative work, not a competitor.** All results below are measured against *our own reproduction* of CoCap under identical settings, on a 5,000-clip subset of VATEX and a reduced sampling budget. They are **not** comparable to the numbers in the CoCap paper and make no claim to beat them. See [Caveats](#caveats-read-before-citing).

---

## Results

VATEX subset (4,999 train / 1,000 val), 12 epochs, effective batch 12, single seed. Best epoch selected by validation CIDEr.

| I-frame encoder | Decoder | BLEU-4 | METEOR | ROUGE-L | **CIDEr** | Best epoch | CIDEr @ final epoch |
|---|---|---:|---:|---:|---:|:---:|---:|
| CLIP ViT-B/16 | BERT-style (CoCap) | 29.67 | 23.40 | 48.92 | 54.85 | 11 | 54.85 |
| **SigLIP2** | **BERT-style** | **31.01** | **24.06** | **49.43** | **59.33** | 9 | **59.11** |
| SigLIP2 | GPT-2 | 29.71 | 23.63 | 48.75 | 56.56 | 2 | 45.30 |

**Encoder swap (row 1 to 2):** +4.48 CIDEr, and it holds at the final epoch rather than depending on epoch selection.

**Decoder swap (row 2 to 3):** worse on every metric, and the gap is worse than it looks. GPT-2 peaks at **epoch 2** and then loses 11.3 CIDEr over the following nine epochs while its training loss falls 5.7x. That is textbook overfitting: a 124M-parameter pretrained decoder is too much capacity for a 5k-clip training set.

### Inference latency

Median of 50 runs, batch size 1, RTX 4070 Laptop, synthetic tensors (no I/O), milliseconds:

| Variant | Visual encode | Greedy decode | Total |
|---|---:|---:|---:|
| CLIP + BERT-style | 50.6 | 184.3 | 249.2 |
| **SigLIP2 + BERT-style** | 50.6 | 173.4 | **222.9** |
| SigLIP2 + GPT-2 | 55.0 | 400.1 | 458.7 |

The encoder upgrade is **free** — SigLIP2-base and CLIP ViT-B/16 are the same shape, so visual encoding is identical to the tenth of a millisecond. The decoder swap costs **1.84x end-to-end**, because the caption head goes from 2 layers to 12 and greedy decoding pays that cost once per generated token.

So the recommended variant is both more accurate *and* slightly faster than the baseline, which matters for a method whose entire premise is speed.

---

## Caveats (read before citing)

These are stated up front rather than buried, because several of them would change how you read the table.

1. **Not comparable to published CoCap numbers.** This trains on 4,999 VATEX clips, not the full ~26k, at a reduced sampling budget (5 GOPs x 16 B/P frames instead of 8 x 59), and evaluates on a 1,000-clip subset. Against CoCap's published VATEX result the reproduction lands mixed rather than uniformly below — CIDEr 54.85 vs 52.7 and METEOR 23.40 vs 23.2 sit above it, BLEU-4 29.67 vs 31.4 and ROUGE-L 48.92 vs 49.4 below — which reflects the evaluation subset, not an improvement. Only the *within-table* comparisons are meaningful.
2. **Single seed per variant.** No variance estimate. The +4.48 CIDEr encoder gain is large and consistent across all four metrics and across epochs, so it is unlikely to be noise — but it is not a significance test.
3. **The baseline's seeding is inconsistent.** Epochs 0–7 of the baseline ran before seeding was pinned; epochs 8–11 and both SigLIP2 variants ran under seed 42. The baseline was not re-run. This is a real limitation of the comparison, disclosed rather than smoothed over.
4. **"Best epoch" is selection-inflated.** Picking the top epoch by validation CIDEr and reporting that same validation CIDEr is optimistic. The final-epoch column is given so you can read the honest number; the encoder conclusion survives either way, and the GPT-2 conclusion gets *worse* under final-epoch.
5. **The variants did not all converge equally cleanly.** The baseline plateaus clearly (per-epoch CIDEr gains of +1.2, +0.7, +0.1, +0.2 over its last four epochs), but SigLIP2 gains +2.8 as late as epoch 8 before settling within 0.6 CIDEr across its final three. No run is truncated mid-improvement, but what residual under-training exists sits on the SigLIP2 side — which makes the +4.48 CIDEr encoder gain a conservative estimate rather than an inflated one.
6. **The GPT-2 result shows it is unsuited *at this data scale*** — it does not show GPT-2 is a bad decoder in general. Separating "wrong architecture" from "under-regularised / too little data" would need a frozen-decoder or larger-data run, which was not performed.

---

## What actually changed vs. upstream CoCap

| Component | CoCap | Here |
|---|---|---|
| I-frame encoder | CLIP ViT-B/16 | SigLIP2 `google/siglip2-base-patch16-224` |
| Caption decoder | BERT-style blocks, causal mask | GPT-2 prefix-LM (variant 3 only) |
| Motion vectors | `56x56x4` | `56x56x2` — B-frames excluded at re-encode |
| Motion / residual encoders | random init | unchanged (random init, no pretraining) |
| Dataset | MSRVTT / MSVD / VATEX | VATEX subset only |

SigLIP2 has **no CLS token** — it uses an attention-pooling (MAP) head. The port uses `pooler_output` (768-d) wherever CoCap used CLIP's CLS embedding, and `last_hidden_state` (196x768) as cross-attention memory for the action encoder. SigLIP also normalises with mean = std = 0.5 rather than the ImageNet statistics, which is set per-variant in the configs.

New code lives in:

- [`cocap/modules/siglip/`](cocap/modules/siglip/) — SigLIP2 I-frame encoder
- [`cocap/modules/gpt2/`](cocap/modules/gpt2/) — GPT-2 prefix-LM caption head
- [`cocap/data/tokenizers.py`](cocap/data/tokenizers.py) — CLIP BPE / GPT-2 tokenizer abstraction
- [`configs/exp/train/`](configs/exp/train/) — the three `vatex_subset_*` experiment configs
- [`tools/`](tools/) — dataset prep, pre-extraction, validation, latency benchmark

The AGDTR module is deliberately **not** included here; it is separate follow-up work.

---

## Reproducing

Full step-by-step instructions, including building the `cv_reader` H.264 parser under WSL2 and the FFmpeg/NumPy build patches it needs, are in **[docs/runbook.md](docs/runbook.md)**. Rented-GPU notes are in [docs/remote-gpu.md](docs/remote-gpu.md).

```bash
# 1. Dependencies (see docs/runbook.md for the cv_reader build, which is the fiddly part)
sudo apt update && sudo apt install default-jre -y   # pycocoevalcap needs a JRE
pip3 install -e .

# 2. CLIP weights (baseline encoder, and the word embeddings both decoders start from)
bash model_zoo/download_model.sh

# 3. Point at the dataset and build the metadata
export VATEX_SUBSET_ROOT=/path/to/vatex_subset
python tools/prepare_vatex_subset.py --help

# 4. Verify the compressed-domain pipeline before burning GPU hours on it
python tools/validate_data_pipeline.py
python tools/verify_motion_channels.py     # expects 2 channels, not 4

# 5. Pre-extract (decoding is the bottleneck otherwise)
python tools/pre_extract.py --video_dir "$VATEX_SUBSET_ROOT/train" --workers 6
python tools/pre_extract.py --video_dir "$VATEX_SUBSET_ROOT/val" --workers 6
```

Then train each variant. `budget=laptop_8gb` is what produced the numbers above; use `budget=paper` on a 24 GB or larger card.

```bash
python tools/train_net.py --config-name=exp/train/vatex_subset_baseline budget=laptop_8gb reader=pre_extract
python tools/train_net.py --config-name=exp/train/vatex_subset_siglip budget=laptop_8gb reader=pre_extract
python tools/train_net.py --config-name=exp/train/vatex_subset_siglip_gpt2 budget=laptop_8gb reader=pre_extract
```

Inspect results and latency:

```bash
python tools/show_metrics.py
python tools/benchmark_latency.py --variant siglip
```

Tests (no GPU and no `cv_reader` required — the compressed-domain tensors are synthesised):

```bash
pytest test/
```

---

## Credit

This repository builds directly on **[yaojie-shen/CoCap](https://github.com/yaojie-shen/CoCap)**. The compressed-video transformer, action encoder, training loop and evaluation harness are their work, and the upstream commit history is preserved here. Their original documentation and setup instructions live in the [upstream repository](https://github.com/yaojie-shen/CoCap).

```bibtex
@inproceedings{shen2023accurate,
  title     = {Accurate and Fast Compressed Video Captioning},
  author    = {Shen, Yaojie and Gu, Xin and Xu, Kai and Fan, Heng and Wen, Longyin and Zhang, Libo},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023}
}
```

Ablation study and the SigLIP2 / GPT-2 ports by **Ashim Nepal**, Department of Computer Engineering, Pulchowk Campus, IOE, Tribhuvan University.

Licensed under the same terms as upstream — see [LICENSE](LICENSE).
