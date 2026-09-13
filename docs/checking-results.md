# Checking training results

Quick reference for inspecting a run without disturbing it. Everything here is read-only and
safe to run while training is in progress.

---

## Open a shell and show the metrics

From a **new** Windows CMD or PowerShell window:

```
wsl -d Ubuntu-24.04
```

Then inside Linux:

```bash
cd ~/CoCap
source .venv-wsl/bin/activate     # prompt must show (.venv-wsl)
ls -d logs/*/                     # which runs exist
python tools/show_metrics.py logs/*/
```

`show_metrics.py` reads the TensorBoard event files and prints CIDEr, BLEU-4, METEOR, ROUGE-L
and training loss per epoch, naming the best epoch by CIDEr. Metrics are logged rather than
printed during training, so this is the only way to see them short of opening TensorBoard.

---

## Why a resumed run splits across two directories

Log directories carry the seed (`logs/<variant>_seed<N>/`). A run started before seeding was
added, or resumed without `trainer.default_root_dir` pointing back at the original directory,
lands somewhere else. The baseline is split exactly this way:

| directory | epochs |
|---|---|
| `logs/vatex_subset_baseline/` | 0-7 (original run) |
| `logs/vatex_subset_baseline_seed42/` | 8-11 (resumed) |

The second table's `epoch` column **restarts at 0**, so read it with an offset: its epoch 0 is
really epoch 8. The `best CIDEr at epoch N` line is likewise per-directory, so compare the two
tables rather than trusting either in isolation.

`logs/*/` catches both. To keep a resumed run in one place, pass
`trainer.default_root_dir=./logs/<original dir>` when resuming.

---

## Read the generated captions

Worth doing at least once per run - metrics do not reveal degenerate repetition, and captions do.

```bash
python - <<'PY'
import json, glob
f = sorted(glob.glob('logs/*/caption_greedy_pred_validation_*.json'))[-1]
print(f)
for k, v in list(json.load(open(f))['results'].items())[:10]:
    print(f"{k}\n  pred: {v[0]['sentence']}\n  gt  : {v[0]['gt_sentence']}\n")
PY
```

---

## Checkpoints

```bash
find logs -name "*.ckpt" -exec ls -lh {} \;
```

Every epoch is saved (`epochNN.ckpt`), plus `best-epochNN-ciderX.XXXX.ckpt` tracking the peak
CIDEr. Each is about 2 GB. To resume:

```bash
python tools/train_net.py --config-name=exp/train/vatex_subset_baseline \
    budget=laptop_8gb reader=pre_extract \
    trainer.default_root_dir=./logs/vatex_subset_baseline \
    ckpt_path=logs/vatex_subset_baseline/lightning_logs/version_2/checkpoints/epoch07.ckpt
```

`ckpt_path` is an argument to `fit`, not to the Trainer - `trainer.ckpt_path=...` is silently
ignored.

---

## Watch a run in progress

```bash
tmux attach -t train        # Ctrl+B then D to detach, leaving it running
tmux ls                     # list sessions
nvidia-smi                  # VRAM, power draw, and the training process
```

Healthy on the RTX 4070 Laptop at `budget=laptop_8gb`: VRAM around 5 GB of 8188 MiB, power at
its cap, roughly 5 it/s, about 1 h 25 m per epoch.

If `tmux ls` reports no server, the WSL VM was shut down and the run is gone - resume from the
last checkpoint. `tmux` survives a terminal closing, but not WSL restarting.

---

## Inference latency

Speed is the property the compressed-domain approach exists for, so these belong alongside the
accuracy numbers, measured on the same GPU.

```bash
for v in baseline siglip siglip_gpt2; do
    python tools/benchmark_latency.py --variant $v --device cuda -n 50 budget=laptop_8gb
done
```

---

## Recorded: CoCap reproduction baseline

12 epochs, `budget=laptop_8gb reader=pre_extract`, VATEX 4,999/1,000 subset. Epochs 0-7
unseeded, 8-11 at seed 42.

| epoch | CIDEr | BLEU-4 | METEOR | ROUGE-L | loss |
|---|---|---|---|---|---|
| 0 | 11.64 | 15.03 | 16.08 | 40.01 | 217.32 |
| 1 | 22.04 | 20.74 | 19.14 | 44.43 | 140.18 |
| 2 | 31.35 | 23.83 | 20.55 | 46.29 | 112.62 |
| 3 | 41.29 | 26.95 | 21.70 | 47.89 | 103.01 |
| 4 | 43.82 | 27.10 | 22.19 | 47.87 | 96.68 |
| 5 | 46.98 | 28.55 | 22.62 | 48.47 | 92.08 |
| 6 | 50.21 | 28.86 | 22.83 | 48.86 | 88.53 |
| 7 | 52.68 | 30.03 | 22.90 | 48.93 | 85.65 |
| 8 | 53.89 | **30.28** | 23.24 | 49.02 | 83.31 |
| 9 | 54.55 | 29.54 | **23.30** | **49.05** | 81.33 |
| 10 | **54.65** | 29.70 | 23.26 | 48.77 | 79.59 |
| 11 | **54.85** | 29.67 | **23.40** | 48.92 | 78.11 |

**Converged.** Per-epoch CIDEr gains fell from +2.5 (epoch 7) to +1.2, +0.7, +0.1, +0.2 - so 12 epochs
is sufficient and the model is not under-trained. BLEU-4 peaked at epoch 8 and METEOR/ROUGE at
epoch 9, while CIDEr continued to creep up; best-checkpoint selection therefore depends on the
monitored metric, and we monitor CIDEr.
