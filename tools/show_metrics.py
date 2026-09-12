# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : show_metrics.py
"""
Print per-epoch validation metrics from a run's TensorBoard logs.

The caption metrics are logged rather than printed, so the terminal only shows pycocoevalcap's
internal BLEU counters. This reads the event files and lays the metrics out per epoch, which is
what you need in order to see where CIDEr peaks and whether it is still climbing at the end of
training.

Usage::

    python tools/show_metrics.py logs/vatex_subset_baseline_seed42
    python tools/show_metrics.py logs/*                     # compare runs side by side
"""

import argparse
import glob
import os
import sys

METRICS = ["CIDEr", "Bleu_4", "METEOR", "ROUGE_L"]


def read_scalars(log_dir):
    """Collect scalar series from every event file under ``log_dir``."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    series = {}
    event_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not event_files:
        return series

    for path in sorted(event_files):
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            series.setdefault(tag, []).extend(
                (event.step, event.value) for event in acc.Scalars(tag)
            )
    # a resumed run can write the same step twice; keep the last value for each
    return {tag: dict(sorted(points)) for tag, points in series.items()}


def report(log_dir):
    series = read_scalars(log_dir)
    if not series:
        print(f"{log_dir}: no event files found")
        return

    present = [m for m in METRICS if m in series]
    if not present:
        print(f"{log_dir}: no caption metrics logged yet "
              f"(available tags: {', '.join(sorted(series)[:8])})")
        return

    # metrics are logged on_epoch, so their steps line up across metrics
    steps = sorted({s for m in present for s in series[m]})

    print(f"=== {log_dir}")
    header = f"{'epoch':>6}" + "".join(f"{m:>10}" for m in present)
    if "loss_epoch" in series:
        header += f"{'loss':>12}"
    print(header)

    best_epoch, best_cider = None, float("-inf")
    for i, step in enumerate(steps):
        row = f"{i:>6}"
        for m in present:
            v = series[m].get(step)
            # BLEU/METEOR/ROUGE/CIDEr are logged in [0,1]; report the x100 convention papers use
            row += f"{v * 100:>10.2f}" if v is not None else f"{'-':>10}"
        if "loss_epoch" in series:
            loss = series["loss_epoch"].get(step)
            row += f"{loss:>12.2f}" if loss is not None else f"{'-':>12}"
        print(row)

        cider = series.get("CIDEr", {}).get(step)
        if cider is not None and cider > best_cider:
            best_epoch, best_cider = i, cider

    if best_epoch is not None:
        print(f"\n  best CIDEr {best_cider * 100:.2f} at epoch {best_epoch} of {len(steps) - 1}")
        if best_epoch == len(steps) - 1 and len(steps) > 1:
            print("  -> peaked on the final epoch, so it may still be improving; "
                  "more epochs could be worth it")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_dirs", nargs="+", help="run directories under logs/")
    args = parser.parse_args()

    dirs = [d for d in args.log_dirs if os.path.isdir(d)]
    if not dirs:
        sys.exit(f"no directories among: {' '.join(args.log_dirs)}")
    for d in dirs:
        report(d)


if __name__ == "__main__":
    main()
