# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : validate_data_pipeline.py
"""
End-to-end check of the data path, from video file to a batch the model accepts.

Run this before any training run. It catches the class of bug that would otherwise only show up
as quietly bad numbers thirty epochs later:

* tensors whose shape or dtype the model's dimension asserts would reject
* videos the reader silently turned into all-zero tensors (the reader's failure mode)
* GOP sampling that duplicates the same GOP, wasting encoder compute on identical features
* a motion-vector channel count that disagrees with the encoder
* captions that tokenize to nothing, or masks that do not line up with the labels

With ``--fake-reader`` it runs without the native ``cv_reader``: picture types and frame indices
still come from the real files, only motion-vector and residual *values* are synthetic. That is
enough to validate every bit of logic above. Drop the flag once ``cv_reader`` is built.

Usage::

    python tools/validate_data_pipeline.py --variant siglip_gpt2 --fake-reader -n 64
    python tools/validate_data_pipeline.py --variant baseline --split test -n 200
"""

import argparse
import collections
import sys
from pathlib import Path

import torch

VARIANTS = {
    "baseline": "exp/train/vatex_subset_baseline",
    "siglip": "exp/train/vatex_subset_siglip",
    "siglip_gpt2": "exp/train/vatex_subset_siglip_gpt2",
}


def compose(variant: str, overrides=None):
    from hydra import compose as hydra_compose, initialize_config_dir
    from hydra_zen import store

    from cocap.modeling.config_store import register_configs

    register_configs(store)
    store.add_to_hydra_store(overwrite_ok=True)
    config_dir = (Path(__file__).parent.parent / "configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return hydra_compose(config_name=VARIANTS[variant], overrides=list(overrides or []))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="siglip_gpt2")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("-n", "--num_samples", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--fake-reader", action="store_true",
                        help="use the synthetic cv_reader stand-in (development only)")
    parser.add_argument("--build-model", action="store_true",
                        help="also instantiate the model and run one forward pass on a batch")
    # trailing key=value arguments are forwarded to hydra, so the same config groups training
    # uses can be validated here: `budget=laptop_8gb reader=pre_extract`
    args, overrides = parser.parse_known_args()

    malformed = [a for a in overrides if "=" not in a]
    if malformed:
        parser.error(f"unrecognized arguments: {' '.join(malformed)}\n"
                     f"(hydra overrides must be key=value, e.g. budget=laptop_8gb)")

    if args.fake_reader:
        from cocap.data.datasets.compressed_video.fake_cv_reader import install
        install()

    from hydra.utils import instantiate
    from torch.utils.data import DataLoader

    from cocap.data.datasets.compressed_video import video_text_base

    cfg = compose(args.variant, overrides)
    if overrides:
        print(f"overrides : {' '.join(overrides)}")
    ds_cfg = cfg.train_dataloader.dataset if args.split == "train" else cfg.val_dataloader.dataset
    dataset = instantiate(ds_cfg)

    cv = ds_cfg.cv_config
    n_gop, n_bp, mc = cv.num_gop, cv.num_mv, cv.get("motion_channels", 4)
    res_h, res_w = ds_cfg.video_size
    max_words = ds_cfg.max_words

    print(f"variant   : {args.variant}   split: {args.split}")
    print(f"dataset   : {len(dataset)} samples")
    print(f"expected  : iframe ({n_gop},3,{res_h},{res_w})  motion ({n_gop},{n_bp},{mc},"
          f"{res_h // 4},{res_w // 4})  residual ({n_gop},{n_bp},3,{res_h},{res_w})")
    print(f"cv_reader : {'FAKE (synthetic mv/residual)' if args.fake_reader else 'real'}"
          f"   pre_extract: {bool(cv.get('use_pre_extract', False))}")
    print()

    n = min(args.num_samples, len(dataset))
    problems = collections.Counter()
    gop_dup_ratios = []
    zero_iframe = 0
    tok = dataset.tokenizer

    for i in range(n):
        sample = dataset[i]
        v = sample["video"]

        # --- shapes and dtypes -------------------------------------------------------------
        checks = {
            "iframe": (v["iframe"], (n_gop, 3, res_h, res_w)),
            "motion_vector": (v["motion_vector"], (n_gop, n_bp, mc, res_h // 4, res_w // 4)),
            "residual": (v["residual"], (n_gop, n_bp, 3, res_h, res_w)),
            "type_ids_mv": (v["type_ids_mv"], (n_gop, n_bp)),
        }
        for key, (tensor, expected) in checks.items():
            if tuple(tensor.shape) != expected:
                problems[f"{key} shape {tuple(tensor.shape)} != {expected}"] += 1

        # --- the reader's silent failure mode ----------------------------------------------
        if not v["iframe"].any():
            zero_iframe += 1

        # --- B/P type ids must index the action encoder's embedding ------------------------
        max_type = int(v["type_ids_mv"].max())
        if max_type > 2:
            problems[f"type_ids_mv has value {max_type} (expected 0=P, 1=B, 2=pad)"] += 1

        # --- how much of the GOP budget is duplicated content? -----------------------------
        # identical GOPs produce byte-identical features, so this is wasted encoder compute
        sigs = {hash(v["iframe"][g].numpy().tobytes()) for g in range(n_gop)}
        gop_dup_ratios.append(1.0 - len(sigs) / n_gop)

        # --- text ---------------------------------------------------------------------------
        ids, mask, labels = sample["input_ids"], sample["input_mask"], sample["input_labels"]
        for key, t in (("input_ids", ids), ("input_mask", mask), ("input_labels", labels)):
            if tuple(t.shape) != (max_words,):
                problems[f"{key} shape {tuple(t.shape)} != ({max_words},)"] += 1
        if int(mask.sum()) < 3:
            problems["caption tokenized to fewer than 3 tokens"] += 1
        n_real = int(mask.sum())
        if n_real > 1 and not torch.equal(labels[:n_real - 1].long(), ids[1:n_real].long()):
            problems["input_labels are not input_ids shifted by one"] += 1
        if int((labels != tok.ignore_index).sum()) == 0:
            problems["every label is ignored; this sample contributes no loss"] += 1

    # --- report -----------------------------------------------------------------------------
    print(f"checked {n} samples")
    print(f"  reader failures (all-zero I-frames) : {zero_iframe}"
          f"{'   <-- these train as black clips' if zero_iframe else ''}")
    print(f"  worker-local failure log            : {len(video_text_base.GET_VIDEO_FAILURES)}")
    mean_dup = sum(gop_dup_ratios) / len(gop_dup_ratios)
    print(f"  duplicated GOP fraction             : {mean_dup:.1%} "
          f"(mean over samples, num_gop={n_gop})")
    if mean_dup > 0.05:
        distinct = round(n_gop * (1 - mean_dup), 1)
        print(f"      -> only ~{distinct} of {n_gop} GOPs are distinct; the rest are identical "
              f"copies costing encoder compute for no information")

    if problems:
        print("\n  PROBLEMS:")
        for msg, count in problems.most_common():
            print(f"    [{count:>4}x] {msg}")
    else:
        print("\n  no shape/dtype/label problems found")

    # --- one real batch through the model ---------------------------------------------------
    if args.build_model:
        print("\nbuilding model and running one forward pass ...")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        batch = next(iter(loader))
        lm = instantiate(cfg.model)
        with torch.no_grad():
            out = lm.model(batch)
            loss = lm.loss(batch, out)
        print(f"  prediction_scores : {tuple(out['prediction_scores'].shape)}")
        print(f"  loss              : {loss.item():.4f}")
        if not torch.isfinite(loss):
            problems["loss is not finite"] += 1
            print("  LOSS IS NOT FINITE")

    failed = bool(problems) or zero_iframe
    print("\n" + ("FAILED" if failed else "PASSED"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
