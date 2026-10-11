# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : eval_net.py
"""
Evaluate a trained checkpoint on a split, without training.

``tools/train_net.py`` calls ``trainer.fit`` and nothing else, so until now the only way to get
caption metrics was as a side effect of a training run's validation epoch. Scoring a held-out
test split needs the same thing on its own.

This deliberately reuses the existing ``validation_step`` / ``on_validation_epoch_end`` hooks
via ``trainer.validate`` rather than implementing a second decode-and-score path. Greedy
decoding, the prediction dump and the ``pycocoevalcap`` scoring are then provably the same code
that produced the numbers reported for the validation split - there is no second implementation
to drift out of sync.

The config is composed through the same ``register_configs`` store the training entry point
uses, so the model is built exactly as training built it. That matters: the checkpoint only
carries weights (its ``hyper_parameters`` are empty), so the architecture comes entirely from the
config and a mismatched ``budget`` would silently evaluate a differently-shaped model.

Usage::

    python tools/eval_net.py --config-name=exp/train/vatex_subset_siglip \\
        budget=laptop_8gb reader=decode \\
        ckpt_path=logs/.../best-epoch09-cider0.5933.ckpt \\
        val_dataloader.dataset.video_root="$VATEX_SUBSET_ROOT/test" \\
        val_dataloader.dataset.metadata=./dataset/vatex_subset/vatex_testset_caption.json \\
        val_dataloader.dataset.id_to_file=./dataset/vatex_subset/id_to_file_test.json \\
        trainer.default_root_dir=./logs/testset_siglip

``budget`` must be the one the checkpoint was trained under. Leaving the ``val_dataloader``
overrides off evaluates the validation split instead, which is how to check that a checkpoint
still reproduces its published numbers.

The ``ModelCheckpoint`` and ``LearningRateMonitor`` callbacks inherited from the experiment
config are inert here - Lightning skips checkpoint saving unless the trainer is fitting, and the
LR monitor only hooks training - so they need no overriding.
"""

import json
import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra_zen import store, zen
from torch.utils.data import DataLoader

from cocap.modeling.config_store import register_configs, seed_from_config

logger = logging.getLogger(__name__)

METRIC_ORDER = ["Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]


def evaluate(
        model: pl.LightningModule,
        val_dataloader: DataLoader,
        trainer: pl.Trainer,
        ckpt_path: str = None,
        matmul_precision: str = "high",
        seed: int = 42,
):
    """Score one checkpoint over ``val_dataloader``'s split.

    The signature is a subset of :func:`cocap.modeling.config_store.train`, so the same
    experiment configs compose against it unchanged. ``train_dataloader`` is deliberately absent:
    ``zen`` only instantiates the config entries a parameter asks for, so leaving it out means
    the training split is never even opened.

    :param ckpt_path: the checkpoint to score. Required - evaluating freshly initialised weights
        is never what is wanted, so it is an error rather than a default.
    """
    if not ckpt_path:
        raise ValueError("ckpt_path is required; pass ckpt_path=logs/.../best-....ckpt")
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    if matmul_precision:
        # Must match training: it perturbs numerics, so a different setting would make the
        # reproduction check below fail for a reason that has nothing to do with the data.
        torch.set_float32_matmul_precision(matmul_precision)

    dataset = val_dataloader.dataset
    logger.info("checkpoint : %s", ckpt_path)
    logger.info("split      : %s from %s", dataset.split, dataset.video_root)
    logger.info("samples    : %d scored against %d references over %d clips",
                len(dataset), sum(len(v) for v in dataset.json_ref.values()),
                len(dataset.json_ref))
    logger.info("seed       : %s", seed)

    results = trainer.validate(model=model, dataloaders=val_dataloader, ckpt_path=ckpt_path)

    metrics = dict(results[0]) if results else {}
    ordered = {k: metrics[k] for k in METRIC_ORDER if k in metrics}
    ordered.update({k: v for k, v in metrics.items() if k not in ordered})

    print("\n" + "=" * 64)
    print(f"{os.path.basename(ckpt_path)}  on  {dataset.split}  "
          f"({len(dataset.json_ref)} clips)")
    print("=" * 64)
    for key in METRIC_ORDER:
        if key in metrics:
            print(f"  {key:10s} {100 * metrics[key]:6.2f}")
    for key, value in metrics.items():
        if key not in METRIC_ORDER:
            print(f"  {key:10s} {value:6.4f}")
    print("=" * 64)

    summary = {
        "checkpoint": ckpt_path,
        "split": dataset.split,
        "video_root": dataset.video_root,
        "clips": len(dataset.json_ref),
        "samples": len(dataset),
        "cv_config": {k: getattr(dataset.h265_cfg, k)
                      for k in ("num_gop", "num_mv", "num_res", "motion_channels",
                                "with_residual", "use_pre_extract", "sample")
                      if hasattr(dataset.h265_cfg, k)},
        "metrics": ordered,
    }
    out = os.path.join(trainer.default_root_dir, "eval_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {out}\n")
    return metrics


if __name__ == "__main__":
    register_configs(store)
    store.add_to_hydra_store()

    zen(evaluate, pre_call=seed_from_config).hydra_main(
        config_path=(Path(__file__).parent.parent / "configs").as_posix(),
        version_base=None,
    )
