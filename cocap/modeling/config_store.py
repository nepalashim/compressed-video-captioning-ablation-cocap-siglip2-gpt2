# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : config_store.py
"""
Hydra config registration, shared by every entry point.

Kept in the library rather than in ``tools/train_net.py`` so that anything needing to build a
model exactly as training would - the latency benchmark, evaluation, notebooks - composes the
same experiment configs instead of reconstructing the model by hand.
"""

__all__ = ["train", "register_configs"]

import logging

import pytorch_lightning as pl
import torch
from hydra_zen import builds
from omegaconf import MISSING
from torch.utils.data import DataLoader

from .lm_cocap import register_model_configs

logger = logging.getLogger(__name__)


def seed_from_config(cfg):
    """Seed every RNG *before* hydra instantiates anything.

    Registered as ``zen``'s ``pre_call``, because by the time ``train`` runs the model has
    already been built and its randomly-initialized weights drawn. Seeding there would cover
    data order and dropout but not initialization, which is the opposite of reproducible.

    ``workers=True`` extends the seed to dataloader workers, which matters here: the per-GOP
    B/P frame sampling and the choice of which of a video's 10 captions to use both happen
    inside the workers.
    """
    seed = getattr(cfg, "seed", None)
    if seed is None:
        logger.warning("no seed configured; this run is not reproducible")
        return
    pl.seed_everything(int(seed), workers=True)
    logger.info("seeded everything with %d (workers included)", int(seed))


def train(
        model: pl.LightningModule,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        trainer: pl.Trainer,
        matmul_precision: str = "high",
        seed: int = 42,
        ckpt_path: str = None,
):
    """
    :param matmul_precision: float32 matmul precision. "high" lets Ampere/Ada cards use their
        Tensor Cores for fp32 matmuls, which is a substantial speedup for a small precision
        cost. Set "highest" to disable it. Keep this the same across every run being compared,
        since it does perturb numerics.
    :param seed: consumed by :func:`seed_from_config` before instantiation; declared here so it
        appears in the config and is recorded with the run. Vary it to measure run-to-run
        spread, which is what decides whether a difference between variants is real.
    :param ckpt_path: resume from this checkpoint, restoring weights, optimizer state, LR
        schedule and epoch counter. This is an argument to ``fit``, not to the Trainer, so
        ``trainer.ckpt_path=...`` does not work - pass it at the top level::

            ckpt_path=logs/<run>/lightning_logs/version_N/checkpoints/epoch07.ckpt
    """
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)
        logger.info("float32 matmul precision: %s", matmul_precision)
    logger.info("seed: %s", seed)
    if ckpt_path:
        logger.info("resuming from %s", ckpt_path)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader,
                ckpt_path=ckpt_path)


def register_configs(store):
    """Register the ``model`` config group and the top-level ``train`` config."""
    register_model_configs(store)

    # `model` stays MISSING rather than defaulting to one variant's generated dataclass: a typed
    # node only accepts that exact dataclass, so selecting another variant through the defaults
    # list would fail to merge. Every experiment config picks one with `- /model@model: <name>`.
    store(
        train,
        model=MISSING,
        train_dataloader=builds(
            DataLoader,
            dataset=MISSING,
            populate_full_signature=True,
        ),
        val_dataloader=builds(
            DataLoader,
            dataset=MISSING,
            populate_full_signature=True,
        ),
        trainer=builds(pl.Trainer, populate_full_signature=True),
        populate_full_signature=True,
        name="train",
    )
    return store
