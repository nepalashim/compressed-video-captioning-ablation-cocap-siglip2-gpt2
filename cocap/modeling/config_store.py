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


def train(
        model: pl.LightningModule,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        trainer: pl.Trainer,
        matmul_precision: str = "high",
):
    """
    :param matmul_precision: float32 matmul precision. "high" lets Ampere/Ada cards use their
        Tensor Cores for fp32 matmuls, which is a substantial speedup for a small precision
        cost. Set "highest" to disable it. Keep this the same across every run being compared,
        since it does perturb numerics.
    """
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)
        logger.info("float32 matmul precision: %s", matmul_precision)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


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
