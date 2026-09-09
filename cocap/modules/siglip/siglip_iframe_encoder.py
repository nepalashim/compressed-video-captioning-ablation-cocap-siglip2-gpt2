# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : siglip_iframe_encoder.py
"""
SigLIP2 I-frame encoder — drop-in replacement for the CLIP-based
:class:`~cocap.modules.compressed_video.compressed_video_transformer.IFrameEncoder`.

The two encoders differ in one structural way that matters here: CLIP's ViT prepends a ``[CLS]``
token and CoCap uses its projected output as the per-GOP context feature, whereas SigLIP has **no
CLS token**. SigLIP instead ends in a ``MultiheadAttentionPoolingHead`` (MAP head): a single
learnable *probe* vector attends over the patch tokens, and its output is ``pooler_output``. That
pooled vector is the natural stand-in for CLIP's CLS feature, and ``last_hidden_state`` (already
post-layernormed) supplies the patch tokens that the action encoder cross-attends into.

Checkpoint note: the fixed-resolution SigLIP2 checkpoints (``google/siglip2-base-patch16-224``
and friends) declare ``model_type: "siglip"``, so they load through ``SiglipVisionModel`` whose
forward takes plain ``pixel_values``. ``Siglip2VisionModel`` is only for the **NaFlex** variants
and additionally requires ``pixel_attention_mask`` and ``spatial_shapes``; it is not used here.

Preprocessing differs too — SigLIP normalizes with mean = std = 0.5 rather than the ImageNet
statistics the CLIP baseline uses. See ``normalize_mean``/``normalize_std`` on the dataset.
"""

__all__ = [
    "SiglipIFrameEncoder",
    "siglip_iframe_encoder_cfg",
    "siglip_iframe_encoder_pretrained_cfg",
]

import logging
from typing import Tuple

import torch
import torch.nn as nn
from hydra_zen import builds

logger = logging.getLogger(__name__)

#: fixed-resolution SigLIP2 checkpoint used by default (768-wide, 12 layers, 14x14 = 196 patches)
DEFAULT_SIGLIP2 = "google/siglip2-base-patch16-224"


class SiglipIFrameEncoder(nn.Module):
    """Wraps ``transformers.SiglipVisionModel`` behind ``IFrameEncoder``'s calling convention."""

    def __init__(
            self,
            pretrained_model_name_or_path: str = DEFAULT_SIGLIP2,
            output_dim: int = None,
            freeze: bool = False,
            gradient_checkpointing: bool = False,
    ):
        """
        :param pretrained_model_name_or_path: a fixed-resolution SigLIP/SigLIP2 checkpoint
        :param output_dim: optional projection width. ``None`` keeps the backbone's hidden size,
            which avoids an untrained bottleneck between the backbone and the action encoder.
        :param freeze: freeze the backbone; useful on small training sets where the 5k clips are
            not enough to fine-tune a 12-layer ViT without overfitting
        :param gradient_checkpointing: trade compute for activation memory in the backbone
        """
        super().__init__()
        from transformers import SiglipVisionModel

        self.model = SiglipVisionModel.from_pretrained(pretrained_model_name_or_path)
        cfg = self.model.config
        self.hidden_size = cfg.hidden_size
        self.input_resolution = cfg.image_size
        self.patch_size = cfg.patch_size
        self.output_dim = output_dim or cfg.hidden_size

        if self.output_dim != self.hidden_size:
            self.proj = nn.Linear(self.hidden_size, self.output_dim, bias=False)
        else:
            self.proj = None

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.freeze = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

        logger.info(
            "SiglipIFrameEncoder: %s (hidden=%d, res=%d, patch=%d, tokens=%d, frozen=%s)",
            pretrained_model_name_or_path, self.hidden_size, self.input_resolution,
            self.patch_size, (self.input_resolution // self.patch_size) ** 2, freeze,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:  # a frozen backbone stays in eval mode so dropout/LN behave consistently
            self.model.eval()
        return self

    def forward(self, x: torch.Tensor, output_all_features: bool = False,
                output_attention_map: bool = False):
        """
        :param x: ``(batch, 3, H, W)`` I-frames, normalized with SigLIP statistics
        :return: ``(context_feature[, patch_tokens][, attention_map])``, matching
            ``IFrameEncoder.forward`` so the two are interchangeable
        """
        out = self.model(pixel_values=x)

        ctx = out.pooler_output                 # (batch, hidden) — MAP head, replaces CLS
        outputs = (ctx if self.proj is None else self.proj(ctx),)

        if output_all_features:
            hidden = out.last_hidden_state      # (batch, n_patches, hidden), post-layernormed
            outputs += (hidden if self.proj is None else self.proj(hidden),)
        if output_attention_map:
            # SigLIP via transformers does not surface per-layer attention maps; the maps are
            # only used for visualization and are never consumed by the caption head.
            outputs += (None,)
        return outputs

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str = DEFAULT_SIGLIP2,
            output_dim: int = None,
            freeze: bool = False,
            gradient_checkpointing: bool = False,
    ) -> Tuple["SiglipIFrameEncoder", int, int, int]:
        """Mirror ``IFrameEncoder.from_pretrained``'s return contract.

        :return: ``(encoder, image_resolution, vision_width, embed_dim)``
        """
        encoder = cls(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            output_dim=output_dim,
            freeze=freeze,
            gradient_checkpointing=gradient_checkpointing,
        )
        return encoder, encoder.input_resolution, encoder.hidden_size, encoder.output_dim


# Build configs for organizing modules with hydra
siglip_iframe_encoder_cfg = builds(SiglipIFrameEncoder, populate_full_signature=True)
siglip_iframe_encoder_pretrained_cfg = builds(
    SiglipIFrameEncoder.from_pretrained, populate_full_signature=True
)
