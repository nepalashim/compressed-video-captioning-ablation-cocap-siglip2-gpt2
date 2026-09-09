# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : gpt2_caption_head.py
"""
GPT-2 caption head — drop-in replacement for
:class:`~cocap.modules.compressed_video.compressed_video_captioner.CaptionHead`.

The baseline head is a 2-layer BERT-style block that concatenates the visual tokens with the
caption tokens and applies a shifted causal mask, so visual tokens are visible to every text
step while text stays left-to-right. GPT-2 already has exactly that attention pattern, so the
swap is a **prefix-LM**: project the visual tokens into GPT-2's embedding space, prepend them to
the caption's word embeddings, and hand the result to GPT-2 as ``inputs_embeds``.

Interface parity with ``CaptionHead`` is deliberate — ``forward(visual_output, input_ids,
input_mask) -> prediction_scores`` and a ``cap_config`` carrying ``max_t_len`` / ``BOS_id`` /
``EOS_id`` / ``PAD_id`` — so ``CompressedVideoCaptioner`` and ``CoCapLM.validation_step`` work
against either head unchanged.

Position information comes from GPT-2's own learned positional embeddings, which cover the whole
concatenated sequence. Only a *type* embedding is added on top, distinguishing context tokens
from action tokens; it is zero-initialized so the head starts as a faithful GPT-2.
"""

__all__ = [
    "GPT2CaptionHead",
    "gpt2_caption_head_cfg",
]

import logging
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from hydra_zen import builds
from torch import Tensor

logger = logging.getLogger(__name__)


class GPT2CaptionHead(nn.Module):

    #: token type ids for the visual prefix
    TYPE_CONTEXT = 0
    TYPE_ACTION = 1

    def __init__(
            self,
            pretrained_model_name_or_path: str = "gpt2",
            visual_feature_size: int = 768,
            max_v_len: int = 16,
            max_t_len: int = 32,
            freeze: bool = False,
            gradient_checkpointing: bool = False,
            verbose: Optional[Union[int, bool]] = False,
    ):
        """
        :param visual_feature_size: width of ``feature_context`` / ``feature_action``
        :param max_v_len: number of visual tokens, i.e. ``2 * num_gop``
        :param max_t_len: caption length in tokens; must match the dataset's ``max_words``
        :param freeze: freeze the GPT-2 body (the projection and type embedding still train)
        """
        super().__init__()
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.gpt2 = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
        d_model = self.gpt2.config.n_embd
        tok = GPT2TokenizerFast.from_pretrained(pretrained_model_name_or_path)

        if max_v_len + max_t_len > self.gpt2.config.n_positions:
            raise ValueError(
                f"visual prefix ({max_v_len}) + caption ({max_t_len}) exceeds GPT-2's context "
                f"window ({self.gpt2.config.n_positions})"
            )

        # mirrors CaptionHead.cap_config so the Lightning module and the captioner are agnostic
        # to which head is in use
        self.cap_config = edict(
            max_v_len=max_v_len,
            max_t_len=max_t_len,
            hidden_size=d_model,
            video_feature_size=visual_feature_size,
            vocab_size=len(tok),                  # 50257
            BOS_id=tok.eos_token_id,              # GPT-2 has no BOS; <|endoftext|> starts decoding
            EOS_id=tok.eos_token_id,
            PAD_id=tok.eos_token_id,
            ignore_index=-100,
        )
        logger.debug("GPT-2 caption head configuration: %s", self.cap_config)

        self.visual_proj = nn.Sequential(
            nn.LayerNorm(visual_feature_size),
            nn.Linear(visual_feature_size, d_model),
        )
        self.visual_type_embedding = nn.Embedding(2, d_model)
        nn.init.zeros_(self.visual_type_embedding.weight)  # start as a no-op

        if gradient_checkpointing:
            self.gpt2.gradient_checkpointing_enable()
        self.freeze = freeze
        if freeze:
            for p in self.gpt2.parameters():
                p.requires_grad_(False)

        # debug output cfgs, matching CaptionHead
        if verbose:
            self.log_interval = 1 if isinstance(verbose, bool) else int(verbose)
        else:
            self.log_interval = float("inf")
        self.step_counter = 1

        logger.info("GPT2CaptionHead: %s (d_model=%d, visual=%d->%d, v_len=%d, t_len=%d, frozen=%s)",
                    pretrained_model_name_or_path, d_model, visual_feature_size, d_model,
                    max_v_len, max_t_len, freeze)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.gpt2.eval()
        return self

    def _build_visual_prefix(self, visual_output: Dict[str, Tensor]) -> Tensor:
        """``feature_context`` and ``feature_action`` -> ``(batch, max_v_len, d_model)``."""
        ctx, act = visual_output["feature_context"], visual_output["feature_action"]
        v = torch.cat([ctx, act], dim=1)  # (B, 2N, visual_dim)

        type_ids = torch.cat([
            torch.full((ctx.size(0), ctx.size(1)), self.TYPE_CONTEXT,
                       dtype=torch.long, device=ctx.device),
            torch.full((act.size(0), act.size(1)), self.TYPE_ACTION,
                       dtype=torch.long, device=act.device),
        ], dim=1)

        return self.visual_proj(v) + self.visual_type_embedding(type_ids)

    def forward(self, visual_output, input_ids, input_mask):
        assert input_ids.size(1) == self.cap_config.max_t_len, \
            f"{input_ids.size(1)} vs {self.cap_config.max_t_len}"

        visual_embeds = self._build_visual_prefix(visual_output)
        n_visual = visual_embeds.size(1)
        assert n_visual == self.cap_config.max_v_len, \
            f"got {n_visual} visual tokens, head was built for {self.cap_config.max_v_len}"

        text_embeds = self.gpt2.transformer.wte(input_ids.long())
        inputs_embeds = torch.cat([visual_embeds, text_embeds.to(visual_embeds.dtype)], dim=1)

        attention_mask = torch.cat([
            torch.ones(visual_embeds.size()[:2], dtype=input_mask.dtype, device=input_mask.device),
            input_mask,
        ], dim=1)

        # GPT-2 is causal by default: every text step sees the whole visual prefix and only the
        # text tokens before it, which is the information flow the BERT head built by hand.
        out = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        prediction_scores = out.logits[:, -self.cap_config.max_t_len:]

        if self.step_counter % self.log_interval == 0:
            logger.debug("GT  : %s", self.ids2text(input_ids))
            logger.debug("Pred: %s", self.probability2text(prediction_scores))
        self.step_counter += 1
        return prediction_scores

    @staticmethod
    @torch.no_grad()
    def probability2text(predict_scores=None):
        return GPT2CaptionHead.ids2text(predict_scores.max(-1)[1])

    @staticmethod
    @torch.no_grad()
    def ids2text(gt_ids):
        from cocap.data.tokenizers import build_tokenizer

        tokenizer = build_tokenizer("gpt2")
        if isinstance(gt_ids, Tensor):
            gt_ids = gt_ids.detach().cpu().numpy()
        assert 0 < len(gt_ids.shape) <= 2, f"expected a 1 or 2 dim array/tensor, got {gt_ids.shape}"
        if len(gt_ids.shape) == 1:
            return tokenizer.decode(gt_ids.tolist())
        return [tokenizer.decode(ids) for ids in gt_ids.tolist()]


# Build configs for organizing modules with hydra
gpt2_caption_head_cfg = builds(GPT2CaptionHead, populate_full_signature=True)
