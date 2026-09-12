# -*- coding: utf-8 -*-
# @Time    : 6/16/25
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : cocap.py
import copy
import logging
import os
from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from hydra_zen import builds
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import LambdaLR

from cocap.modules.bert import BertLayerNorm
from cocap.modules.compressed_video import (CompressedVideoCaptioner,
                                            compressed_video_captioner_pretrained_cfg,
                                            compressed_video_captioner_siglip_cfg,
                                            compressed_video_captioner_siglip_gpt2_cfg)
from .eval_captioning import evaluate
from .loss import LossBase, LabelSmoothingLoss, label_smoothing_loss_cfg
from .optimization import BertAdam
from ..utils.json import save_json
from ..utils.train_utils import gather_object_multiple_gpu, get_timestamp

logger = logging.getLogger(__name__)


def convert_ids_to_sentence(tokens, tokenizer: str = "clip"):
    """Decode generated ids into a caption using the configured text tokenizer."""
    from cocap.data.tokenizers import build_tokenizer

    return build_tokenizer(tokenizer).decode(tokens)


#: parameter prefixes carrying pretrained weights, per backbone/decoder choice. Anything matched
#: here is excluded from weight decay and trained at its group's (lower) learning rate.
CLIP_VISION_PREFIXES = [
    "compressed_video_transformer.rgb_encoder.conv1",
    "compressed_video_transformer.rgb_encoder.class_embedding",
    "compressed_video_transformer.rgb_encoder.positional_embedding",
    "compressed_video_transformer.rgb_encoder.ln_pre",
    "compressed_video_transformer.rgb_encoder.transformer",
    "compressed_video_transformer.rgb_encoder.ln_post",
    "compressed_video_transformer.rgb_encoder.proj",
]
SIGLIP_VISION_PREFIXES = [
    "compressed_video_transformer.rgb_encoder.model",
]
BERT_DECODER_PREFIXES = [
    "caption_head.cap_sa_decoder.word_embeddings",
    "caption_head.prediction_head.decoder",
]
GPT2_DECODER_PREFIXES = [
    "caption_head.gpt2",
]


class CoCapLM(pl.LightningModule):
    """CoCap Lightning Module"""

    def __init__(
            self,
            cocap_model: CompressedVideoCaptioner,
            loss: LossBase,
            lr: float = 1e-4,
            clip_lr: float = 1e-6,
            decoder_lr: float = None,
            warmup_ratio: float = 0.05,
            lr_decay_gamma: float = 0.95,
            tokenizer: str = "clip",
            vision_pretrained_prefixes: list = None,
            decoder_pretrained_prefixes: list = None,
    ):
        """
        :param lr: learning rate for randomly initialized parameters
        :param clip_lr: learning rate for the pretrained vision backbone
        :param decoder_lr: learning rate for pretrained decoder parameters; defaults to
            ``clip_lr``, which reproduces the baseline where both sat in one group. A pretrained
            LM decoder (GPT-2) usually wants something between ``clip_lr`` and ``lr``.
        :param tokenizer: name of the text tokenizer, must match the dataset and caption head
        :param vision_pretrained_prefixes: parameter prefixes of the pretrained vision backbone
        :param decoder_pretrained_prefixes: parameter prefixes of pretrained decoder weights
        """
        super().__init__()
        self.model = cocap_model
        self.loss = loss
        self.lr = lr
        self.clip_lr = clip_lr
        self.decoder_lr = clip_lr if decoder_lr is None else decoder_lr
        self.warmup_ratio = warmup_ratio
        self.lr_decay_gamma = lr_decay_gamma
        self.tokenizer_name = tokenizer
        self.vision_pretrained_prefixes = (
            list(CLIP_VISION_PREFIXES) if vision_pretrained_prefixes is None
            else list(vision_pretrained_prefixes)
        )
        self.decoder_pretrained_prefixes = (
            list(BERT_DECODER_PREFIXES) if decoder_pretrained_prefixes is None
            else list(decoder_pretrained_prefixes)
        )

        self.batch_res = None

    @property
    def total_steps(self):
        return self.trainer.estimated_stepping_batches

    @property
    def epoch_steps(self):
        return self.trainer.estimated_stepping_batches // self.trainer.max_epochs

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # based on:
        # https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136
        model = self.model

        decay = set()
        no_decay = set()

        vision_prefixes = self.vision_pretrained_prefixes
        decoder_prefixes = self.decoder_pretrained_prefixes
        pretrained_modules = vision_prefixes + decoder_prefixes
        whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention, nn.Conv2d)
        blacklist_weight_modules = (nn.LayerNorm, nn.BatchNorm2d, nn.Embedding, BertLayerNorm)
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if any(fpn.startswith(p_fpn) for p_fpn in pretrained_modules):  # pretrained
                    no_decay.add(fpn)
                elif pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("proj") or pn.endswith("projection"):
                    decay.add(fpn)
                elif fpn.endswith("embedding"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in model.named_parameters()}
        inter_params = decay & no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)

        # A swapped-in module may use layer types the rules above do not name (GPT-2's Conv1D,
        # for instance). Fall back on tensor rank rather than failing, so a new backbone or
        # decoder does not require editing this function.
        unclassified = param_dict.keys() - (decay | no_decay)
        if unclassified:
            logger.warning("Parameters not matched by the decay rules, classified by rank: %s",
                           "\n   " + "\n   ".join(sorted(unclassified)))
            for fpn in unclassified:
                (decay if param_dict[fpn].dim() >= 2 else no_decay).add(fpn)

        def _matches(pn, prefixes):
            return any(pn.startswith(p) for p in prefixes)

        vision_no_decay = [pn for pn in sorted(no_decay) if _matches(pn, vision_prefixes)]
        decoder_no_decay = [pn for pn in sorted(no_decay) if _matches(pn, decoder_prefixes)]
        other_no_decay = [pn for pn in sorted(no_decay) if not _matches(pn, pretrained_modules)]

        logger.debug("Parameter group decay_param: %s",
                     "\n   " + "\n   ".join(sorted(decay)))
        logger.debug("Parameter group no_decay_vision_pretrained_param: %s",
                     "\n   " + "\n   ".join(vision_no_decay))
        logger.debug("Parameter group no_decay_decoder_pretrained_param: %s",
                     "\n   " + "\n   ".join(decoder_no_decay))
        logger.debug("Parameter group no_decay_not_pretrained_param: %s",
                     "\n   " + "\n   ".join(other_no_decay))
        logger.info("Parameter groups: decay=%d, vision_pretrained=%d (lr=%g), "
                    "decoder_pretrained=%d (lr=%g), other_no_decay=%d",
                    len(decay), len(vision_no_decay), self.clip_lr,
                    len(decoder_no_decay), self.decoder_lr, len(other_no_decay))

        def _params(names):
            # frozen modules contribute no gradients; keep them out of the optimizer entirely
            return [param_dict[pn] for pn in names if param_dict[pn].requires_grad]

        optimizer_grouped_parameters = [
            {"params": _params(sorted(decay))},
            {"params": _params(vision_no_decay), "weight_decay": 0.0, "lr": self.clip_lr},
            {"params": _params(decoder_no_decay), "weight_decay": 0.0, "lr": self.decoder_lr},
            {"params": _params(other_no_decay), "weight_decay": 0.0},
        ]
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if g["params"]]

        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=self.lr,
            weight_decay=0.01,
            max_grad_norm=1.0
        )

        def lr_lambda(current_step):
            warmup_steps = self.warmup_ratio * self.total_steps
            if current_step < warmup_steps:
                return current_step / warmup_steps
            else:
                return self.lr_decay_gamma ** ((current_step - warmup_steps) // self.epoch_steps)

        # Step-based warmup, epoch-based decay scheduler
        warmup_decay_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warmup_decay_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_decay"
            }
        }

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        loss = self.loss(batch, outputs)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=batch["input_labels"].size(0))
        return loss

    def on_validation_epoch_start(self) -> None:
        self.batch_res = {"version": "VERSION 1.0",
                          "results": defaultdict(list),
                          "external_data": {"used": "true", "details": "ay"}}

    def validation_step(self, batch, batch_idx):
        inputs_ids = batch["input_ids"]
        input_masks = batch["input_mask"]
        cap_config = self.model.caption_head.cap_config
        max_t_len = cap_config.max_t_len  # hard-code sentence length, for speed test, set it to 21
        inputs_ids[:, :] = cap_config.PAD_id
        input_masks[:, :] = 0.
        assert torch.sum(input_masks[:, :]) == 0, "Initially, all text tokens should be masked"
        bsz = len(inputs_ids)
        next_symbols = torch.IntTensor([cap_config.BOS_id] * bsz)  # (N, )

        warn_visual_output = False
        for dec_idx in range(max_t_len):
            inputs_ids[:, dec_idx] = next_symbols.clone()
            input_masks[:, dec_idx] = 1
            outputs = self.model(batch)
            pred_scores = outputs["prediction_scores"]
            next_words = pred_scores[:, dec_idx].max(1)[1]
            next_symbols = next_words.cpu()
            if "visual_output" in outputs:
                batch["visual_output"] = outputs["visual_output"]
            elif not warn_visual_output:
                logger.warning("visual_output is not in the output of model, this may slow down the caption test")
                warn_visual_output = True
        dec_seq = inputs_ids

        for example_idx, (cur_gen_sen, cur_meta) in enumerate(zip(dec_seq, batch['metadata'][1])):
            cur_data = {
                "sentence": convert_ids_to_sentence(cur_gen_sen.tolist(), self.tokenizer_name),
                "gt_sentence": cur_meta
            }
            # MSRVTT keys its references by the numeric id, having stripped the "video" prefix;
            # other datasets key by the full id. removeprefix does the right thing for both,
            # where split() would also cut ids that merely contain the word.
            video_id = batch['metadata'][0][example_idx]
            self.batch_res["results"][video_id.removeprefix("video")].append(cur_data)

    def on_validation_epoch_end(self) -> None:
        json_res = copy.deepcopy(self.batch_res)
        if dist.is_initialized():
            all_results = gather_object_multiple_gpu(list(json_res["results"].items()))
            json_res['results'] = {k: v for k, v in all_results}
            logger.debug("Caption test length: %s", len(json_res["results"].items()))

        # save result tp log for debug
        if not dist.is_initialized() or dist.get_rank() == 0:
            res_filepath = os.path.join(self.trainer.default_root_dir,
                                        "caption_greedy_pred_validation_{}.json".format(get_timestamp()))
            os.makedirs(os.path.dirname(res_filepath), exist_ok=True)
            save_json(json_res, res_filepath, save_pretty=True)

        if not dist.is_initialized() or dist.get_rank() == 0:
            json_ref = self.trainer.val_dataloaders.dataset.json_ref
            metrics = evaluate(json_res, json_ref)
            if metrics:
                self.log_dict(metrics, on_step=False, on_epoch=True, logger=True)
            else:
                # logging nothing would leave a monitored checkpoint callback looking for a key
                # that never appears, so say so plainly instead
                logger.warning("no caption metrics were produced for this validation epoch")

        if dist.is_initialized():
            dist.barrier()


cocap_lm_cfg = builds(
    CoCapLM,
    cocap_model=compressed_video_captioner_pretrained_cfg,
    loss=label_smoothing_loss_cfg,
    populate_full_signature=True
)

# Phase 4: SigLIP2 encoder + baseline BERT decoder. The text side is unchanged, so the CLIP
# tokenizer and its 49408-entry vocabulary stay in place.
cocap_lm_siglip_cfg = builds(
    CoCapLM,
    cocap_model=compressed_video_captioner_siglip_cfg,
    loss=label_smoothing_loss_cfg,
    tokenizer="clip",
    vision_pretrained_prefixes=SIGLIP_VISION_PREFIXES,
    decoder_pretrained_prefixes=BERT_DECODER_PREFIXES,
    populate_full_signature=True
)

# Phase 5: SigLIP2 encoder + GPT-2 decoder. Switches the tokenizer, the vocabulary the loss is
# built over, and the label padding convention (-100 instead of 0).
cocap_lm_siglip_gpt2_cfg = builds(
    CoCapLM,
    cocap_model=compressed_video_captioner_siglip_gpt2_cfg,
    loss=builds(LabelSmoothingLoss, target_vocab_size=50257, ignore_index=-100,
                populate_full_signature=True),
    tokenizer="gpt2",
    vision_pretrained_prefixes=SIGLIP_VISION_PREFIXES,
    decoder_pretrained_prefixes=GPT2_DECODER_PREFIXES,
    populate_full_signature=True
)

#: name -> config for every selectable model variant
MODEL_VARIANTS = {
    "cocap": cocap_lm_cfg,
    "cocap_siglip": cocap_lm_siglip_cfg,
    "cocap_siglip_gpt2": cocap_lm_siglip_gpt2_cfg,
}


def register_model_configs(store):
    """Register the model variants as the hydra `model` config group.

    Lives here rather than in a script so anything that needs to build a model from an
    experiment config (training, benchmarking, evaluation) registers the same group.
    """
    model_store = store(group="model")
    for name, cfg in MODEL_VARIANTS.items():
        model_store(cfg, name=name)
    return store
