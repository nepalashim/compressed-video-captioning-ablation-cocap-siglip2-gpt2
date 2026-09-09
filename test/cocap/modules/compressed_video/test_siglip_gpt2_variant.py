# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : test_siglip_gpt2_variant.py
"""Wiring tests for the SigLIP2 encoder and GPT-2 decoder variants.

These do not need `cv_reader`: the compressed-domain tensors are synthesised directly.
"""

import pytest
import torch

from cocap.data.tokenizers import build_tokenizer
from cocap.modeling.loss import LabelSmoothingLoss
from cocap.modules.compressed_video import CompressedVideoCaptioner
from cocap.modules.compressed_video.compressed_video_transformer import CompressedVideoTransformer
from cocap.modules.gpt2 import GPT2CaptionHead

# kept small so the test stays quick on CPU
BSZ, N_GOP, N_BP, MOTION_C, MAX_T = 2, 2, 4, 2, 12


def _fake_video(resolution: int, motion_channels: int = MOTION_C):
    return {
        "iframe": torch.rand(BSZ, N_GOP, 3, resolution, resolution),
        "motion_vector": torch.rand(BSZ, N_GOP, N_BP, motion_channels,
                                    resolution // 4, resolution // 4),
        "residual": torch.randint(0, 255, (BSZ, N_GOP, N_BP, 3, resolution, resolution)).float(),
        # no B-frames in the target encoding, so every B/P slot is a P-frame
        "type_ids_mv": torch.zeros(BSZ, N_GOP, N_BP, dtype=torch.long),
    }


@pytest.fixture(scope="module")
def siglip_transformer():
    return CompressedVideoTransformer.from_siglip_pretrained(
        motion_channels=MOTION_C, n_bp=N_BP, n_bp_type=3
    )


@pytest.mark.parametrize("name,vocab,ignore", [("clip", 49408, 0), ("gpt2", 50257, -100)])
def test_tokenizer_roundtrip(name, vocab, ignore):
    tok = build_tokenizer(name)
    assert (tok.vocab_size, tok.ignore_index) == (vocab, ignore)

    sentence = "a man is riding a horse on the beach"
    input_ids, input_mask, input_labels = tok.encode(sentence, MAX_T)
    assert input_ids.shape == input_mask.shape == input_labels.shape == (MAX_T,)
    assert tok.decode(input_ids) == sentence
    # labels are the next-token targets, so they lead input_ids by one position
    n_real = int(input_mask.sum())
    assert torch.equal(input_labels[:n_real - 1], input_ids[1:n_real])


def test_tokenizer_truncates_long_caption():
    tok = build_tokenizer("gpt2")
    input_ids, input_mask, _ = tok.encode(" ".join(["word"] * 100), MAX_T)
    assert input_ids.shape == (MAX_T,)
    assert input_ids[-1] == tok.eos_id, "a truncated caption must still terminate with EOS"
    assert int(input_mask.sum()) == MAX_T


def test_siglip_encoder_shapes(siglip_transformer):
    enc = siglip_transformer.rgb_encoder
    n_patches = (enc.input_resolution // enc.patch_size) ** 2

    ctx, patches, attn = enc(torch.rand(BSZ, 3, enc.input_resolution, enc.input_resolution),
                             output_all_features=True, output_attention_map=True)
    assert ctx.shape == (BSZ, enc.output_dim), "pooler_output stands in for CLIP's CLS token"
    assert patches.shape == (BSZ, n_patches, enc.output_dim)
    assert attn is None, "transformers backbones do not expose per-layer attention maps"


def test_siglip_transformer_forward(siglip_transformer):
    res = siglip_transformer.rgb_encoder.input_resolution
    video = _fake_video(res)

    with torch.no_grad():
        out = siglip_transformer(
            iframe=video["iframe"], motion=video["motion_vector"],
            residual=video["residual"] / 128 - 1, bp_type_ids=video["type_ids_mv"],
        )

    dim = siglip_transformer.output_dim
    assert out["feature_context"].shape == (BSZ, N_GOP, dim)
    assert out["feature_action"].shape == (BSZ, N_GOP, dim)
    assert out["iframe_attention_map"] is None


def test_motion_channel_mismatch_is_rejected(siglip_transformer):
    res = siglip_transformer.rgb_encoder.input_resolution
    video = _fake_video(res, motion_channels=4)  # encoder was built for 2
    with pytest.raises(AssertionError, match="motion encoder expects"):
        siglip_transformer(
            iframe=video["iframe"], motion=video["motion_vector"],
            residual=video["residual"], bp_type_ids=video["type_ids_mv"],
        )


def test_gpt2_captioner_end_to_end(siglip_transformer):
    head = GPT2CaptionHead(visual_feature_size=siglip_transformer.output_dim,
                           max_v_len=N_GOP * 2, max_t_len=MAX_T)
    model = CompressedVideoCaptioner(compressed_video_transformer=siglip_transformer,
                                     caption_head=head)
    tok = build_tokenizer("gpt2")
    input_ids, input_mask, input_labels = tok.encode("a man is riding a horse", MAX_T)

    batch = {
        "video": _fake_video(siglip_transformer.rgb_encoder.input_resolution),
        "input_ids": input_ids.unsqueeze(0).repeat(BSZ, 1),
        "input_mask": input_mask.unsqueeze(0).repeat(BSZ, 1),
        "input_labels": input_labels.unsqueeze(0).repeat(BSZ, 1),
    }

    with torch.no_grad():
        outputs = model(batch)
    assert outputs["prediction_scores"].shape == (BSZ, MAX_T, tok.vocab_size)

    loss = LabelSmoothingLoss(target_vocab_size=tok.vocab_size, ignore_index=tok.ignore_index)
    assert torch.isfinite(loss(batch, outputs)), "loss must handle the sliced, non-contiguous logits"

    # the greedy decode loop caches the visual features and re-enters with them
    batch["visual_output"] = outputs["visual_output"]
    with torch.no_grad():
        reused = model(batch)
    assert reused["prediction_scores"].shape == (BSZ, MAX_T, tok.vocab_size)


def test_gpt2_head_rejects_wrong_visual_length(siglip_transformer):
    head = GPT2CaptionHead(visual_feature_size=siglip_transformer.output_dim,
                           max_v_len=N_GOP * 2 + 2, max_t_len=MAX_T)  # deliberately wrong
    tok = build_tokenizer("gpt2")
    input_ids, input_mask, _ = tok.encode("a man is riding a horse", MAX_T)
    visual = {
        "feature_context": torch.rand(BSZ, N_GOP, siglip_transformer.output_dim),
        "feature_action": torch.rand(BSZ, N_GOP, siglip_transformer.output_dim),
    }
    with pytest.raises(AssertionError, match="visual tokens"):
        head(visual, input_ids.unsqueeze(0).repeat(BSZ, 1), input_mask.unsqueeze(0).repeat(BSZ, 1))
