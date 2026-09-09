# -*- coding: utf-8 -*-
# @Time    : 8/2/23
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : test_compressed_video_transformer.py

import torch
from hydra_zen import to_yaml, instantiate
from pytest import main

from cocap.modules.compressed_video.compressed_video_transformer import *

def test_cvt_from_pretrained():
    model = CompressedVideoTransformer.from_pretrained(
        pretrained_clip_name_or_path="ViT-B/16",
        motion_patch_size=8, motion_layers=2, motion_heads=8,
        residual_patch_size=64, residual_layers=2, residual_heads=12,
        action_layers=1, action_heads=8, n_bp=59
    )
    print(model)


# Shape/wiring checks only, so batch size just multiplies memory. At batch 5 the motion and
# residual tensors together are ~2.6 GB, and with autograd recording it is enough to trigger the
# OOM killer on a laptop.
BSZ = 1


def _fake_inputs():
    return dict(
        iframe=torch.rand(BSZ, 8, 3, 224, 224),
        motion=torch.rand(BSZ, 8, 59, 4, 56, 56),
        residual=torch.rand(BSZ, 8, 59, 3, 224, 224),
        bp_type_ids=torch.randint(0, 1, (BSZ, 8, 59)),
    )


@torch.no_grad()
def test_cvt_forward():
    model = CompressedVideoTransformer.from_pretrained()

    output = model(**_fake_inputs())
    for k, v in output.items():
        print(k, v.shape)


@torch.no_grad()
def test_cvt_forward_instantiate():
    model = instantiate(compressed_video_transformer_pretrained_cfg)

    output = model(**_fake_inputs())
    for k, v in output.items():
        print(k, v.shape)


def test_print_config():
    print(to_yaml(iframe_encoder_cfg))
    print(to_yaml(iframe_encoder_pretrained_cfg))
    print(to_yaml(action_encoder_cfg))
    print(to_yaml(compressed_video_transformer_cfg))
    print(to_yaml(compressed_video_transformer_pretrained_cfg))


if __name__ == '__main__':
    main()
