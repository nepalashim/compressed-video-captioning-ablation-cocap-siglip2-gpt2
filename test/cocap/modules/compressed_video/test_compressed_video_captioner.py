# -*- coding: utf-8 -*-
# @Time    : 8/5/23
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : test_compressed_video_captioner.py


import torch
from hydra_zen import instantiate, to_yaml
from pytest import main

from cocap.modules.compressed_video.compressed_video_captioner import *


# These are shape/wiring checks, so the batch size only multiplies memory. At batch 8 the
# residual tensor alone is 8*8*59*3*224*224 int64 = ~4.5 GB before the float conversion, which
# is enough to trigger the OOM killer on a laptop.
BSZ = 1


@torch.no_grad()
def test_forward():
    model = instantiate(compressed_video_captioner_pretrained_cfg)

    outputs = model(
        {
            "video": {
                "iframe": torch.randn(BSZ, 8, 3, 224, 224),
                "motion_vector": torch.randn(BSZ, 8, 59, 4, 56, 56),
                # uint8 matches what the dataset actually produces
                "residual": torch.randint(0, 255, size=(BSZ, 8, 59, 3, 224, 224),
                                          dtype=torch.uint8),
                "type_ids_mv": torch.randint(0, 1, size=(BSZ, 8, 59))
            },
            "input_ids": torch.randint(0, 1000, size=(BSZ, 77)),
            "input_mask": torch.ones((BSZ, 77), dtype=torch.long),
        }
    )
    print(outputs["prediction_scores"].shape)
    print(outputs["visual_output"]["feature_context"].shape)
    print(outputs["visual_output"]["feature_action"].shape)
    print(outputs["visual_output"]["iframe_attention_map"].shape)
    print(outputs["visual_output"]["motion_vector_attention_map"].shape)
    print(outputs["visual_output"]["residual_attention_map"].shape)


def test_print_config():
    print(to_yaml(caption_head_cfg))
    print(to_yaml(caption_head_pretrained_cfg))
    print(to_yaml(compressed_video_captioner_pretrained_cfg))


if __name__ == '__main__':
    main()
