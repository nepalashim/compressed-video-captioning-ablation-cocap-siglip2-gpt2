# -*- coding: utf-8 -*-
# @Time    : 2022/12/12 12:36
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : test_video_readers.py

import os
import unittest

import cv_reader
import flow_vis
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from cocap.data.datasets.compressed_video.video_readers import read_frames_compressed_domain


def _find_test_video():
    """Locate a video to exercise the reader against.

    Prefers the dataset this project actually uses (via ``$VATEX_SUBSET_ROOT``) and falls back
    to upstream's hard-coded MSRVTT path. Returns ``None`` when neither is present, so the tests
    skip rather than fail on a missing data dependency.
    """
    root = os.environ.get("VATEX_SUBSET_ROOT")
    if root:
        train_dir = os.path.join(root, "train")
        if os.path.isdir(train_dir):
            clips = sorted(f for f in os.listdir(train_dir) if f.endswith(".mp4"))
            if clips:
                return os.path.join(train_dir, clips[0])
    legacy = "dataset/msrvtt/videos_h264_keyint_60/video0.mp4"
    return legacy if os.path.exists(legacy) else None


TEST_VIDEO = _find_test_video()
requires_video = unittest.skipIf(
    TEST_VIDEO is None,
    "no test video found; set VATEX_SUBSET_ROOT or provide dataset/msrvtt/videos_h264_keyint_60"
)


class TestCVReader(unittest.TestCase):
    """
    test case for basic video reader
    """
    video = TEST_VIDEO

    @requires_video
    def test_cv_reader(self):
        """read video and print a brief view of data and visualize"""
        data = cv_reader.read_video(self.video)
        print("Number of frames: {}".format(len(data)))
        for k, v in data[0].items():
            if type(v) is np.ndarray:
                print(k, v.shape, v.reshape(-1)[:20])
            else:
                print(k, v)

        output_dir = "test_output/data/dataset/compressed_video/test_video_readers/visualize"
        os.makedirs(output_dir, exist_ok=True)
        for frame_idx in tqdm.tqdm(list(range(len(data)))):
            plt.imsave(os.path.join(output_dir, f"residual_{frame_idx:05d}.jpg"),
                       data[frame_idx]["residual"])
            plt.imsave(os.path.join(output_dir, f"motion_vector_{frame_idx:05d}.jpg"),
                       flow_vis.flow_to_color(data[frame_idx]["motion_vector"][..., :2]))


class TestVideoReader(unittest.TestCase):
    video = TEST_VIDEO

    @requires_video
    def test_read_frames_compressed_domain(self):
        motion_channels = 2
        data, is_success = read_frames_compressed_domain(
            video_path=self.video,
            resample_num_gop=8, resample_num_mv=59, resample_num_res=59,
            with_residual=True, with_bp_rgb=False, pre_extract=False, sample="rand",
            motion_channels=motion_channels
        )

        for k, v in data.items():
            print(f"{k}: {v.shape}")

        # The reader swallows every exception and returns all-zero tensors alongside a False
        # flag, so without this assertion the test passes just as happily on a video it could
        # not read at all.
        self.assertTrue(is_success, f"reader failed on {self.video}; see video_reader_error.log")
        self.assertEqual(data["iframe"].shape[0], 8)
        self.assertEqual(tuple(data["motion_vector"].shape[1:3]), (59, motion_channels))
        self.assertEqual(data["residual"].shape[1], 59)
        self.assertTrue(data["iframe"].any(), "I-frames are all zero")


if __name__ == '__main__':
    unittest.main()
