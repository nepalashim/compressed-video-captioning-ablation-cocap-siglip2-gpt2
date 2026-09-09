# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : verify_motion_channels.py
"""
Verify the motion-vector channel layout that ``motion_channels=2`` assumes.

``cv_reader`` returns 4 motion-vector channels for AVC. The working assumption is that they are
laid out as ``[dx_L0, dy_L0, dx_L1, dy_L1]``: the L0 pair references the past (used by both P-
and B-frames) and the L1 pair references the future (only B-frames use it). For a stream encoded
without B-frames the L1 pair should therefore be identically zero, which is what makes it safe to
slice the tensor down to its first two channels.

This script checks that empirically on real videos. **Run it once in WSL after building
cv_reader, before trusting `motion_channels: 2` in the dataset config.**

Usage::

    python tools/verify_motion_channels.py --video_dir /path/to/vatex_subset/train -n 20
"""

import argparse
import os
import random
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("-n", "--num_videos", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        import cv_reader
    except ImportError:
        sys.exit("cv_reader is not installed — build it first (see docs/implementation-plan.md 1.4)")

    files = sorted(f for f in os.listdir(args.video_dir) if f.endswith(".mp4"))
    random.seed(args.seed)
    sample = random.sample(files, min(args.num_videos, len(files)))

    n_frames = 0
    pict_types = {}
    per_channel_absmax = None
    n_shape_bad = 0

    for name in sample:
        frames = cv_reader.read_video(os.path.join(args.video_dir, name))
        for f in frames:
            pt = f.get("pict_type")
            pict_types[pt] = pict_types.get(pt, 0) + 1
            mv = f.get("motion_vector")
            if mv is None:
                continue
            if mv.ndim != 3 or mv.shape[-1] != 4:
                n_shape_bad += 1
                continue
            absmax = np.abs(mv.astype(np.float64)).reshape(-1, 4).max(axis=0)
            per_channel_absmax = absmax if per_channel_absmax is None \
                else np.maximum(per_channel_absmax, absmax)
            n_frames += 1

    print(f"videos sampled      : {len(sample)}")
    print(f"frames with motion  : {n_frames}")
    print(f"picture types       : {pict_types}")
    if n_shape_bad:
        print(f"WARNING: {n_shape_bad} motion tensors were not (H, W, 4)")
    if per_channel_absmax is None:
        sys.exit("no motion vectors found — nothing to verify")

    print(f"per-channel |max|   : {per_channel_absmax.tolist()}")
    print()

    b_frames = pict_types.get("B", 0)
    l1_dead = bool((per_channel_absmax[2:] == 0).all())

    if b_frames:
        print(f"FAIL: {b_frames} B-frames present. Re-encode without B-frames "
              f"(x264 `bframes=0`) or keep motion_channels=4.")
        sys.exit(1)
    if not l1_dead:
        print("FAIL: channels 2:4 are NOT all zero even though there are no B-frames, so they "
              "are not the L1 pair. Keep motion_channels=4 and re-derive the layout before "
              "slicing.")
        sys.exit(1)

    print("PASS: no B-frames, and channels 2:4 are identically zero.")
    print("      `motion_channels: 2` is safe — it drops only dead data.")


if __name__ == "__main__":
    main()
