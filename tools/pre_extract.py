# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : pre_extract.py
"""
Pre-extract compressed-domain features so training stops re-decoding video.

Without this, every training sample runs the full H.264 parse: `cv_reader` decodes the whole
clip and materialises motion vectors and residuals for all ~300 frames, of which the sampler
keeps a few hundred. With `unfold_sentences: True` each video carries 10 captions, so over a
13-epoch run **every clip is decoded ~130 times**. Measured on an RTX 4070 Laptop that is
0.04 it/s, or 6.5 days per epoch.

This script does that parse **once per clip** and writes the four files
`read_frames_compressed_domain(pre_extract=True)` expects, alongside the video:

    <video>.pict_type       pickle           frame type sequence ("I"/"P"/"B")
    <video>.rgb_gop         pickle           I-frame RGB, JPEG-encoded
    <video>.motion_vector   pickle + lz4     per-frame motion vectors (raw arrays)
    <video>.residual        pickle + lz4     per-frame residuals, JPEG-encoded

Then set `use_pre_extract: True` in the dataset config.

Usage::

    # measure the storage cost on a small sample first
    python tools/pre_extract.py --video_dir "$VATEX_SUBSET_ROOT/train" --limit 20 --workers 4

    # then the real thing
    python tools/pre_extract.py --video_dir "$VATEX_SUBSET_ROOT/train" --workers 6
    python tools/pre_extract.py --video_dir "$VATEX_SUBSET_ROOT/val"   --workers 6

Re-runs skip clips that already have a complete set, so it is safe to interrupt and resume.
"""

import argparse
import os
import pickle
import sys
import traceback

import numpy as np

# the suffixes read_frames_compressed_domain looks for, and whether they are lz4-compressed
ARTIFACTS = {
    "pict_type": False,
    "rgb_gop": False,
    "motion_vector": True,
    "residual": True,
}


def _paths(video_path):
    return {name: f"{video_path}.{name}" for name in ARTIFACTS}


def is_complete(video_path):
    return all(os.path.exists(p) and os.path.getsize(p) > 0 for p in _paths(video_path).values())


def extracted_size(video_path):
    return sum(os.path.getsize(p) for p in _paths(video_path).values() if os.path.exists(p))


def _write(path, key, value, compress):
    """Write one artifact atomically, so an interrupted run leaves no half-file behind."""
    import lz4.frame

    tmp = f"{path}.tmp"
    opener = lz4.frame.open if compress else open
    with opener(tmp, "wb") as f:
        pickle.dump({key: value}, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def extract_one(video_path, quality=85, overwrite=False):
    """Returns (video_path, ok, bytes_written, error)."""
    import cv_reader
    import decord

    from cocap.data.datasets.compressed_video.compressed_video_utils import serialize

    if not overwrite and is_complete(video_path):
        return video_path, True, extracted_size(video_path), None

    try:
        frames = cv_reader.read_video(video_path)
        if not frames:
            return video_path, False, 0, "cv_reader returned no frames"

        pict_type = [f["pict_type"] for f in frames]

        # The non-pre-extract path pulls I-frames from the container by index, so do the same
        # here rather than relying on cv_reader carrying RGB.
        iframe_idx = [f["frame_idx"] for f in frames if f["pict_type"] == "I"]
        if not iframe_idx:
            return video_path, False, 0, "no I-frames found"
        reader = decord.VideoReader(video_path, num_threads=1)
        rgb_gop = [np.asarray(img) for img in reader.get_batch(iframe_idx).asnumpy()]

        # The reader indexes these lists by frame position, so they must cover every frame.
        # I-frames carry no inter-prediction data; the sampler never draws from them (it only
        # takes B/P frames), so placeholders are never read - they just keep the indexing aligned.
        ref_mv = next((f["motion_vector"] for f in frames if f.get("motion_vector") is not None),
                      None)
        ref_res = next((f["residual"] for f in frames if f.get("residual") is not None), None)
        if ref_mv is None or ref_res is None:
            return video_path, False, 0, "no motion vectors or residuals in the stream"

        zero_mv = np.zeros_like(ref_mv)
        # residuals are stored unsigned and centred on 128, so a neutral placeholder is 128
        gray_res = np.full_like(ref_res, 128)

        motion_vector, residual = [], []
        for f in frames:
            mv = f.get("motion_vector")
            res = f.get("residual")
            motion_vector.append(zero_mv if mv is None else mv)
            residual.append(gray_res if res is None else res)

        # JPEG-encodes rgb_gop and residual; motion vectors stay as raw arrays
        payload = serialize(
            {"rgb_gop": rgb_gop, "residual": residual, "motion_vector": motion_vector},
            quality=quality,
        )
        payload["pict_type"] = pict_type

        for name, compress in ARTIFACTS.items():
            _write(f"{video_path}.{name}", name, payload[name], compress)

        return video_path, True, extracted_size(video_path), None

    except Exception:
        for p in _paths(video_path).values():  # do not leave a partial set behind
            for cand in (p, f"{p}.tmp"):
                if os.path.exists(cand):
                    os.remove(cand)
        return video_path, False, 0, traceback.format_exc(limit=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--quality", type=int, default=85,
                        help="JPEG quality for residuals and I-frames. Residuals are mostly flat, "
                             "so 85 is visually lossless here; drop to 75 to trade a little "
                             "fidelity for disk")
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N clips - use this to measure storage before "
                             "committing to the full set")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        import cv_reader  # noqa: F401
    except ImportError:
        sys.exit("cv_reader is not installed - see docs/runbook.md section 6")

    import joblib
    import tqdm

    videos = sorted(os.path.join(args.video_dir, f)
                    for f in os.listdir(args.video_dir) if f.endswith(".mp4"))
    if not videos:
        sys.exit(f"no .mp4 files in {args.video_dir}")
    # keep the full count: --limit exists to size up the whole directory from a sample, so the
    # projection has to be against every clip, not just the ones processed
    n_in_dir = len(videos)
    if args.limit:
        videos = videos[:args.limit]

    todo = videos if args.overwrite else [v for v in videos if not is_complete(v)]
    already = len(videos) - len(todo)
    print(f"clips        : {len(videos)}  ({already} already extracted, {len(todo)} to do)")
    print(f"workers      : {args.workers}   jpeg quality: {args.quality}")
    print()

    runner = joblib.Parallel(n_jobs=args.workers, return_as="generator_unordered")(
        joblib.delayed(extract_one)(v, args.quality, args.overwrite) for v in todo
    )

    import time
    started = time.perf_counter()
    total_bytes, failures = 0, []
    for video_path, ok, size, error in tqdm.tqdm(runner, total=len(todo), dynamic_ncols=True,
                                                 desc="extracting"):
        if ok:
            total_bytes += size
        else:
            failures.append((video_path, error))

    elapsed = time.perf_counter() - started
    done = len(todo) - len(failures)
    print()
    if done:
        per_clip = total_bytes / done
        print(f"extracted    : {done} clips, {total_bytes / 1e9:.2f} GB "
              f"({per_clip / 1e6:.1f} MB/clip)")
        print(f"projected    : {per_clip * n_in_dir / 1e9:.1f} GB for all {n_in_dir} clips "
              f"in {args.video_dir}")
        if args.limit and done:
            rate = done / max(elapsed, 1e-9)
            print(f"             : ~{n_in_dir / rate / 60:.0f} min to do all {n_in_dir} "
                  f"at the observed {rate:.1f} clips/s")
    if failures:
        print(f"FAILED       : {len(failures)} clips")
        for video_path, error in failures[:5]:
            print(f"  {os.path.basename(video_path)}")
            print("   ", (error or "").strip().splitlines()[-1])
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")

    print()
    print("Next: set `use_pre_extract: True` in configs/dataset/vatex_subset.yaml, then")
    print("      python tools/validate_data_pipeline.py --variant baseline -n 50 --build-model")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
