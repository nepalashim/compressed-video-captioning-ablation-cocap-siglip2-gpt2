# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : check_encoding_parity.py
"""
Assert that a video directory is encoded the same way as the clips the model was trained on.

The compressed-domain pipeline is not encoding-agnostic. Three of its settings are tied to
properties of the bitstream rather than to the architecture:

* ``motion_channels: 2`` holds only while the clips carry no B-frames, since the discarded L1
  motion-vector pair is populated by B-frames alone.
* ``num_mv``/``num_res`` are sized against the GOP length, i.e. the encoder's ``KeyInt``. Our
  clips use 60, giving M = 59 available B/P frames per GOP.
* ``num_gop: 5`` is sized against how many GOPs a ten-second clip holds at that ``KeyInt``.

and one more belongs to the transform rather than the model: ``DictCenterCrop((224, 224))``
crops a 224x224 window out of a frame whose short edge is 240. Hand it a frame that is already
224x224 and the crop degenerates to a no-op, so the network sees the whole frame squashed to
square instead of a centre crop at the native aspect ratio.

A clip set that differs on any of these is not comparable with the trained checkpoints, and the
resulting metric change says nothing about the architecture under test. This tool checks each
property explicitly and compares the GOP histogram against a reference split.

Usage::

    python tools/check_encoding_parity.py \\
        --video_dir  "$VATEX_SUBSET_ROOT/test" \\
        --reference  "$VATEX_SUBSET_ROOT/train" --reference_sample 200
"""

import argparse
import json
import os
import random
import subprocess
import sys
from collections import Counter

import av
import joblib
import tqdm

# What our train and val clips are, as measured: the HuggingFace extraction pass emitted
# 240 px short edge at native aspect, H.264 High / yuv420p, KeyInt 60 and no B-frames.
EXPECTED = {
    "codec_name": "h264",
    "profile": "High",
    "pix_fmt": "yuv420p",
    "short_edge": 240,
    "encoder": "Lavf61.1.100",
    "max_keyint": 60,
    "b_frames": 0,
}

# DictCenterCrop's window, from configs/dataset/vatex_subset.yaml's video_size.
CROP_SIZE = 224

# ``frame.pict_type`` is an int in recent PyAV rather than an enum member.
PICT = {0: "NONE", 1: "I", 2: "P", 3: "B", 4: "S", 5: "SI", 6: "SP", 7: "BI"}


def probe(path):
    """Container/stream metadata via ffprobe, frame types via a full PyAV decode."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate",
             "-show_entries", "format=duration:format_tags=encoder",
             "-of", "json", path],
            capture_output=True, text=True, check=True).stdout
        meta = json.loads(out)
        stream = meta["streams"][0]
        fmt = meta.get("format", {})

        types, gaps, last_i = Counter(), [], None
        with av.open(path) as container:
            for index, frame in enumerate(container.decode(video=0)):
                kind = getattr(frame.pict_type, "name", None) or PICT.get(int(frame.pict_type), "?")
                types[kind] += 1
                if kind == "I":
                    if last_i is not None:
                        gaps.append(index - last_i)
                    last_i = index
    except Exception as exc:                                  # noqa: BLE001 - reported per file
        return {"file": os.path.basename(path), "error": f"{type(exc).__name__}: {exc}"}

    frames = sum(types.values())
    width, height = stream["width"], stream["height"]
    return {
        "file": os.path.basename(path),
        "codec_name": stream.get("codec_name"),
        "profile": stream.get("profile"),
        "pix_fmt": stream.get("pix_fmt"),
        "width": width,
        "height": height,
        "short_edge": min(width, height),
        "square": width == height,
        "fps": stream.get("r_frame_rate"),
        "duration": round(float(fmt.get("duration") or 0), 2),
        "encoder": (fmt.get("tags") or {}).get("encoder"),
        "frames": frames,
        "gops": types["I"],
        "b_frames": types["B"],
        # a clip with a single I-frame has no gap to measure; its GOP is the whole clip
        "max_keyint": max(gaps) if gaps else frames,
    }


def violations(row):
    """Every way in which one clip departs from the training encoding."""
    if "error" in row:
        return [row["error"]]
    out = []
    for key in ("codec_name", "profile", "pix_fmt", "encoder"):
        if row[key] != EXPECTED[key]:
            out.append(f"{key}={row[key]!r} (want {EXPECTED[key]!r})")
    if row["short_edge"] != EXPECTED["short_edge"]:
        out.append(f"short_edge={row['short_edge']} (want {EXPECTED['short_edge']})")
    # The crop has to actually crop. At a 240 px short edge it takes a 224x224 window out of a
    # larger frame; at 224 it returns the frame whole, so the network sees the full field of
    # view rescaled rather than a centre crop. That is the family-B tell, and it is a property
    # of the short edge alone - a natively square 240x240 clip crops perfectly well, and about
    # 1% of the clips in every split are exactly that.
    if row["short_edge"] <= CROP_SIZE:
        out.append(f"short_edge={row['short_edge']} <= crop {CROP_SIZE}: "
                   f"DictCenterCrop would be a no-op")
    if row["b_frames"] != EXPECTED["b_frames"]:
        out.append(f"b_frames={row['b_frames']} (want 0; motion_channels=2 depends on it)")
    if row["max_keyint"] > EXPECTED["max_keyint"]:
        out.append(f"max_keyint={row['max_keyint']} (want <= {EXPECTED['max_keyint']})")
    return out


def sweep(directory, sample=None, seed=42, jobs=None):
    files = sorted(f for f in os.listdir(directory) if f.endswith(".mp4"))
    if sample and sample < len(files):
        files = random.Random(seed).sample(files, sample)
        files.sort()
    paths = [os.path.join(directory, f) for f in files]
    runner = joblib.Parallel(n_jobs=jobs or os.cpu_count(), return_as="generator")(
        joblib.delayed(probe)(p) for p in paths
    )
    return list(tqdm.tqdm(runner, total=len(paths), desc=os.path.basename(directory.rstrip("/\\"))))


def summarise(label, rows):
    ok = [r for r in rows if "error" not in r]
    gops = Counter(r["gops"] for r in ok)
    keyint = Counter(r["max_keyint"] for r in ok)
    n = len(ok)
    print(f"\n{label}  ({len(rows)} clips, {len(rows) - n} unreadable)")
    if not n:
        return {}
    med = sorted(r["gops"] for r in ok)[n // 2]
    ge5 = sum(c for g, c in gops.items() if g >= 5)
    sizes = Counter("{}x{}".format(r["width"], r["height"]) for r in ok)
    print(f"  resolutions      : {dict(sizes.most_common(5))}")
    print(f"  short edge       : {dict(Counter(r['short_edge'] for r in ok))}")
    print(f"  codec / profile  : {dict(Counter((r['codec_name'], r['profile']) for r in ok))}")
    print(f"  encoder tag      : {dict(Counter(r['encoder'] for r in ok))}")
    print(f"  B-frames anywhere: {sum(1 for r in ok if r['b_frames'])}")
    print(f"  largest I-gap    : max {max(keyint)}, distribution top-5 {dict(keyint.most_common(5))}")
    print(f"  GOPs per clip    : min {min(gops)}  median {med}  max {max(gops)}")
    print(f"  GOP histogram    : {dict(sorted(gops.items()))}")
    print(f"  clips with >= 5  : {ge5} ({100 * ge5 / n:.0f}%)")
    print(f"  duration         : median {sorted(r['duration'] for r in ok)[n // 2]} s")
    return {"median_gops": med, "pct_ge5": 100 * ge5 / n,
            "histogram": dict(sorted(gops.items())), "n": n}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video_dir", required=True, help="directory to check, in full")
    parser.add_argument("--reference", help="directory to compare the GOP distribution against")
    parser.add_argument("--reference_sample", type=int, default=200,
                        help="how many reference clips to sample (0 = all)")
    parser.add_argument("--jobs", type=int, default=0, help="0 = os.cpu_count()")
    parser.add_argument("--output", "-o", default=None, help="write the per-clip rows as json")
    args = parser.parse_args()

    jobs = args.jobs or None
    rows = sweep(args.video_dir, jobs=jobs)
    stats = summarise(f"TARGET  {args.video_dir}", rows)

    bad = {r["file"]: violations(r) for r in rows if violations(r)}
    print(f"\nParity against the training encoding: {len(rows) - len(bad)}/{len(rows)} clips conform")
    if bad:
        print(f"  {len(bad)} clip(s) violate it:")
        for name, reasons in list(bad.items())[:20]:
            print(f"    {name}: {'; '.join(reasons)}")
        if len(bad) > 20:
            print(f"    ... and {len(bad) - 20} more")
        tally = Counter(r.split("=")[0].split(" ")[0] for reasons in bad.values() for r in reasons)
        print(f"  by property: {dict(tally)}")
    else:
        print("  every checked property matches: codec, profile, pix_fmt, 240 px short edge "
              "(so the centre crop crops), zero B-frames, KeyInt <= 60, encoder tag")

    if args.reference:
        ref = sweep(args.reference, sample=args.reference_sample or None, jobs=jobs)
        ref_stats = summarise(f"REFERENCE  {args.reference}", ref)
        if stats and ref_stats:
            print(f"\nGOP distribution: target median {stats['median_gops']} vs reference "
                  f"{ref_stats['median_gops']}; "
                  f">= 5 GOPs {stats['pct_ge5']:.0f}% vs {ref_stats['pct_ge5']:.0f}%")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"target": rows, "violations": bad}, f, indent=1)
        print(f"\nwrote {args.output}")

    sys.exit(2 if bad else 0)


if __name__ == "__main__":
    main()
