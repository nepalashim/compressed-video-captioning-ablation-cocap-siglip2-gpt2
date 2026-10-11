# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : prepare_vatex_testset.py
"""
Build the metadata for the 2,000-clip VATEX test split.

Same job as ``tools/prepare_vatex_subset.py`` - convert the COCO-style annotation file into the
``{"train": [...], "test": [...], "metadata": [...]}`` layout ``VATEXCaptioningDataset`` wants,
plus the ``clip_id -> filename`` map ``VATEXSubsetCaptioningDataset`` needs because the files are
named by bare YouTube id while the captions are keyed by clip id.

Written as a separate tool rather than a flag on ``prepare_vatex_subset.py`` so the train/val
metadata that produced the published numbers stays byte-identical and the existing results stay
reproducible. The output goes to its own pair of files for the same reason.

``"train"`` is emitted empty: this split is only ever consumed as ``split: "test"``, and an empty
train list makes it impossible to accidentally train on it.

Refuses to write if any test clip id or video file also appears in the train/val metadata, since
a test split that overlaps training is worse than no test split at all.

Usage::

    python tools/prepare_vatex_testset.py \\
        --annotations "$VATEX_SUBSET_ROOT/VATEX_CAPTIONS_test.json" \\
        --test_dir    "$VATEX_SUBSET_ROOT/test" \\
        --output_dir  ./dataset/vatex_subset
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

EXPECTED_CAPTIONS = 10


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True,
                        help="COCO-style annotation file covering the test clips")
    parser.add_argument("--test_dir", required=True, help="directory holding the test .mp4 files")
    parser.add_argument("--output_dir", default="./dataset/vatex_subset")
    parser.add_argument("--split_name", default="test",
                        help="name of the test split inside the annotation file")
    parser.add_argument("--existing", default="./dataset/vatex_subset/vatex_subset_caption.json",
                        help="train/val metadata to check for leakage against")
    parser.add_argument("--existing_map", default="./dataset/vatex_subset/id_to_file.json")
    args = parser.parse_args()

    meta = load_json(args.annotations)
    videos, annotations = meta["videos"], meta["annotations"]
    print(f"annotations file: {len(videos)} videos, {len(annotations)} captions")

    on_disk = {f for f in os.listdir(args.test_dir) if f.endswith(".mp4")}
    print(f"files on disk   : {len(on_disk)}")

    # ---- clip_id -> filename, keeping only entries whose file is present ------------------
    test_ids, id_to_file = [], {}
    file_users = defaultdict(list)
    dropped = Counter()

    for v in videos:
        if v.get("split") != args.split_name:
            dropped[f"split != {args.split_name}"] += 1
            continue
        filename = v.get("filename") or f"{v['videoID']}.mp4"
        if filename not in on_disk:
            dropped["missing file"] += 1
            continue
        clip_id = v["video_id"]
        if clip_id in id_to_file:
            dropped["duplicate clip id"] += 1
            continue
        test_ids.append(clip_id)
        id_to_file[clip_id] = filename
        file_users[filename].append(clip_id)

    # ---- collisions: one file, several clip timestamps ------------------------------------
    # The files are named by bare YouTube id, so when two clip ids (different start/end
    # seconds) share a file there is no way to tell which segment the file holds. Measured at
    # zero for this split, unlike the train/val subset - but assert it rather than assume it.
    collisions = {f: ids for f, ids in file_users.items() if len(ids) > 1}
    if collisions:
        example = next(iter(collisions.items()))
        print(f"\nERROR: {len(collisions)} file(s) claimed by several clip ids, e.g. {example}")
        sys.exit("ambiguous clip timestamps; refusing to write")

    # ---- captions -------------------------------------------------------------------------
    kept = set(id_to_file)
    sentences = [{"video_id": a["video_id"], "sentence": a["caption"]}
                 for a in annotations if a["video_id"] in kept]
    per_clip = Counter(s["video_id"] for s in sentences)

    short = {c: n for c, n in per_clip.items() if n != EXPECTED_CAPTIONS}
    missing = kept - set(per_clip)
    if missing:
        print(f"\nERROR: {len(missing)} clips have no captions: {sorted(missing)[:10]}")
        sys.exit("uncaptioned clips would never be scored; refusing to write")
    if short:
        print(f"\nWARNING: {len(short)} clips do not have exactly {EXPECTED_CAPTIONS} captions: "
              f"{dict(list(short.items())[:10])}")

    # ---- leakage check against train/val ---------------------------------------------------
    if os.path.exists(args.existing):
        existing = load_json(args.existing)
        train_val_ids = set(existing.get("train", [])) | set(existing.get("test", []))
        clip_overlap = kept & train_val_ids
        file_overlap = set()
        if os.path.exists(args.existing_map):
            file_overlap = set(id_to_file.values()) & set(load_json(args.existing_map).values())
        print(f"\nleakage check against {args.existing}:")
        print(f"  train+val clip ids : {len(train_val_ids)}")
        print(f"  overlapping clip ids: {len(clip_overlap)}")
        print(f"  overlapping files   : {len(file_overlap)}")
        if clip_overlap or file_overlap:
            print(f"  {sorted(clip_overlap)[:10]} {sorted(file_overlap)[:10]}")
            sys.exit("test split overlaps training data; refusing to write")
    else:
        print(f"\nWARNING: {args.existing} not found, skipping the leakage check")

    caption_json = {"train": [], "test": sorted(test_ids), "metadata": sentences}
    out_caption = os.path.join(args.output_dir, "vatex_testset_caption.json")
    out_map = os.path.join(args.output_dir, "id_to_file_test.json")
    save_json(caption_json, out_caption)
    save_json(id_to_file, out_map)

    print()
    print(f"  test clips : {len(caption_json['test'])}")
    print(f"  captions   : {len(sentences)} (per clip: {sorted(set(per_clip.values()))})")
    print(f"  dataloader : {len(caption_json['test'])} samples scored against "
          f"{len(sentences)} references (unfold_sentences keeps one sample per clip on test)")
    if dropped:
        print(f"  dropped    : {dict(dropped)}")
    print()
    print(f"wrote {out_caption}")
    print(f"wrote {out_map}")


if __name__ == "__main__":
    main()
