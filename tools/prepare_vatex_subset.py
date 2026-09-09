# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : prepare_vatex_subset.py
"""
Convert the HuggingFace VATEX subset into the metadata layout CoCap's dataset classes expect.

The downloaded subset ships a COCO-style annotation file::

    {"videos":      [{"video_id", "videoID", "filename", "split"}, ...],
     "annotations": [{"id", "video_id", "caption", "split"}, ...]}

while ``VATEXCaptioningDataset`` expects::

    {"train":    ["<video_id>", ...],
     "test":     ["<video_id>", ...],
     "metadata": [{"video_id": ..., "sentence": ...}, ...]}

The video files are also named by ``videoID`` (the bare YouTube id) rather than by ``video_id``
(which carries the clip's start/end timestamps), so a separate id -> filename map is emitted and
consumed by ``VATEXSubsetCaptioningDataset``.

Usage::

    python tools/prepare_vatex_subset.py \
        --annotations "C:/Research-Personal/Datasetextract5000train1000validfromhuggingface/VATEX_Caption.json" \
        --train_dir   "C:/Research-Personal/Datasetextract5000train1000validfromhuggingface/train" \
        --val_dir     "C:/Research-Personal/Datasetextract5000train1000validfromhuggingface/val" \
        --output_dir  "./dataset/vatex_subset"
"""

import argparse
import json
import os
from collections import Counter, defaultdict


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
    parser.add_argument("--annotations", required=True, help="COCO-style VATEX_Caption.json")
    parser.add_argument("--train_dir", required=True, help="directory holding the train .mp4 files")
    parser.add_argument("--val_dir", required=True, help="directory holding the val .mp4 files")
    parser.add_argument("--output_dir", default="./dataset/vatex_subset")
    parser.add_argument("--val_split_name", default="val",
                        help="name of the validation split inside the annotation file")
    parser.add_argument("--on_collision", choices=["drop", "keep"], default="drop",
                        help="what to do when several video_ids (different clip timestamps) map "
                             "to a single video file: 'drop' removes every colliding id, since "
                             "the correct segment is not recoverable; 'keep' pairs the same clip "
                             "with each id's captions")
    args = parser.parse_args()

    meta = load_json(args.annotations)
    videos, annotations = meta["videos"], meta["annotations"]
    print(f"annotations file: {len(videos)} videos, {len(annotations)} captions")

    files_on_disk = {
        "train": {f for f in os.listdir(args.train_dir) if f.endswith(".mp4")},
        "test": {f for f in os.listdir(args.val_dir) if f.endswith(".mp4")},
    }
    print(f"files on disk:    train={len(files_on_disk['train'])}, val={len(files_on_disk['test'])}")

    # ---- map video_id -> filename, keeping only entries whose file is present -------------
    split_ids = {"train": [], "test": []}
    id_to_file = {}
    file_users = defaultdict(list)   # filename -> [video_id, ...], to detect collisions
    dropped = Counter()

    for v in videos:
        split = "train" if v["split"] == "train" else "test" if v["split"] == args.val_split_name else None
        if split is None:
            dropped["unknown split"] += 1
            continue
        filename = v.get("filename") or f"{v['videoID']}.mp4"
        if filename not in files_on_disk[split]:
            dropped[f"missing file ({split})"] += 1
            continue
        vid = v["video_id"]
        if vid in id_to_file:
            dropped["duplicate video_id"] += 1
            continue
        split_ids[split].append(vid)
        id_to_file[vid] = filename
        file_users[filename].append(vid)

    # ---- collisions: one file, several clip timestamps -----------------------------------
    # VATEX video_ids embed the clip's start/end seconds, but the files are named by the bare
    # YouTube id. When two ids share a file we cannot tell which segment the file actually is,
    # so pairing either id's captions with it would be a known-wrong supervision signal.
    collisions = {f: ids for f, ids in file_users.items() if len(ids) > 1}
    if collisions:
        example = next(iter(collisions.items()))
        print(f"WARNING: {len(collisions)} video file(s) claimed by several video_ids, e.g. {example}")
        if args.on_collision == "drop":
            colliding_ids = {vid for ids in collisions.values() for vid in ids}
            for split in split_ids:
                split_ids[split] = [v for v in split_ids[split] if v not in colliding_ids]
            for vid in colliding_ids:
                id_to_file.pop(vid, None)
            dropped["ambiguous clip (collision)"] += len(colliding_ids)
            print(f"         -> dropped {len(colliding_ids)} ambiguous ids (--on_collision=drop)")
        else:
            print("         -> keeping them; the same clip is reused for each id "
                  "(--on_collision=keep)")

    # ---- captions, restricted to the kept ids --------------------------------------------
    kept = set(id_to_file)
    sentences = [{"video_id": a["video_id"], "sentence": a["caption"]}
                 for a in annotations if a["video_id"] in kept]

    per_video = Counter(s["video_id"] for s in sentences)
    missing_caps = kept - set(per_video)
    if missing_caps:
        print(f"WARNING: {len(missing_caps)} videos have no captions and will never be sampled")

    caption_json = {
        "train": sorted(split_ids["train"]),
        "test": sorted(split_ids["test"]),
        "metadata": sentences,
    }

    out_caption = os.path.join(args.output_dir, "vatex_subset_caption.json")
    out_map = os.path.join(args.output_dir, "id_to_file.json")
    save_json(caption_json, out_caption)
    save_json(id_to_file, out_map)

    counts = sorted(set(per_video.values()))
    print()
    print(f"  train ids : {len(caption_json['train'])}")
    print(f"  test  ids : {len(caption_json['test'])}")
    print(f"  captions  : {len(sentences)}  (captions/video: {counts})")
    if dropped:
        print(f"  dropped   : {dict(dropped)}")
    print()
    print(f"wrote {out_caption}")
    print(f"wrote {out_map}")


if __name__ == "__main__":
    main()
