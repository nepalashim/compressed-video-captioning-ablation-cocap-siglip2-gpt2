# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : fetch_vatex_testset.py
"""
Download the 2,000-clip VATEX test split in the encoding that matches our train/val clips.

``Blazewild/Vatex_reencoded`` publishes VATEX twice, in two incompatible encodings:

* **family A** - ``train/``, ``train1/``, ``train2/``, ``val/``, ``test/``, ``test1/``
  240 px short edge at the clip's native aspect ratio, ``KeyInt`` 60, ~5 GOPs per clip.
  This is what our 4,999 train and 1,000 val clips are, and what ``num_gop: 5`` /
  ``num_mv: 16`` were sized for.
* **family B** - ``selected_8k/``, ``test_videos/``
  224x224 square, ``KeyInt`` 30, ~9-10 GOPs per clip.

``test_videos/`` curates a 2,000-clip subset with 10 captions each, which is the selection we
want - but in family B, where ``DictCenterCrop((224, 224))`` degenerates to a no-op and a GOP
spans half the time it should. Both break comparability with the trained checkpoints, and
neither is recoverable by re-encoding: the square resize has already discarded the aspect ratio.

Every one of those 2,000 ids is however *also* published in family A (1,907 in ``test/``, 174 in
``test1/``), so this tool takes the curated id list from ``test_videos/`` and fetches each clip
from ``test/`` or ``test1/`` instead. Same clips, same captions, encoding that matches training.
``FAMILY_B`` is refused outright so a missing file can never silently fall back to it.

Usage::

    # annotations only (run first; cheap)
    python tools/fetch_vatex_testset.py --captions-only --out "$VATEX_SUBSET_ROOT"

    # the 2,000 clips
    python tools/fetch_vatex_testset.py --out "$VATEX_SUBSET_ROOT/test"

    # once you have your own copy of the caption file, confirm it agrees
    python tools/fetch_vatex_testset.py --verify-captions /path/to/VATEXCAPTIONS.json \\
        --out "$VATEX_SUBSET_ROOT"
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "Blazewild/Vatex_reencoded"

# Searched in order. Family A only - see the module docstring.
SOURCE_DIRS = ("test", "test1")
FAMILY_B = ("test_videos", "selected_8k")

CURATED_LIST = "test_videos/VATEX_CAPTIONS_test.json"   # id list only; its mp4s are not used
FULL_CAPTIONS = "VATEX_CAPTIONS.json"
EXPECTED = 2000


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path, pretty=True):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1 if pretty else None)


def fetch_captions(out_dir):
    """Pull both annotation files into the dataset root and report what is in them."""
    from huggingface_hub import hf_hub_download

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name in (FULL_CAPTIONS, CURATED_LIST):
        src = hf_hub_download(REPO, name, repo_type="dataset", local_dir=out_dir)
        dst = os.path.join(out_dir, os.path.basename(name))
        if os.path.abspath(src) != os.path.abspath(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        paths[name] = dst
        print(f"  {os.path.basename(dst):28s} {os.path.getsize(dst) / 2 ** 20:7.1f} MiB")

    # strip the now-empty test_videos/ directory the download created
    stray = os.path.join(out_dir, "test_videos")
    if os.path.isdir(stray) and not os.listdir(stray):
        os.rmdir(stray)

    full = load_json(paths[FULL_CAPTIONS])
    curated = load_json(paths[CURATED_LIST])
    print(f"\n  {FULL_CAPTIONS}: {len(full['videos'])} videos, "
          f"{len(full['annotations'])} captions, "
          f"splits {dict(Counter(v['split'] for v in full['videos']))}")
    print(f"  {os.path.basename(CURATED_LIST)}: {len(curated['videos'])} videos, "
          f"{len(curated['annotations'])} captions")
    return paths


def curated_ids(out_dir):
    """The 2,000 videoIDs to fetch, read from the curated caption file."""
    path = os.path.join(out_dir, os.path.basename(CURATED_LIST))
    if not os.path.exists(path):
        sys.exit(f"{path} is missing - run with --captions-only first")
    meta = load_json(path)
    ids = sorted({v["videoID"] for v in meta["videos"]})
    if len(ids) != EXPECTED:
        sys.exit(f"expected {EXPECTED} curated ids, found {len(ids)}")
    return ids


def resolve(ids):
    """Map each videoID to a family-A repo path, or report it as unavailable."""
    from huggingface_hub import HfApi

    info = HfApi().repo_info(REPO, repo_type="dataset")
    available = defaultdict(dict)   # top-level dir -> {videoID: repo path}
    for sibling in info.siblings:
        name = sibling.rfilename
        if "/" not in name or not name.endswith(".mp4"):
            continue
        top, base = name.split("/", 1)
        available[top][base[:-4]] = name

    resolved, missing = {}, []
    for vid in ids:
        for d in SOURCE_DIRS:
            if vid in available[d]:
                resolved[vid] = available[d][vid]
                break
        else:
            missing.append(vid)

    where = Counter(p.split("/")[0] for p in resolved.values())
    print(f"  resolved {len(resolved)}/{len(ids)} ids from family A: {dict(where)}")
    if missing:
        in_b = [v for v in missing if any(v in available[d] for d in FAMILY_B)]
        print(f"  UNRESOLVED: {len(missing)} ids are absent from {SOURCE_DIRS}")
        if in_b:
            print(f"    {len(in_b)} of them exist only in {FAMILY_B}, which this tool refuses "
                  f"to use (wrong encoding; see the module docstring)")
        for v in missing[:10]:
            print(f"      {v}")
        sys.exit("refusing to proceed with an incomplete selection")
    return resolved


def download(resolved, out_dir, workers):
    from huggingface_hub import hf_hub_download

    os.makedirs(out_dir, exist_ok=True)
    staging = os.path.join(out_dir, ".staging")
    os.makedirs(staging, exist_ok=True)

    def one(vid, repo_path):
        dst = os.path.join(out_dir, f"{vid}.mp4")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return vid, repo_path, dst, True
        src = hf_hub_download(REPO, repo_path, repo_type="dataset", local_dir=staging)
        os.replace(src, dst)        # same filesystem, so atomic
        return vid, repo_path, dst, False

    manifest, done, skipped = {}, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, v, p) for v, p in sorted(resolved.items())]
        for fut in as_completed(futures):
            vid, repo_path, dst, cached = fut.result()
            manifest[vid] = {
                "source": repo_path,
                "bytes": os.path.getsize(dst),
                "sha256": sha256(dst),
            }
            done += 1
            skipped += cached
            if done % 250 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} files")

    shutil.rmtree(staging, ignore_errors=True)
    if skipped:
        print(f"  ({skipped} already present, left untouched)")
    return manifest


def verify(out_dir, resolved, manifest):
    on_disk = {f[:-4] for f in os.listdir(out_dir) if f.endswith(".mp4")}
    wanted = set(resolved)
    total = sum(m["bytes"] for m in manifest.values())
    print(f"\n  files on disk : {len(on_disk)}")
    print(f"  total size    : {total / 2 ** 30:.2f} GiB")

    extra, absent = on_disk - wanted, wanted - on_disk
    if absent:
        print(f"  MISSING ({len(absent)}): {sorted(absent)[:10]}")
    if extra:
        # deliberately not deleted - the operator looks at the list first
        print(f"  UNEXPECTED ({len(extra)}): {sorted(extra)[:10]}")
        print("  nothing was deleted; review the list above and remove them by hand")
    if not extra and not absent:
        print(f"  set equality  : OK, exactly the {EXPECTED} curated ids")
    return not extra and not absent


def verify_captions(user_path, out_dir):
    """Diff a user-supplied caption file against the downloaded one over the 2,000 clips."""
    ours = load_json(os.path.join(out_dir, os.path.basename(CURATED_LIST)))
    theirs = load_json(user_path)

    def index(meta):
        by_file = {v["videoID"]: v["video_id"] for v in meta["videos"]}
        caps = defaultdict(list)
        for a in meta["annotations"]:
            caps[a["video_id"]].append(a["caption"])
        return by_file, caps

    our_files, our_caps = index(ours)
    their_files, their_caps = index(theirs)
    ids = set(our_files)

    print(f"  ours  : {len(our_files)} videos")
    print(f"  theirs: {len(their_files)} videos, {len(theirs['annotations'])} captions")

    absent = sorted(ids - set(their_files))
    if absent:
        print(f"  {len(absent)} of our 2,000 ids are absent from your file: {absent[:10]}")

    clash, differ = [], []
    for vid in sorted(ids & set(their_files)):
        if our_files[vid] != their_files[vid]:
            clash.append(vid)
            continue
        clip = our_files[vid]
        if sorted(our_caps[clip]) != sorted(their_caps[clip]):
            differ.append(vid)
    if clash:
        print(f"  {len(clash)} ids map to a different clip id: {clash[:10]}")
    if differ:
        print(f"  {len(differ)} clips have different caption text: {differ[:10]}")

    if not absent and not clash and not differ:
        print("  the two files agree exactly over all 2,000 clips")
        return True
    print("\n  your file disagrees; it takes precedence - rebuild the metadata from it")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True,
                        help="destination; the dataset root for --captions-only / "
                             "--verify-captions, the test clip directory otherwise")
    parser.add_argument("--captions-only", action="store_true",
                        help="fetch the two annotation files and stop")
    parser.add_argument("--verify-captions", metavar="PATH",
                        help="diff a caption file of your own against the downloaded one")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.verify_captions:
        print("Comparing caption files")
        sys.exit(0 if verify_captions(args.verify_captions, args.out) else 2)

    if args.captions_only:
        print(f"Fetching annotations into {args.out}")
        fetch_captions(args.out)
        return

    dataset_root = os.path.dirname(os.path.abspath(args.out))
    print(f"Reading the curated id list from {dataset_root}")
    ids = curated_ids(dataset_root)
    print(f"  {len(ids)} ids\n\nResolving against {REPO}")
    resolved = resolve(ids)
    print(f"\nDownloading into {args.out}")
    manifest = download(resolved, args.out, args.workers)

    manifest_path = os.path.join(dataset_root, "test_manifest.json")
    save_json({"repo": REPO, "source_dirs": list(SOURCE_DIRS), "files": manifest},
              manifest_path)
    print(f"  wrote {manifest_path}")

    print("\nVerifying")
    sys.exit(0 if verify(args.out, resolved, manifest) else 2)


if __name__ == "__main__":
    main()
