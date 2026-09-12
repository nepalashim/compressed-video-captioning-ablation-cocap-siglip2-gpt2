# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : benchmark_latency.py
"""
Measure per-clip inference latency, the axis CoCap's central claim lives on.

Reports the two halves separately, because they scale differently when components are swapped:

* **visual**  one pass of the compressed-video transformer (I-frame + motion + residual encoders
  and the action encoder). Changing the I-frame backbone moves this number.
* **decode**  greedy caption generation. The baseline head is 2 BERT layers; GPT-2 is 12, so a
  decoder swap is expected to move this number and it needs to be quantified, not assumed.

Runs on synthetic tensors by default, so it needs neither ``cv_reader`` nor the video files and
can be used to compare architectures before the data path is up. The shapes are taken from the
experiment config, so they match what training will actually see.

Usage::

    python tools/benchmark_latency.py --variant siglip_gpt2
    python tools/benchmark_latency.py --variant baseline --device cuda --n 50
"""

import argparse
import statistics
import time

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path

VARIANTS = {
    "baseline": "exp/train/vatex_subset_baseline",
    "siglip": "exp/train/vatex_subset_siglip",
    "siglip_gpt2": "exp/train/vatex_subset_siglip_gpt2",
}


def build(variant: str, overrides=None):
    """Instantiate exactly the model an experiment config would train, so the timing reflects
    the real thing rather than a hand-built approximation."""
    from hydra_zen import store

    from cocap.modeling.config_store import register_configs

    config_dir = (Path(__file__).parent.parent / "configs").resolve()
    register_configs(store)
    store.add_to_hydra_store(overwrite_ok=True)

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=VARIANTS[variant], overrides=list(overrides or []))
    lm = instantiate(cfg.model)
    cv = OmegaConf.to_container(cfg.train_dataloader.dataset.cv_config)
    return lm.model, cv, cfg.train_dataloader.dataset.max_words


def synth_batch(model, cv, max_t_len, batch_size, device):
    res = 224
    n_gop, n_bp = cv["num_gop"], cv["num_mv"]
    mc = cv.get("motion_channels", 4)
    cap = model.caption_head.cap_config
    return {
        "video": {
            "iframe": torch.rand(batch_size, n_gop, 3, res, res, device=device),
            "motion_vector": torch.rand(batch_size, n_gop, n_bp, mc, res // 4, res // 4,
                                        device=device),
            "residual": torch.randint(0, 255, (batch_size, n_gop, n_bp, 3, res, res),
                                      device=device).float(),
            "type_ids_mv": torch.zeros(batch_size, n_gop, n_bp, dtype=torch.long, device=device),
        },
        "input_ids": torch.full((batch_size, max_t_len), cap.PAD_id, dtype=torch.long,
                                device=device),
        "input_mask": torch.zeros(batch_size, max_t_len, dtype=torch.long, device=device),
    }


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def run(model, batch, max_t_len, device):
    """One clip's worth of work: encode once, then greedily decode, reusing the visual features
    exactly the way CoCapLM.validation_step does."""
    cap = model.caption_head.cap_config
    video = batch["video"]

    # call the transformer directly rather than the captioner, which would also run one decoder
    # pass and fold it into the "visual" number
    sync(device)
    t0 = time.perf_counter()
    visual_output = model.compressed_video_transformer(
        iframe=video["iframe"],
        motion=model.dropout_motion(video["motion_vector"]),
        residual=model.dropout_residual(video["residual"] / 128 - 1),
        bp_type_ids=video["type_ids_mv"],
    )
    sync(device)
    t_visual = time.perf_counter() - t0

    decode_batch = dict(batch)
    decode_batch["visual_output"] = visual_output
    input_ids = batch["input_ids"].clone()
    input_mask = batch["input_mask"].clone()
    input_ids[:] = cap.PAD_id
    input_mask[:] = 0
    next_symbols = torch.full((input_ids.size(0),), cap.BOS_id, dtype=torch.long, device=device)

    sync(device)
    t0 = time.perf_counter()
    for step in range(max_t_len):
        input_ids[:, step] = next_symbols
        input_mask[:, step] = 1
        decode_batch["input_ids"] = input_ids
        decode_batch["input_mask"] = input_mask
        scores = model(decode_batch)["prediction_scores"]
        next_symbols = scores[:, step].max(-1)[1]
    sync(device)
    t_decode = time.perf_counter() - t0

    return t_visual, t_decode


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="the paper reports batch size 1")
    parser.add_argument("-n", "--num_iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    # trailing key=value arguments are forwarded to hydra, so latency can be measured at the
    # same budget training used: `budget=laptop_8gb`
    args, overrides = parser.parse_known_args()
    malformed = [a for a in overrides if '=' not in a]
    if malformed:
        parser.error(f"unrecognized arguments: {' '.join(malformed)}")

    device = torch.device(args.device)
    model, cv, max_words = build(args.variant, overrides)
    model = model.to(device).eval()

    max_t_len = model.caption_head.cap_config.max_t_len
    batch = synth_batch(model, cv, max_words, args.batch_size, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"variant     : {args.variant}")
    print(f"device      : {device}")
    print(f"parameters  : {n_params / 1e6:.1f}M")
    print(f"shapes      : gop={cv['num_gop']} n_bp={cv['num_mv']} "
          f"motion_channels={cv.get('motion_channels', 4)} max_t_len={max_t_len}")
    print(f"batch size  : {args.batch_size}")

    for _ in range(args.warmup):
        run(model, batch, max_t_len, device)

    visual, decode = [], []
    for _ in range(args.num_iters):
        tv, td = run(model, batch, max_t_len, device)
        visual.append(tv * 1000 / args.batch_size)
        decode.append(td * 1000 / args.batch_size)

    total = [v + d for v, d in zip(visual, decode)]
    print()
    print(f"{'stage':<10}{'median':>12}{'mean':>12}{'stdev':>12}   (ms per clip)")
    for label, xs in (("visual", visual), ("decode", decode), ("total", total)):
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        print(f"{label:<10}{statistics.median(xs):>12.1f}{statistics.mean(xs):>12.1f}{sd:>12.1f}")

    if device.type == "cpu":
        print("\nNOTE: CPU timings are only useful for comparing variants against each other. "
              "Report GPU numbers.")


if __name__ == "__main__":
    main()
