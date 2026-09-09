# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : fake_cv_reader.py
"""
Development stand-in for the native ``cv_reader`` module.

``cv_reader`` (https://github.com/yaojie-shen/Compressed-Video-Reader) is a C++ extension that
parses motion vectors and residuals out of an H.264 bitstream. It is not on PyPI and needs an
FFmpeg-linked CMake build, which makes it awkward to have available while developing the rest of
the pipeline.

This module reproduces its *interface* so that everything downstream — GOP grouping, B/P
sampling, motion-channel slicing, padding and masks, the dict transforms, collation, and the
model's dimension checks — can be exercised against real video files today.

What is real and what is not:

* **real** — the picture-type sequence and frame indices, demuxed from the actual file with
  PyAV, so GOP boundaries, GOP lengths and GOP counts are exactly what ``cv_reader`` would see
* **synthetic** — the motion vector and residual *values*. Shapes, dtypes and the zeroed L1
  motion-vector pair (correct for a stream without B-frames) match, but the numbers are noise.

.. danger::
   Any metric produced with this installed is meaningless. It refuses to install unless
   explicitly asked, logs a warning every time, and sets ``cv_reader.IS_FAKE = True`` so callers
   can assert against it.
"""

__all__ = ["read_video", "install", "IS_FAKE"]

import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)

#: marker so training/eval entry points can refuse to run against synthetic data
IS_FAKE = True

#: FFmpeg AVPictureType -> the single-letter codes cv_reader emits
_PICT_TYPES = {0: "NONE", 1: "I", 2: "P", 3: "B", 4: "S", 5: "SI", 6: "SP", 7: "BI"}


def _pict_type_name(pict_type) -> str:
    if pict_type is None:
        return "NONE"
    if isinstance(pict_type, int):
        return _PICT_TYPES.get(pict_type, str(pict_type))
    return getattr(pict_type, "name", str(pict_type))


def read_video(video_path: str, seed: int = 0):
    """Mimic ``cv_reader.read_video``.

    :return: one dict per frame, with ``pict_type``, ``frame_idx``, ``motion_vector``
        ``(H//4, W//4, 4)`` and ``residual`` ``(H, W, 3)`` uint8.
    """
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    width, height = stream.codec_context.width, stream.codec_context.height

    pict_types = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            pict_types.append(_pict_type_name(frame.pict_type))
    container.close()

    rng = np.random.default_rng(seed + (hash(video_path) & 0xFFFF))
    mv_h, mv_w = height // 4, width // 4
    frames = []
    for idx, pict_type in enumerate(pict_types):
        # motion vectors are [dx_L0, dy_L0, dx_L1, dy_L1]; the L1 pair is only populated by
        # B-frames, so on a B-frame-free stream it is identically zero
        motion_vector = np.zeros((mv_h, mv_w, 4), dtype=np.int32)
        if pict_type != "I":
            motion_vector[..., :2] = rng.integers(-32, 33, size=(mv_h, mv_w, 2), dtype=np.int32)
            if pict_type == "B":
                motion_vector[..., 2:] = rng.integers(-32, 33, size=(mv_h, mv_w, 2),
                                                      dtype=np.int32)
        frames.append({
            "pict_type": pict_type,
            "frame_idx": idx,
            "motion_vector": motion_vector,
            # residuals are stored unsigned and centred on 128 by the reader
            "residual": rng.integers(112, 145, size=(height, width, 3), dtype=np.uint8),
        })
    return frames


def install(force: bool = False):
    """Register this module as ``cv_reader`` in :data:`sys.modules`.

    :param force: replace a real ``cv_reader`` if one is importable. Never do this outside a
        deliberate comparison — the real module is always what you want when it exists.
    """
    if "cv_reader" in sys.modules and not getattr(sys.modules["cv_reader"], "IS_FAKE", False):
        if not force:
            logger.info("real cv_reader is already imported; leaving it in place")
            return sys.modules["cv_reader"]
    if not force:
        try:
            import cv_reader as real  # noqa: F401
            logger.info("real cv_reader is available; not installing the fake")
            return real
        except ImportError:
            pass

    logger.warning(
        "Installing the FAKE cv_reader. Picture types and frame indices are read from the real "
        "files, but motion vectors and residuals are noise. Use this only to exercise the data "
        "pipeline — any metric computed now is meaningless."
    )
    sys.modules["cv_reader"] = sys.modules[__name__]
    return sys.modules[__name__]
