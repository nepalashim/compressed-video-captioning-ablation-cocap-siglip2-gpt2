# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : dataset_vatex_subset.py
"""
VATEX subset (5k train / 1k val) downloaded from HuggingFace.

Identical to :class:`VATEXCaptioningDataset` except that the video files are named by the bare
YouTube id (``--33Lscn6sk.mp4``) while the captions are keyed by the VATEX clip id, which also
carries the segment timestamps (``--33Lscn6sk_000000_000010``). ``tools/prepare_vatex_subset.py``
emits the ``video_id -> filename`` map this class consumes.
"""

__all__ = ["VATEXSubsetCaptioningDataset"]

import logging
import os

from cocap.utils.json import load_json
from .dataset_vatex import VATEXCaptioningDataset

logger = logging.getLogger(__name__)


class VATEXSubsetCaptioningDataset(VATEXCaptioningDataset):

    def __init__(self, *args, id_to_file: str, **kwargs):
        """
        :param id_to_file: path to the ``video_id -> filename`` json written by
                           ``tools/prepare_vatex_subset.py``
        """
        self._id_to_file = load_json(id_to_file)
        super().__init__(*args, **kwargs)
        logger.info("VATEX subset (%s): %d samples over %d videos",
                    self.split, len(self.sentences), len(set(v for v, _ in self.sentences)))

    def _get_video_path(self, video_id):
        try:
            filename = self._id_to_file[video_id]
        except KeyError:
            raise KeyError(
                f"video_id {video_id!r} is not in the id_to_file map. Re-run "
                f"tools/prepare_vatex_subset.py so the map and the caption json agree."
            ) from None
        return os.path.join(self.video_root, filename)
