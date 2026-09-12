# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : test_eval_captioning.py
"""Tests for caption scoring, in particular partial validation passes.

The scorers assert that prediction and reference keys match exactly. Lightning's sanity check
and `limit_val_batches` both produce predictions for only part of the validation set, so
`evaluate` has to score the intersection rather than fail.
"""

import shutil

import pytest

from cocap.modeling.eval_captioning import evaluate

# pycocoevalcap's PTBTokenizer and METEOR shell out to Java; without a JRE they fail with a bare
# FileNotFoundError from subprocess, which says nothing about the real cause.
requires_java = pytest.mark.skipif(shutil.which("java") is None,
                                   reason="pycocoevalcap needs a JRE (apt install default-jre)")

REFERENCE = {
    "vid1": ["a man is riding a horse", "someone rides a horse on a beach"],
    "vid2": ["a woman is cooking in a kitchen", "a person prepares food"],
    "vid3": ["a dog runs through a field", "a dog is running outside"],
}


def _submission(*video_ids):
    return {"results": {vid: [{"sentence": f"a caption for {vid}"}] for vid in video_ids}}


@requires_java
@pytest.mark.parametrize("predicted", [
    ("vid1",),                      # a sanity check's worth
    ("vid1", "vid2"),               # a partial pass
    ("vid1", "vid2", "vid3"),       # the full set
])
def test_scores_the_intersection(predicted):
    metrics = evaluate(_submission(*predicted), REFERENCE)
    assert {"Bleu_4", "METEOR", "ROUGE_L", "CIDEr"} <= set(metrics)
    assert all(isinstance(v, float) for v in metrics.values())


@requires_java
def test_predictions_absent_from_references_are_skipped():
    # a partial pass plus one id the references know nothing about
    metrics = evaluate(_submission("vid1", "not_a_real_video"), REFERENCE)
    assert "CIDEr" in metrics


def test_no_overlap_returns_empty_rather_than_raising():
    # nothing scoreable: return empty so the caller can report it, rather than blowing up
    assert evaluate(_submission("nope1", "nope2"), REFERENCE) == {}
