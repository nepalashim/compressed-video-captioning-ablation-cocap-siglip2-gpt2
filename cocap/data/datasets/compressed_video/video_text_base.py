# -*- coding: utf-8 -*-
# @Time    : 2022/12/3 14:54
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : video_text_base.py
import logging
import os.path
import random
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

#: paths the reader failed on in this process. Dataloader workers each keep their own list, so
#: treat it as a sampling signal during validation rather than a global tally.
GET_VIDEO_FAILURES: list = []


def get_tokenized_words(sentence: str, tokenizer, max_words):
    words = tokenizer.tokenize(sentence)
    words = ["[CLS]"] + words
    total_length_with_cls = max_words - 1
    if len(words) > total_length_with_cls:
        words = words[:total_length_with_cls]
    words = words + ["[SEP]"]
    return words


def get_text_inputs(sentence: str, tokenizer, max_words):
    """
    1. tokenize
    2. add [CLS] and [SEP] token, limit the length
    3. create mask and token type
    4. pad to max_words
    :param sentence:
    :param tokenizer:
    :param max_words:
    :return: 1 dim tensor, shape is (max_words,)
    """
    words = get_tokenized_words(sentence, tokenizer, max_words)
    input_ids = tokenizer.convert_tokens_to_ids(words)
    input_mask = [1] * len(input_ids)  # 1 is keep, 0 is mask out
    segment_ids = [0] * len(input_ids)
    while len(input_ids) < max_words:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == len(input_mask) == len(segment_ids) == max_words
    return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids)


def get_text_inputs_with_mlm(sentence: str, tokenizer, max_words):
    """
    Add mlm inputs and labels based on `get_text_inputs`
    :param sentence:
    :param tokenizer:
    :param max_words:
    :return: 1 dim tensor, shape is (max_words,)
    """
    input_ids, input_mask, segment_ids = get_text_inputs(sentence, tokenizer, max_words)

    # Mask Language Model <-----
    token_labels = []
    masked_tokens = get_tokenized_words(sentence, tokenizer, max_words)
    for token_id, token in enumerate(masked_tokens):
        if token_id == 0 or token_id == len(masked_tokens) - 1:
            token_labels.append(-1)
            continue
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15
            # 80% randomly change token to mask token
            if prob < 0.8:
                masked_tokens[token_id] = "[MASK]"
            # 10% randomly change token to random token
            elif prob < 0.9:
                masked_tokens[token_id] = random.choice(list(tokenizer.vocab.items()))[0]
            # -> rest 10% randomly keep current token
            # append current token to output (we will predict these later)
            try:
                token_labels.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                token_labels.append(tokenizer.vocab["[UNK]"])
                logger.debug("Cannot find token '{}' in vocab. Using [UNK] instead".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            token_labels.append(-1)
    # -----> Mask Language Model
    masked_token_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    while len(masked_token_ids) < max_words:
        masked_token_ids.append(0)
        token_labels.append(-1)
    assert len(masked_token_ids) == len(token_labels) == max_words
    return input_ids, input_mask, segment_ids, torch.tensor(masked_token_ids), torch.tensor(token_labels)


@dataclass
class CVConfig:
    num_gop: int
    num_mv: int
    num_res: int
    with_residual: bool
    use_pre_extract: bool
    sample: str
    #: motion vector channels to keep; 2 drops the all-zero L1 (B-frame) pair on a stream
    #: encoded without B-frames, 4 preserves the original behaviour
    motion_channels: int = 4


def get_video(video_reader, video_path, max_frames, sample, hevc_config: None | CVConfig = None,
              strict: bool = False):
    """
    :param strict: raise if the reader failed. The compressed-domain reader returns all-zero
        tensors on any error, which are indistinguishable from real data downstream, so a
        corrupt or unreadable video would otherwise train silently as a black clip. Leave it
        off to tolerate a few bad files, but never report numbers without checking how many
        failed (``tools/validate_data_pipeline.py`` counts them).
    """
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    video_mask = torch.ones((max_frames,), dtype=torch.int)
    if video_reader.__name__ in ["read_frames_compressed_domain"]:
        assert hevc_config is not None, "hevc_config should be set when using read_frames_compressed_domain"
        video, ok = video_reader(video_path,
                                 resample_num_gop=hevc_config.num_gop, resample_num_mv=hevc_config.num_mv,
                                 resample_num_res=hevc_config.num_res,
                                 with_residual=hevc_config.with_residual,
                                 pre_extract=hevc_config.use_pre_extract,
                                 sample=hevc_config.sample if hevc_config.sample == "pad" else sample,
                                 motion_channels=getattr(hevc_config, "motion_channels", 4))
    else:
        # the RGB readers return sampled frame indices as their second value and raise on
        # failure, so reaching here means success
        video, _ = video_reader(video_path, max_frames, sample)
        ok = True

    if not ok:
        if strict:
            raise RuntimeError(
                f"Failed to read {video_path}; the reader returned zero tensors. See "
                f"video_reader_error.log. Set strict=False to train through it."
            )
        GET_VIDEO_FAILURES.append(video_path)
    return video, video_mask
