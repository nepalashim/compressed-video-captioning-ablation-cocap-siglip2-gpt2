# -*- coding: utf-8 -*-
# @Project : CoCap
# @File    : tokenizers.py
"""
Text tokenizer abstraction.

The baseline CoCap model is tied to the CLIP BPE vocabulary (49408 tokens, ``[CLS]``/``[EOS]``
special tokens). Swapping the caption decoder for GPT-2 requires the GPT-2 BPE vocabulary
(50257 tokens, a single ``<|endoftext|>`` token used for both BOS and EOS). Both are exposed
here behind one interface so the dataset classes stay decoder-agnostic.

Each tokenizer produces the three tensors the caption head consumes:

``input_ids``     teacher-forcing input, left-aligned, padded to ``max_words``
``input_mask``    1 for real tokens, 0 for padding
``input_labels``  ``input_ids`` shifted left by one, ``ignore_index`` on positions
                  that must not contribute to the loss
"""

__all__ = [
    "TextTokenizer",
    "ClipTextTokenizer",
    "GPT2TextTokenizer",
    "build_tokenizer",
    "clip_tokenizer_cfg",
    "gpt2_tokenizer_cfg",
]

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
from hydra_zen import builds

logger = logging.getLogger(__name__)


class TextTokenizer(ABC):
    """Common interface for the caption tokenizers."""

    #: size of the output softmax
    vocab_size: int
    #: token used to start greedy decoding
    bos_id: int
    #: token that terminates a caption
    eos_id: int
    #: token used to pad ``input_ids`` up to ``max_words``
    pad_id: int
    #: label value that :class:`~cocap.modeling.loss.LabelSmoothingLoss` skips
    ignore_index: int

    @abstractmethod
    def encode(self, sentence: str, max_words: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(input_ids, input_mask, input_labels)``, each of shape ``(max_words,)``."""

    @abstractmethod
    def decode(self, ids: Union[List[int], torch.Tensor]) -> str:
        """Turn a generated id sequence back into a caption, stopping at the first EOS."""


class ClipTextTokenizer(TextTokenizer):
    """CLIP BPE tokenizer — reproduces the baseline CoCap text pipeline exactly.

    Padding uses id ``0`` and the loss ignores label ``0``, which is what the original
    implementation relies on.
    """

    def __init__(self):
        from cocap.modules.clip import clip

        self._clip = clip
        self.vocab_size = 49408
        self.bos_id = self.vocab_size - 2  # 49406 <|startoftext|>
        self.eos_id = self.vocab_size - 1  # 49407 <|endoftext|>
        self.pad_id = 0
        self.ignore_index = 0

    def encode(self, sentence: str, max_words: int):
        input_ids = self._clip.tokenize(sentence, context_length=max_words, truncate=True)[0]
        input_mask = torch.zeros(max_words, dtype=torch.long)
        input_mask[:len(self._clip._tokenizer.encode(sentence)) + 2] = 1
        input_labels = torch.cat((input_ids[1:], torch.IntTensor([self.ignore_index])))
        return input_ids, input_mask, input_labels

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        text = self._clip._tokenizer.decode(ids)
        words = text.split(" ")
        out = []
        for i, w in enumerate(words):
            if i == 0:
                out.append(w.split(">")[-1])  # strip a leading <|startoftext|>
            elif "<|endoftext|>" in w:
                break
            else:
                out.append(w)
        return " ".join(out)


class GPT2TextTokenizer(TextTokenizer):
    """GPT-2 BPE tokenizer.

    GPT-2 has no dedicated BOS or PAD token, so ``<|endoftext|>`` (50256) serves as both the
    decode-start symbol and the caption terminator, and padding reuses it too. Because that id
    is a *real* token the model must be able to emit, padding cannot be identified by id — so
    ``input_labels`` uses ``-100`` on every position that should not contribute to the loss.
    """

    def __init__(self, pretrained_model_name_or_path: str = "gpt2"):
        from transformers import GPT2TokenizerFast

        self._tok = GPT2TokenizerFast.from_pretrained(pretrained_model_name_or_path)
        self._tok.pad_token = self._tok.eos_token
        self.vocab_size = len(self._tok)  # 50257
        self.bos_id = self._tok.eos_token_id  # 50256
        self.eos_id = self._tok.eos_token_id  # 50256
        self.pad_id = self._tok.eos_token_id
        self.ignore_index = -100

    def encode(self, sentence: str, max_words: int):
        # <|endoftext|> acts as BOS; the caption is terminated by a second <|endoftext|>
        body = self._tok.encode(sentence.strip(), add_special_tokens=False)
        body = body[:max_words - 2]
        ids = [self.bos_id] + body + [self.eos_id]
        n_real = len(ids)

        input_ids = torch.full((max_words,), self.pad_id, dtype=torch.long)
        input_ids[:n_real] = torch.tensor(ids, dtype=torch.long)

        input_mask = torch.zeros(max_words, dtype=torch.long)
        input_mask[:n_real] = 1

        # next-token targets; everything past the real sequence is ignored
        input_labels = torch.full((max_words,), self.ignore_index, dtype=torch.long)
        input_labels[:n_real - 1] = input_ids[1:n_real]
        return input_ids, input_mask, input_labels

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        # drop the leading BOS, then stop at the first EOS
        if ids and ids[0] == self.bos_id:
            ids = ids[1:]
        if self.eos_id in ids:
            ids = ids[:ids.index(self.eos_id)]
        return self._tok.decode(ids, skip_special_tokens=True).strip()


_REGISTRY = {"clip": ClipTextTokenizer, "gpt2": GPT2TextTokenizer}
_CACHE = {}


def build_tokenizer(name: str = "clip", **kwargs) -> TextTokenizer:
    """Build (and memoize) a tokenizer by name.

    Dataset workers each construct one, so the cache keeps repeated vocabulary loads cheap.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown tokenizer {name!r}; available: {sorted(_REGISTRY)}")
    key = (name, tuple(sorted(kwargs.items())))
    if key not in _CACHE:
        _CACHE[key] = _REGISTRY[name](**kwargs)
        logger.debug("Built tokenizer %s (vocab_size=%d)", name, _CACHE[key].vocab_size)
    return _CACHE[key]


# Build configs for organizing modules with hydra
clip_tokenizer_cfg = builds(ClipTextTokenizer, populate_full_signature=True)
gpt2_tokenizer_cfg = builds(GPT2TextTokenizer, populate_full_signature=True)
