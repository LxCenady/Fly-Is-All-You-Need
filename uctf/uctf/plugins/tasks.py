"""Tasks: the data a connectome model is trained and evaluated on."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.registry import register


@dataclass
class Split:
    part: str        # the text used (training + validation)
    chars: list      # character set; ids index into it
    ids: np.ndarray  # the text as character ids (characters outside the set dropped)
    n_tr: int        # the first n_tr ids are for training, the rest for validation
    dropped: int = 0


def load_split(path, train_chars=None, val_chars=None, offset=0, vocab=None,
               max_vocab=256, val_fraction_of_train=0.1) -> Split:
    """A training part of a text and the validation part that follows it.

    train_chars None: all but the validation part (default 10 % of the text,
    1k-100k characters). vocab: a vocab.json ({"chars": ...}) or any text file
    whose distinct characters are used; else the max_vocab most frequent
    characters of the selected text.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n")[offset:]
    if train_chars is None:
        n_val = val_chars or max(1000, min(len(text) // 10, 100_000))
        n_tr = len(text) - n_val
    else:
        n_tr = train_chars
        n_val = val_chars or max(1000, int(n_tr * val_fraction_of_train))
    if n_tr < 1000 or n_tr + n_val > len(text):
        raise ValueError(f"not enough text: {len(text):,} characters after "
                         f"offset {offset}, need {n_tr:,} + {n_val:,}")
    part = text[:n_tr + n_val]
    freq = Counter(part)
    chars = _charset(vocab, freq, max_vocab)
    keep = set(chars)
    dropped = sum(v for c, v in freq.items() if c not in keep)
    idx = {c: i for i, c in enumerate(chars)}
    ids = np.asarray([idx[c] for c in part if c in idx], np.int64)
    n_tr -= sum(1 for c in part[:n_tr] if c not in idx)
    return Split(part, chars, ids, n_tr, dropped)


def _charset(vocab, freq, max_vocab) -> list:
    if not vocab:
        return sorted(c for c, _ in freq.most_common(max_vocab))
    src = Path(vocab).read_text(encoding="utf-8")
    try:
        return list(json.loads(src)["chars"])
    except (ValueError, KeyError, TypeError):
        return sorted(set(src))


@register("task", "next_char")
class NextChar:
    """Predict the next character of a text. Params: path, train_chars,
    val_chars, offset, vocab, max_vocab, val_fraction_of_train."""

    def __init__(self, **params):
        self.params = params

    def split(self) -> Split:
        return load_split(**self.params)
