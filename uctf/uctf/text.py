"""Text: a training part and the validation part that follows it, and the character set."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Split:
    part: str             # the text used (training + validation)
    chars: list           # character set; ids index into it
    ids: np.ndarray       # the text as character ids (characters outside the set dropped)
    n_tr: int             # the first n_tr ids are training characters, the rest validation
    dropped: int = 0


def load_split(path, train_chars=None, val_chars=None, offset=0, vocab=None, max_vocab=256,
               val_fraction_of_train=0.1) -> Split:
    """train_chars None: all but the validation part (default 10% of the text, 1k-100k).
    vocab: a vocab.json ({"chars": ...}) or any text file whose distinct characters are used;
    otherwise the max_vocab most frequent characters of the selected text."""
    text = Path(path).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")[offset:]
    if train_chars is None:
        n_val = val_chars or max(1000, min(len(text) // 10, 100_000))
        n_tr = len(text) - n_val
    else:
        n_tr = train_chars
        n_val = val_chars or max(1000, int(n_tr * val_fraction_of_train))
    if n_tr < 1000 or n_tr + n_val > len(text):
        raise ValueError(f"not enough text: {len(text):,} characters after offset {offset}, "
                         f"need {n_tr:,} for training + {n_val:,} for validation")
    part = text[:n_tr + n_val]
    freq = Counter(part)
    if vocab:
        src = Path(vocab).read_text(encoding="utf-8")
        try:
            chars = list(json.loads(src)["chars"])
        except (ValueError, KeyError, TypeError):
            chars = sorted(set(src))
    else:
        chars = sorted(c for c, _ in freq.most_common(max_vocab))
    keep = set(chars)
    dropped = sum(v for c, v in freq.items() if c not in keep)
    idx = {c: i for i, c in enumerate(chars)}
    ids = np.asarray([idx[c] for c in part if c in idx], np.int64)
    n_tr = n_tr - sum(1 for c in part[:n_tr] if c not in idx)
    return Split(part, chars, ids, n_tr, dropped)
