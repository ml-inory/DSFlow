"""Text front-end: phonemization (g2p-en) with a character-level fallback."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Sequence

try:
    from g2p_en import G2p

    _G2P: Optional[G2p] = G2p()
except Exception:  # g2p_en optional; char fallback keeps the pipeline runnable
    _G2P = None


PAD, BOS, EOS, UNK = range(4)
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

_STRESS_RE = re.compile(r"\d$")
_CHAR_RE = re.compile(r"[^a-z0-9]")


def _strip_stress(phone: str) -> str:
    return _STRESS_RE.sub("", phone)


def phonemize(text: str) -> Optional[List[str]]:
    """Return ARPAbet phones for *text* (word boundaries removed), or None if g2p is unavailable."""
    if _G2P is None:
        return None
    try:
        phones = _G2P(text)
    except Exception:
        return None
    # Drop punctuation and spaces; DSFlow aligns every phone to a mel span.
    return [_strip_stress(p) for p in phones if p not in {" ", ",", ".", "?", "!", ";", ":", "'", '"'}]


def _char_units(text: str) -> List[str]:
    lowered = _CHAR_RE.sub("", text.lower())
    return [c for c in lowered if c != " "]


class TextTokenizer:
    """Maps phonemes (or chars) to ids with <pad>/<bos>/<eos>/<unk> specials."""

    def __init__(self, symbols: Sequence[str]):
        seen = set(SPECIALS)
        self.symbols = list(SPECIALS) + [s for s in symbols if s not in seen]
        self.stoi = {s: i for i, s in enumerate(self.symbols)}

    @property
    def vocab_size(self) -> int:
        return len(self.symbols)

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, phones: Optional[List[str]] = None, add_specials: bool = True) -> List[int]:
        if phones is None:
            phones = phonemize(text)
        units = phones if phones is not None else _char_units(text)
        ids = [self.stoi.get(u, UNK) for u in units]
        if add_specials:
            ids = [BOS] + ids + [EOS]
        return ids

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.symbols[i] if 0 <= i < len(self.symbols) else "<unk>" for i in ids]

    @classmethod
    def from_corpus(cls, texts: Sequence[str], use_phonemes: bool = True) -> "TextTokenizer":
        units = set()
        for t in texts:
            if use_phonemes:
                phones = phonemize(t)
            else:
                phones = None
            units.update(phones if phones is not None else _char_units(t))
        return cls(sorted(units))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"symbols": self.symbols}))

    @classmethod
    def load(cls, path) -> "TextTokenizer":
        return cls(json.loads(Path(path).read_text())["symbols"])
