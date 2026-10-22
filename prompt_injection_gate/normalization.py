"""
normalization.py
================
Bounded, defensive normalisation for DETECTION ONLY (spec sections 6A, G).

Two jobs, both non-destructive to the original prompt:
  1. `normalize(text)` -> a folded view for rule matching (NFKC, confusables
     mapped to ASCII, zero-width and bidi controls stripped, whitespace run
     collapsed) plus structured signals about what was found.
  2. `safe_decode(text)` -> candidate plaintexts for base64/hex/rot13 segments,
     DECODED AS DATA and returned for inspection. Nothing is ever executed, no
     URL is fetched, decoding depth is 1, and output length is capped.

The original prompt is never mutated; the returned `safe_prompt_for_next_stage`
in a SAFE result is always the untouched input.
"""
from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata
from dataclasses import dataclass, field

# A small, explicit confusables map: Cyrillic/Greek look-alikes and common
# fullwidth forms that NFKC does not fold to ASCII. Not exhaustive by design --
# it raises the cost of trivial homoglyph evasion, it is not a guarantee.
_CONFUSABLES = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "х": "x", "у": "y", "і": "i", "ԁ": "d", "н": "h",
    "ο": "o", "α": "a", "ε": "e", "ρ": "p", "Α": "A",
    "Β": "B", "Ε": "E", "Ο": "O", "А": "A", "Е": "E",
    "О": "O", "Р": "P", "С": "C", "Х": "X",
}
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_BIDI = dict.fromkeys(map(ord, "‪‫‬‭‮⁦⁧⁨⁩"), None)

_B64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_HEX = re.compile(r"(?:[0-9a-fA-F]{2}[\s:]?){16,}")
_MAX_DECODE_OUT = 4000


@dataclass
class Normalized:
    folded: str                                   # lower-cased, confusables-folded, ws-collapsed
    cleaned: str                                  # case-preserving (for format regexes like key ids)
    signals: dict = field(default_factory=dict)   # structural flags for the rules engine


def normalize(text: str) -> Normalized:
    if not isinstance(text, str):
        text = str(text)
    had_zw = any(ord(c) in _ZERO_WIDTH for c in text)
    had_bidi = any(ord(c) in _BIDI for c in text)
    # strip zero-width + bidi controls, then NFKC fold
    t = text.translate(_ZERO_WIDTH).translate(_BIDI)
    t = unicodedata.normalize("NFKC", t)
    confusables = sum(1 for c in t if c in _CONFUSABLES)
    t = "".join(_CONFUSABLES.get(c, c) for c in t)
    cleaned = re.sub(r"\s+", " ", t).strip()
    folded = cleaned.lower()

    non_ascii = sum(1 for c in text if ord(c) > 127)
    letters = [c for c in text if c.isalpha()]
    non_latin = sum(1 for c in letters if ord(c) > 0x024F)
    signals = {
        "had_zero_width": had_zw,
        "had_bidi_control": had_bidi,
        "confusable_chars": confusables,
        "non_ascii_chars": non_ascii,
        "non_latin_letter_ratio": round(non_latin / len(letters), 3) if letters else 0.0,
        "length": len(text),
    }
    return Normalized(folded=folded, cleaned=cleaned, signals=signals)


def _rot13(s: str) -> str:
    return codecs.decode(s, "rot_13")


def _try_b64(seg: str) -> str | None:
    pad = seg + "=" * (-len(seg) % 4)
    try:
        raw = base64.b64decode(pad, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        txt = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    printable = sum(c.isprintable() or c.isspace() for c in txt)
    if len(txt) >= 4 and printable / len(txt) > 0.9:
        return txt[:_MAX_DECODE_OUT]
    return None


def _try_hex(seg: str) -> str | None:
    hexchars = re.sub(r"[\s:]", "", seg)
    if len(hexchars) % 2 or len(hexchars) < 8:
        return None
    try:
        raw = bytes.fromhex(hexchars)
        txt = raw.decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    printable = sum(c.isprintable() or c.isspace() for c in txt)
    if printable / len(txt) > 0.9:
        return txt[:_MAX_DECODE_OUT]
    return None


@dataclass
class Decoded:
    kind: str          # "base64" | "hex" | "rot13"
    source_span: str   # the encoded slice (truncated)
    plaintext: str     # decoded DATA, never executed


def safe_decode(text: str) -> list[Decoded]:
    """Return candidate decodings of encoded segments. Depth 1, data only."""
    found: list[Decoded] = []
    if not isinstance(text, str) or not text:
        return found
    for m in _B64.finditer(text):
        pt = _try_b64(m.group())
        if pt is not None:
            found.append(Decoded("base64", m.group()[:80], pt))
    for m in _HEX.finditer(text):
        pt = _try_hex(m.group())
        if pt is not None:
            found.append(Decoded("hex", m.group()[:80], pt))
    # rot13 only on the whole text when it is mostly alphabetic (cheap, bounded)
    if 8 <= len(text) <= _MAX_DECODE_OUT:
        alpha = sum(c.isalpha() for c in text)
        if alpha / len(text) > 0.6:
            found.append(Decoded("rot13", text[:80], _rot13(text)[:_MAX_DECODE_OUT]))
    return found[:16]
