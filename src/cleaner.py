"""Removal and replacement of typographic symbols typically produced by LLMs."""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

# character -> (replacement, human readable name)
SYMBOL_MAP: Dict[str, tuple[str, str]] = {
    # Dashes and hyphens
    "—": (" - ", "em dash"),
    "–": ("-", "en dash"),
    "‒": ("-", "figure dash"),
    "―": ("-", "horizontal bar"),
    "−": ("-", "minus sign"),
    "‐": ("-", "hyphen"),
    "‑": ("-", "non-breaking hyphen"),
    "­": ("", "soft hyphen"),
    # Zero-width and invisible characters
    "​": ("", "zero width space"),
    "‌": ("", "zero width non-joiner"),
    "‍": ("", "zero width joiner"),
    "⁠": ("", "word joiner"),
    "﻿": ("", "zero width no-break space (BOM)"),
    "᠎": ("", "mongolian vowel separator"),
    "؜": ("", "arabic letter mark"),
    "‎": ("", "left-to-right mark"),
    "‏": ("", "right-to-left mark"),
    "‪": ("", "left-to-right embedding"),
    "‫": ("", "right-to-left embedding"),
    "‬": ("", "pop directional formatting"),
    "‭": ("", "left-to-right override"),
    "‮": ("", "right-to-left override"),
    "⁡": ("", "function application"),
    "⁢": ("", "invisible times"),
    "⁣": ("", "invisible separator"),
    "⁤": ("", "invisible plus"),
    # Unusual spaces
    " ": (" ", "non-breaking space"),
    " ": (" ", "ogham space mark"),
    " ": (" ", "en quad"),
    " ": (" ", "em quad"),
    " ": (" ", "en space"),
    " ": (" ", "em space"),
    " ": (" ", "three-per-em space"),
    " ": (" ", "four-per-em space"),
    " ": (" ", "six-per-em space"),
    " ": (" ", "figure space"),
    " ": (" ", "punctuation space"),
    " ": (" ", "thin space"),
    " ": (" ", "hair space"),
    " ": (" ", "narrow no-break space"),
    " ": (" ", "medium mathematical space"),
    "　": (" ", "ideographic space"),
    " ": ("\n", "line separator"),
    " ": ("\n\n", "paragraph separator"),
    # Quotes and apostrophes
    "‘": ("'", "left single quotation mark"),
    "’": ("'", "right single quotation mark"),
    "‚": ("'", "single low-9 quotation mark"),
    "‛": ("'", "single high-reversed-9 quotation mark"),
    "“": ('"', "left double quotation mark"),
    "”": ('"', "right double quotation mark"),
    "„": ('"', "double low-9 quotation mark"),
    "‟": ('"', "double high-reversed-9 quotation mark"),
    "«": ('"', "left double angle quotation mark"),
    "»": ('"', "right double angle quotation mark"),
    "‹": ("'", "single left angle quotation mark"),
    "›": ("'", "single right angle quotation mark"),
    "′": ("'", "prime"),
    "″": ('"', "double prime"),
    # Punctuation and symbols
    "…": ("...", "horizontal ellipsis"),
    "•": ("-", "bullet"),
    "‣": ("-", "triangular bullet"),
    "●": ("-", "black circle"),
    "○": ("-", "white circle"),
    "▪": ("-", "black small square"),
    "·": ("-", "middle dot"),
    "⁃": ("-", "hyphen bullet"),
    "⁄": ("/", "fraction slash"),
    "©": ("(c)", "copyright sign"),
    "®": ("(r)", "registered sign"),
    "™": ("(tm)", "trade mark sign"),
    "→": ("->", "rightwards arrow"),
    "←": ("<-", "leftwards arrow"),
    "⇒": ("=>", "rightwards double arrow"),
    "×": ("x", "multiplication sign"),
}

_PATTERN = re.compile("|".join(re.escape(ch) for ch in SYMBOL_MAP))


@dataclass
class SymbolStat:
    symbol: str
    codepoint: str
    name: str
    action: str
    count: int


@dataclass
class CleaningResult:
    text: str
    counts: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def stats(self) -> List[SymbolStat]:
        """Per-symbol statistics, most frequent first."""
        result = []
        for symbol, count in self.counts.most_common():
            replacement, name = SYMBOL_MAP.get(symbol, ("", _unicode_name(symbol)))
            result.append(
                SymbolStat(
                    symbol=symbol,
                    codepoint=f"U+{ord(symbol):04X}",
                    name=name,
                    action="removed" if replacement == "" else f"replaced with {replacement!r}",
                    count=count,
                )
            )
        return result


def _unicode_name(char: str) -> str:
    try:
        return unicodedata.name(char).lower()
    except ValueError:
        return "unnamed format character"


def clean_text(text: str, normalize_whitespace: bool = True) -> CleaningResult:
    """Replace or remove LLM-typical symbols and report what was changed."""
    counts: Counter = Counter()

    def substitute(match: re.Match) -> str:
        symbol = match.group(0)
        counts[symbol] += 1
        return SYMBOL_MAP[symbol][0]

    cleaned = _PATTERN.sub(substitute, text)

    # Catch any remaining invisible formatting characters that are not in the map.
    leftovers = [ch for ch in cleaned if unicodedata.category(ch) == "Cf"]
    if leftovers:
        counts.update(leftovers)
        cleaned = "".join(ch for ch in cleaned if unicodedata.category(ch) != "Cf")

    if normalize_whitespace:
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()

    return CleaningResult(text=cleaned, counts=counts)
