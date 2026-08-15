"""Removal and replacement of typographic symbols typically produced by LLMs.

The symbols are grouped into tiers, because they are not equally safe to touch:

- ``carriers`` are invisible characters. Removing them loses no information and they are the
  ones actually used to carry a mark, so this tier is the reason the tool exists.
- ``typography`` are the visible tells: em dashes, curly quotes, an ellipsis character. A human
  writer produces them too, but an LLM produces them far more consistently.
- ``punctuation`` is ordinary typography that is not a tell at all. Replacing it mangles
  quotations, maths and other languages, so it is off by default.

Over-normalizing is itself a fingerprint: text with no curly quotes, no em dashes and ``(c)``
instead of a copyright sign does not read as human, it reads as scrubbed.
"""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

from loguru import logger

ZERO_WIDTH_JOINER = "‍"
# Variation selectors and skin tone modifiers sit between an emoji and its joiner.
_EMOJI_GLUE = {"︎", "️", "\U0001f3fb", "\U0001f3fc", "\U0001f3fd", "\U0001f3fe", "\U0001f3ff"}

# character -> (replacement, human readable name)
Symbols = Dict[str, Tuple[str, str]]

# Invisible characters. Always safe to drop, and the tier a watermark actually rides on.
CARRIERS: Symbols = {
    # Zero-width and invisible characters
    "​": ("", "zero width space"),
    # NOTE: a zero width non-joiner is load-bearing in Persian and Devanagari. It is treated as
    # a carrier anyway, which is wrong for text in those scripts.
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
    "­": ("", "soft hyphen"),
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
}

# Visible characters an LLM overuses. Replacing them is lossy but rarely changes meaning.
TYPOGRAPHY: Symbols = {
    # Dashes and hyphens
    "—": (" - ", "em dash"),
    "–": ("-", "en dash"),
    "‒": ("-", "figure dash"),
    "―": ("-", "horizontal bar"),
    "−": ("-", "minus sign"),
    "‐": ("-", "hyphen"),
    "‑": ("-", "non-breaking hyphen"),
    # Quotes and apostrophes
    "‘": ("'", "left single quotation mark"),
    "’": ("'", "right single quotation mark"),
    "‚": ("'", "single low-9 quotation mark"),
    "‛": ("'", "single high-reversed-9 quotation mark"),
    "“": ('"', "left double quotation mark"),
    "”": ('"', "right double quotation mark"),
    "„": ('"', "double low-9 quotation mark"),
    "‟": ('"', "double high-reversed-9 quotation mark"),
    "′": ("'", "prime"),
    "″": ('"', "double prime"),
    "…": ("...", "horizontal ellipsis"),
}

# Ordinary typography, not a tell. Off by default: these carry meaning of their own.
PUNCTUATION: Symbols = {
    # Language-specific quotation marks
    "«": ('"', "left double angle quotation mark"),
    "»": ('"', "right double angle quotation mark"),
    "‹": ("'", "single left angle quotation mark"),
    "›": ("'", "single right angle quotation mark"),
    # Bullets
    "•": ("-", "bullet"),
    "‣": ("-", "triangular bullet"),
    "●": ("-", "black circle"),
    "○": ("-", "white circle"),
    "▪": ("-", "black small square"),
    "·": ("-", "middle dot"),
    "⁃": ("-", "hyphen bullet"),
    # Symbols
    "⁄": ("/", "fraction slash"),
    "©": ("(c)", "copyright sign"),
    "®": ("(r)", "registered sign"),
    "™": ("(tm)", "trade mark sign"),
    "→": ("->", "rightwards arrow"),
    "←": ("<-", "leftwards arrow"),
    "⇒": ("=>", "rightwards double arrow"),
    "×": ("x", "multiplication sign"),
}

TIERS: Dict[str, Symbols] = {
    "carriers": CARRIERS,
    "typography": TYPOGRAPHY,
    "punctuation": PUNCTUATION,
}
TIER_NAMES: Tuple[str, ...] = tuple(TIERS)
DEFAULT_TIERS: Tuple[str, ...] = ("carriers", "typography")

# Every known symbol, used to describe what was changed regardless of the active tiers.
SYMBOL_MAP: Symbols = {**CARRIERS, **TYPOGRAPHY, **PUNCTUATION}


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


@lru_cache(maxsize=len(TIERS) ** 2)
def _active(tiers: Tuple[str, ...]) -> Tuple[Symbols, re.Pattern]:
    """The symbol map and pattern for a tier selection, built once per selection."""
    symbols: Symbols = {}
    for tier in tiers:
        symbols.update(TIERS[tier])
    return symbols, re.compile("|".join(re.escape(ch) for ch in symbols))


def _is_emoji(char: str) -> bool:
    return unicodedata.category(char) in {"So", "Sk"} or 0x1F000 <= ord(char) <= 0x1FAFF


def _joins_emoji(text: str, index: int) -> bool:
    """True when the zero width joiner at `index` glues two emoji into one grapheme."""
    left = index - 1
    while left >= 0 and text[left] in _EMOJI_GLUE:
        left -= 1
    right = index + 1
    while right < len(text) and text[right] in _EMOJI_GLUE:
        right += 1
    return (
        left >= 0
        and right < len(text)
        and _is_emoji(text[left])
        and _is_emoji(text[right])
    )


def _normalize_tiers(tiers: Iterable[str]) -> Tuple[str, ...]:
    """Validate the tier names and put them in a stable order."""
    requested = set(tiers)
    unknown = requested - set(TIERS)
    if unknown:
        raise ValueError(
            f"Unknown cleaning tier(s) {sorted(unknown)}, expected any of {list(TIER_NAMES)}"
        )
    return tuple(name for name in TIER_NAMES if name in requested)


def clean_text(
    text: str,
    normalize_whitespace: bool = True,
    tiers: Sequence[str] = DEFAULT_TIERS,
) -> CleaningResult:
    """Replace or remove LLM-typical symbols of the selected tiers and report what changed."""
    active_tiers = _normalize_tiers(tiers)
    counts: Counter = Counter()

    if not active_tiers:
        logger.warning("Cleaning ran with no tier selected, the text is returned unchanged")
        return CleaningResult(text=text, counts=counts)

    symbols, pattern = _active(active_tiers)

    def substitute(match: re.Match) -> str:
        symbol = match.group(0)
        if symbol == ZERO_WIDTH_JOINER and _joins_emoji(text, match.start()):
            return symbol
        counts[symbol] += 1
        return symbols[symbol][0]

    cleaned = pattern.sub(substitute, text)

    # Catch any remaining invisible formatting characters that are not in the map.
    if "carriers" in active_tiers:
        # Removal is by position, a guarded joiner must not be dropped in place of another one.
        leftovers = {
            index
            for index, char in enumerate(cleaned)
            if unicodedata.category(char) == "Cf"
            and not (char == ZERO_WIDTH_JOINER and _joins_emoji(cleaned, index))
        }
        if leftovers:
            logger.debug(
                "Removed {} formatting characters that are not in the symbol map: {}",
                len(leftovers),
                sorted({f"U+{ord(cleaned[index]):04X}" for index in leftovers}),
            )
            counts.update(cleaned[index] for index in leftovers)
            cleaned = "".join(
                char for index, char in enumerate(cleaned) if index not in leftovers
            )

    if normalize_whitespace:
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()

    result = CleaningResult(text=cleaned, counts=counts)
    logger.info(
        "Cleaned {} characters down to {}, {} symbols replaced or removed, tiers: {}",
        len(text),
        len(cleaned),
        result.total,
        list(active_tiers),
    )
    return result
