"""Tests for the symbol cleaner, with an emphasis on what it must leave alone."""

import pytest

from src.cleaner import DEFAULT_TIERS, TIER_NAMES, clean_text

ZWSP = "​"
ZWJ = "‍"
BOM = "﻿"
LRM = "‎"
NBSP = " "
FAMILY = "👨‍👩‍👧‍👦"


class TestCarriers:
    """Invisible characters are the point of the tool, they always go."""

    @pytest.mark.parametrize("carrier", [ZWSP, BOM, LRM, "⁠", "­", "؜"])
    def test_invisible_characters_are_removed(self, carrier):
        assert clean_text(f"a{carrier}b").text == "ab"

    def test_exotic_spaces_become_plain_spaces(self):
        assert clean_text(f"a{NBSP}b", normalize_whitespace=False).text == "a b"
        assert clean_text("a　b", normalize_whitespace=False).text == "a b"

    def test_unmapped_format_characters_are_swept(self):
        # U+E0041 is a language tag character, not in the symbol map.
        assert clean_text("a\U000e0041b").text == "ab"

    def test_sweep_is_skipped_when_the_tier_is_off(self):
        assert clean_text("a\U000e0041b", tiers=["typography"]).text == "a\U000e0041b"


class TestEmojiJoiner:
    """A zero width joiner glues emoji together, removing it destroys the grapheme."""

    def test_family_emoji_survives(self):
        assert clean_text(f"family: {FAMILY} done").text == f"family: {FAMILY} done"

    def test_joiner_between_letters_is_removed(self):
        assert clean_text(f"a{ZWJ}b").text == "ab"

    def test_joiner_after_a_variation_selector_survives(self):
        heart = "❤️‍🔥"  # heart + VS16 + ZWJ + fire
        assert clean_text(heart).text == heart

    def test_preserved_joiner_is_not_counted(self):
        result = clean_text(FAMILY)
        assert result.total == 0

    def test_removed_joiner_is_counted_and_emoji_one_is_kept(self):
        result = clean_text(f"{FAMILY} a{ZWJ}b")
        assert result.text == f"{FAMILY} ab"
        assert result.counts[ZWJ] == 1


class TestTypography:
    def test_dashes_and_quotes_are_replaced_by_default(self):
        assert clean_text("a — b").text == "a - b"
        assert clean_text("“quoted”").text == '"quoted"'
        assert clean_text("wait…").text == "wait..."

    def test_tier_can_be_turned_off(self):
        assert clean_text("a — b", tiers=["carriers"]).text == "a — b"


class TestPunctuation:
    """Ordinary typography must survive the default settings."""

    @pytest.mark.parametrize(
        "text",
        [
            "Er sagte «Hallo»",
            "3 × 4 → 12",
            "© 2026 Someone",
            "l·l",
            "• first item",
        ],
    )
    def test_left_alone_by_default(self, text):
        assert clean_text(text).text == text

    def test_replaced_when_the_tier_is_enabled(self):
        tiers = [*DEFAULT_TIERS, "punctuation"]
        result = clean_text("«Hallo» 3 × 4 → 12. © 2026. l·l", tiers=tiers)
        assert result.text == '"Hallo" 3 x 4 -> 12. (c) 2026. l-l'


class TestWhitespace:
    def test_normalization_collapses_runs_and_trims(self):
        assert clean_text("a  b \nc\n\n\n\nd  ").text == "a b\nc\n\nd"

    def test_normalization_can_be_turned_off(self):
        assert clean_text("a  b  ", normalize_whitespace=False).text == "a  b  "


class TestTierSelection:
    def test_unknown_tier_is_rejected(self):
        with pytest.raises(ValueError, match="typograhpy"):
            clean_text("text", tiers=["typograhpy"])

    def test_empty_selection_returns_the_text_unchanged(self):
        result = clean_text(f"a{ZWSP}b — c", tiers=[])
        assert result.text == f"a{ZWSP}b — c"
        assert result.total == 0

    def test_order_of_the_selection_does_not_matter(self):
        text = "«x» a — b"
        assert (
            clean_text(text, tiers=["punctuation", "carriers", "typography"]).text
            == clean_text(text, tiers=list(TIER_NAMES)).text
        )


class TestStats:
    def test_counts_and_actions_describe_what_happened(self):
        result = clean_text(f"a{ZWSP}b — c — d")
        assert result.counts[ZWSP] == 1
        assert result.counts["—"] == 2
        assert result.total == 3

        by_symbol = {stat.symbol: stat for stat in result.stats}
        assert by_symbol[ZWSP].action == "removed"
        assert by_symbol[ZWSP].codepoint == "U+200B"
        assert by_symbol["—"].name == "em dash"
        assert by_symbol["—"].count == 2

    def test_clean_text_does_not_mutate_the_input(self):
        text = f"a{ZWSP}b"
        clean_text(text)
        assert text == f"a{ZWSP}b"
