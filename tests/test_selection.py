"""Tests for picking the least watermarked paraphrase candidate."""

import pytest

from src import pipeline, verifier
from src.config import AppConfig
from src.paraphraser import Paraphraser
from src.pipeline import run_pipeline
from src.verifier import StageScore, VerificationResult, select_candidate

DETECTOR = {"detector_repo": "repo"}


def scores(*pairs):
    """A VerificationResult from (score, z_score) pairs, in candidate order."""
    return VerificationResult(
        stages=[
            StageScore(
                stage=f"candidate {number}",
                score=score,
                z_score=z,
                verdict="not watermarked",
                token_count=250,
                reliable=True,
            )
            for number, (score, z) in enumerate(pairs, 1)
        ]
    )


class TestSelectCandidate:
    def test_keeps_the_lowest_score(self, monkeypatch):
        monkeypatch.setattr(
            verifier, "verify", lambda stages, **kwargs: scores((0.9, 5.0), (0.1, 2.0), (0.5, 0.0))
        )
        selection = select_candidate(["a", "b", "c"], **DETECTOR)
        assert selection.text == "b"
        assert selection.index == 1
        assert len(selection.scores) == 3

    def test_z_score_breaks_a_tie_on_the_posterior(self, monkeypatch):
        # The posterior saturates at 0.0000 once the watermark is gone, which is the common case.
        monkeypatch.setattr(
            verifier, "verify", lambda stages, **kwargs: scores((0.0, 1.4), (0.0, -0.8), (0.0, 0.3))
        )
        assert select_candidate(["a", "b", "c"], **DETECTOR).text == "b"

    def test_the_posterior_still_wins_over_the_z_score(self, monkeypatch):
        monkeypatch.setattr(verifier, "verify", lambda stages, **kwargs: scores((0.4, -9.0), (0.0, 9.0)))
        assert select_candidate(["a", "b"], **DETECTOR).text == "b"

    def test_one_candidate_is_not_scored_at_all(self, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("the detector must not be loaded for a single candidate")

        monkeypatch.setattr(verifier, "verify", fail)
        selection = select_candidate(["only"], **DETECTOR)
        assert selection.text == "only" and selection.index == 0

    def test_empty_candidates_are_dropped(self, monkeypatch):
        seen = {}

        def fake_verify(stages, **kwargs):
            seen.update(stages)
            return scores((0.2, 1.0), (0.1, 1.0))

        monkeypatch.setattr(verifier, "verify", fake_verify)
        selection = select_candidate(["a", "   ", "b"], **DETECTOR)
        assert list(seen.values()) == ["a", "b"]
        assert selection.text == "b"

    def test_all_candidates_empty(self, monkeypatch):
        monkeypatch.setattr(verifier, "verify", lambda stages, **kwargs: scores())
        selection = select_candidate(["", "  "], **DETECTOR)
        assert selection.error and selection.text == ""

    def test_a_detector_failure_keeps_the_first_candidate(self, monkeypatch):
        monkeypatch.setattr(
            verifier, "verify", lambda stages, **kwargs: VerificationResult(error="no detector")
        )
        selection = select_candidate(["first", "second"], **DETECTOR)
        assert selection.text == "first"
        assert selection.error == "no detector"
        assert not selection.scores


class TestParaphraseCandidates:
    """The batch call, without building a real chat model."""

    @staticmethod
    def paraphraser(replies):
        instance = object.__new__(Paraphraser)
        instance.config = AppConfig().paraphrase
        instance.chain = type(
            "FakeChain",
            (),
            {"batch": staticmethod(lambda payloads: replies[: len(payloads)])},
        )()
        return instance

    def test_asks_for_one_request_per_candidate(self):
        instance = self.paraphraser([" one ", "two", "three"])
        assert instance.paraphrase_candidates("text", "en", 3) == ["one", "two", "three"]

    def test_paraphrase_returns_a_single_string(self):
        assert self.paraphraser(["only"]).paraphrase("text") == "only"

    def test_a_count_below_one_still_produces_one(self):
        assert self.paraphraser(["only"]).paraphrase_candidates("text", "en", 0) == ["only"]


class TestPipelineWiring:
    @pytest.fixture
    def config(self):
        config = AppConfig()
        config.translation.enabled = False
        config.paraphrase.enabled = True
        return config

    def test_candidates_need_verification(self, config, monkeypatch):
        """Without a detector the extra requests would only cost money."""
        asked = []

        def fake_candidates(self, text, language_code="en", count=1):
            asked.append(count)
            return ["paraphrased"]

        monkeypatch.setattr(Paraphraser, "__init__", lambda self, *args: None)
        monkeypatch.setattr(Paraphraser, "paraphrase_candidates", fake_candidates)
        config.paraphrase.candidates = 5
        config.verification.enabled = False
        run_pipeline("text", config)
        assert asked == [1]

    def test_the_selected_candidate_becomes_the_result(self, config, monkeypatch):
        monkeypatch.setattr(Paraphraser, "__init__", lambda self, *args: None)
        monkeypatch.setattr(
            Paraphraser,
            "paraphrase_candidates",
            lambda self, text, language_code="en", count=1: ["worse", "better"],
        )
        monkeypatch.setattr(
            pipeline,
            "select_candidate",
            lambda candidates, **kwargs: verifier.Selection(text="better", index=1),
        )
        monkeypatch.setattr(pipeline, "verify", lambda stages, **kwargs: VerificationResult())
        config.paraphrase.candidates = 2
        config.verification.enabled = True
        result = run_pipeline("text", config)
        assert result.final == "better"
        assert result.selection.index == 1
