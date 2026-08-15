"""Tests for the pipeline wiring, with translation and paraphrasing switched off.

Those two steps go over the network, so what is covered here is the orchestration around
them: which stages reach the verifier, and that a failing verification still returns the text.
"""

import pytest

from src import pipeline
from src.config import AppConfig
from src.pipeline import run_pipeline
from src.verifier import VerificationResult

ZWSP = "​"


@pytest.fixture
def config():
    """Cleaning only, so nothing in the test touches the network."""
    config = AppConfig()
    config.translation.enabled = False
    config.paraphrase.enabled = False
    return config


class TestCleaningOnly:
    def test_returns_the_cleaned_text(self, config):
        result = run_pipeline(f"a{ZWSP}b — c", config)
        assert result.final == "ab - c"
        assert result.steps == ["cleaning"]

    def test_every_step_disabled_returns_the_input_unchanged(self, config):
        config.cleaning.enabled = False
        result = run_pipeline(f"a{ZWSP}b", config)
        assert result.final == f"a{ZWSP}b"
        assert result.steps == []

    def test_cleaning_uses_the_configured_tiers(self, config):
        config.cleaning.tiers = ["carriers"]
        assert run_pipeline("a — b", config).final == "a — b"


class TestVerification:
    def test_off_by_default(self, config):
        assert run_pipeline("text", config).verification is None

    def test_scores_the_stages_that_differ(self, config, monkeypatch):
        seen = {}

        def fake_verify(stages, **kwargs):
            seen.update(stages)
            return VerificationResult()

        monkeypatch.setattr(pipeline, "verify", fake_verify)
        config.verification.enabled = True
        run_pipeline(f"a{ZWSP}b", config)
        # "cleaned" and "final" are the same text here, only one of them is worth scoring.
        assert seen == {"original": f"a{ZWSP}b", "cleaned": "ab"}

    def test_passes_the_detector_settings_and_the_token(self, config, monkeypatch):
        captured = {}

        def fake_verify(stages, **kwargs):
            captured.update(kwargs)
            return VerificationResult()

        monkeypatch.setattr(pipeline, "verify", fake_verify)
        config.verification.enabled = True
        config.verification.detector_repo = "some/detector"
        config.secrets.hf_token = "hf_xxx"
        run_pipeline("text", config)
        assert captured["detector_repo"] == "some/detector"
        assert captured["hf_token"] == "hf_xxx"

    def test_a_failure_does_not_cost_the_processed_text(self, config, monkeypatch):
        monkeypatch.setattr(
            pipeline, "verify", lambda stages, **kwargs: VerificationResult(error="no detector")
        )
        config.verification.enabled = True
        result = run_pipeline(f"a{ZWSP}b — c", config)
        assert result.final == "ab - c"
        assert result.verification.error == "no detector"
