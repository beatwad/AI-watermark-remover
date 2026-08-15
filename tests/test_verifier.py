"""Tests for the verification step.

The detector itself needs torch, which is an optional extra, so it is faked here. What is
tested is the contract around it: verification never raises, and a detector that cannot be
loaded degrades into a message instead of costing the user the processed text.
"""

from dataclasses import dataclass

import pytest

from src import verifier
from src.verifier import collect_stages, verify


@dataclass
class FakeDetection:
    score: float = 0.0
    verdict: str = "not watermarked"
    token_count: int = 250
    z_score: float = -1.0
    mean_g_value: float = 0.5
    reliable: bool = True


class FakeDetector:
    """Records what it was asked to score and answers with one result per text."""

    def __init__(self, results=None):
        self.results = results
        self.seen = []

    def detect_batch(self, texts):
        self.seen.append(texts)
        if self.results is not None:
            return self.results[: len(texts)]
        return [FakeDetection() for _ in texts]


@pytest.fixture
def detector(monkeypatch):
    fake = FakeDetector()
    monkeypatch.setattr(verifier, "_load_detector", lambda *args: fake)
    return fake


class TestCollectStages:
    def test_keeps_order_and_drops_empty_stages(self):
        stages = collect_stages(("original", "a"), ("cleaned", ""), ("final", "b"))
        assert list(stages) == ["original", "final"]

    def test_drops_stages_that_changed_nothing(self):
        # A disabled step leaves its stage identical to the one before it.
        stages = collect_stages(("original", "a"), ("cleaned", "a"), ("final", "b"))
        assert stages == {"original": "a", "final": "b"}

    def test_drops_none(self):
        assert collect_stages(("original", None), ("final", "b")) == {"final": "b"}


class TestVerify:
    def test_scores_every_stage_in_one_pass(self, detector):
        result = verify({"original": "one", "final": "two"}, detector_repo="repo")
        assert not result.error
        assert result.available
        assert [stage.stage for stage in result.stages] == ["original", "final"]
        assert detector.seen == [["one", "two"]]

    def test_maps_the_detection_onto_the_stage(self, monkeypatch):
        detection = FakeDetection(
            score=0.9998, verdict="watermarked", token_count=412, z_score=5.68, reliable=True
        )
        monkeypatch.setattr(
            verifier, "_load_detector", lambda *args: FakeDetector([detection])
        )
        stage = verify({"original": "text"}, detector_repo="repo").stages[0]
        assert (stage.score, stage.z_score, stage.verdict) == (0.9998, 5.68, "watermarked")
        assert stage.token_count == 412 and stage.reliable

    def test_blank_stages_are_not_scored(self, detector):
        result = verify({"original": "   ", "final": ""}, detector_repo="repo")
        assert result.error and not result.available
        assert detector.seen == []

    def test_no_stages_at_all(self, detector):
        assert verify({}, detector_repo="repo").error


class TestFailuresDegrade:
    """A failure must come back as an error string, never as an exception."""

    def test_missing_extra_is_reported_with_the_install_hint(self, monkeypatch):
        def missing_torch(*args):
            raise ImportError("No module named 'torch'")

        monkeypatch.setattr(verifier, "_load_detector", missing_torch)
        result = verify({"original": "text"}, detector_repo="repo")
        assert verifier.INSTALL_HINT in result.error
        assert not result.available

    def test_a_detector_that_will_not_load_is_reported(self, monkeypatch):
        def gated(*args):
            raise OSError("gated repo")

        monkeypatch.setattr(verifier, "_load_detector", gated)
        result = verify({"original": "text"}, detector_repo="private/detector")
        assert "private/detector" in result.error
        assert "gated repo" in result.error

    def test_a_scoring_failure_is_reported(self, monkeypatch):
        class Broken:
            def detect_batch(self, texts):
                raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(verifier, "_load_detector", lambda *args: Broken())
        result = verify({"original": "text"}, detector_repo="repo")
        assert "CUDA out of memory" in result.error
        assert not result.available


def test_importing_the_verifier_does_not_need_torch():
    """The Streamlit app installs without the detector extra, so the import must stay cheap.

    A subprocess, because torch may well be in sys.modules already: another test imports it.
    """
    import subprocess
    import sys
    from pathlib import Path

    subprocess.run(
        [sys.executable, "-c", "import sys, src.verifier; assert 'torch' not in sys.modules"],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
    )
