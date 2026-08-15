"""Tests for the detector itself. Skipped unless the detector extra is installed.

The scoring tests download the detector and its tokenizer from the Hub, so they are opt-in:

    RUN_DETECTOR_TESTS=1 uv run pytest tests/test_detector.py
"""

import ast
import inspect
import json
import os

import pytest

pytest.importorskip("torch", reason="the detector extra is not installed")

from watermark_detector.detector import (  # noqa: E402  imported after the skip
    UNGATED_GEMMA_TOKENIZER,
    SynthIDDetector,
)

needs_hub = pytest.mark.skipif(
    not os.getenv("RUN_DETECTOR_TESTS"),
    reason="downloads from the Hub, set RUN_DETECTOR_TESTS=1 to run",
)


def notebook_texts(path="watermark_detector/detect_watermark.ipynb"):
    """The sample texts the README's results table was measured on."""
    texts = {}
    for cell in json.load(open(path))["cells"]:
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                texts[node.targets[0].id] = node.value.value
    return texts


def test_the_default_device_is_not_taken_from_availability():
    """The g-function is device-dependent, so cuda must never be picked automatically.

    On cuda the detector scores the known watermarked sample at 0.0000 instead of 1.0000,
    and it does so quietly, which is why this is pinned here rather than left to review.
    """
    source = inspect.getsource(SynthIDDetector.__init__)
    assert 'device or "cpu"' in source
    assert "cuda_is_available" not in source.replace(".", "_")


@pytest.fixture(scope="module")
def detector():
    return SynthIDDetector(tokenizer_repo=UNGATED_GEMMA_TOKENIZER)


@needs_hub
class TestScoring:
    def test_the_watermarked_sample_is_detected(self, detector):
        result = detector.detect(notebook_texts()["WATERMARKED_TEXT"])
        assert result.is_watermarked
        assert result.score > 0.95
        assert result.z_score > 10
        assert result.reliable

    def test_the_paraphrased_sample_is_not_detected(self, detector):
        result = detector.detect(notebook_texts()["PARAPHRASED_TEXT"])
        assert result.is_watermarked is False
        assert result.score < 0.05

    def test_cuda_does_not_agree_with_cpu(self, detector):
        """Pins the reason the default is cpu: the two devices disagree on the same text.

        If a transformers release ever makes the g-function device-independent this test
        starts failing, which is the signal to revisit the cpu default.
        """
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("no cuda device")
        text = notebook_texts()["WATERMARKED_TEXT"]
        on_cuda = SynthIDDetector(tokenizer_repo=UNGATED_GEMMA_TOKENIZER, device="cuda")
        assert detector.detect(text).is_watermarked
        assert not on_cuda.detect(text).is_watermarked
