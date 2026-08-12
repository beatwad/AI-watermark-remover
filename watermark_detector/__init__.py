"""SynthID text watermark detection."""

from watermark_detector.detector import (
    DEFAULT_DETECTOR_REPO,
    GOOGLE_DEMO_DETECTOR_REPO,
    NOT_WATERMARKED,
    UNCERTAIN,
    UNGATED_GEMMA_TOKENIZER,
    WATERMARKED,
    DetectionResult,
    SynthIDDetector,
)

__all__ = [
    "DEFAULT_DETECTOR_REPO",
    "GOOGLE_DEMO_DETECTOR_REPO",
    "UNGATED_GEMMA_TOKENIZER",
    "NOT_WATERMARKED",
    "UNCERTAIN",
    "WATERMARKED",
    "DetectionResult",
    "SynthIDDetector",
]
