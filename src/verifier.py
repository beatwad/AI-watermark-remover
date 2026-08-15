"""Optional verification: score every stage of the pipeline with the SynthID detector.

This answers the question the rest of the pipeline only assumes: did the watermark actually go
away, and at which step. It is diagnostic rather than part of the work, so unlike translation
and paraphrasing it never aborts the pipeline. A detector that will not load is reported as an
error string and the processed text is still returned.

`torch` and `transformers` are an optional extra, so the detector is imported lazily. Importing
this module must stay cheap and must not require the extra.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

from loguru import logger

INSTALL_HINT = "uv sync --extra detector"


@dataclass
class StageScore:
    """What the detector says about the text at one stage of the pipeline."""

    stage: str
    score: float
    z_score: float
    verdict: str
    token_count: int
    reliable: bool


@dataclass
class VerificationResult:
    stages: List[StageScore] = field(default_factory=list)
    error: str = ""

    @property
    def available(self) -> bool:
        return bool(self.stages)


@lru_cache(maxsize=2)
def _load_detector(detector_repo: str, tokenizer_repo: str, device: str, hf_token: str):
    """Load the detector once per configuration; it is slow to build and stateless to use.

    Only successful loads are cached, so a failure is retried on the next run.
    """
    # Imported here so that the app runs without the detector extra installed.
    from watermark_detector.detector import SynthIDDetector

    return SynthIDDetector(
        detector_repo=detector_repo,
        tokenizer_repo=tokenizer_repo or None,
        device=device or None,
        hf_token=hf_token or None,
    )


def verify(
    stages: Dict[str, str],
    detector_repo: str,
    tokenizer_repo: str = "",
    device: str = "",
    hf_token: str = "",
) -> VerificationResult:
    """Score each stage in one pass. Never raises, failures come back as `error`."""
    texts = {name: text for name, text in stages.items() if text.strip()}
    if not texts:
        return VerificationResult(error="There is no text to score.")

    try:
        detector = _load_detector(detector_repo, tokenizer_repo, device, hf_token)
    except ImportError:
        logger.warning("Verification is enabled but the detector extra is not installed")
        return VerificationResult(
            error=f"The detector needs torch and transformers, install them with `{INSTALL_HINT}`."
        )
    except Exception as error:
        logger.exception("Loading the SynthID detector {} failed", detector_repo)
        return VerificationResult(
            error=f"Could not load the detector {detector_repo}: {type(error).__name__}: {error}"
        )

    try:
        results = detector.detect_batch(list(texts.values()))
    except Exception as error:
        logger.exception("Scoring the pipeline stages failed")
        return VerificationResult(
            error=f"The detector could not score the text: {type(error).__name__}: {error}"
        )

    scored = [
        StageScore(
            stage=name,
            score=result.score,
            z_score=result.z_score,
            verdict=result.verdict,
            token_count=result.token_count,
            reliable=result.reliable,
        )
        for name, result in zip(texts, results)
    ]
    logger.info(
        "Verified {} stages: {}",
        len(scored),
        ", ".join(f"{stage.stage} {stage.score:.4f}" for stage in scored),
    )
    return VerificationResult(stages=scored)


@dataclass
class Selection:
    """Which of several paraphrase candidates was kept, and how they all scored."""

    text: str
    index: int = 0
    scores: List[StageScore] = field(default_factory=list)
    error: str = ""


def select_candidate(
    candidates: List[str],
    detector_repo: str,
    tokenizer_repo: str = "",
    device: str = "",
    hf_token: str = "",
) -> Selection:
    """Keep the candidate the detector likes least.

    Ranking is by posterior first and z-score second. The posterior saturates: once a watermark
    is gone it reads 0.0000 for every candidate, and on a tie there is nothing left to choose on.
    The z-score keeps resolving below that floor, so it is the tie-break rather than a second
    opinion here.
    """
    usable = [text for text in candidates if text.strip()]
    if not usable:
        return Selection(text=candidates[0] if candidates else "", error="Every candidate is empty.")
    if len(usable) == 1:
        return Selection(text=usable[0])

    result = verify(
        {f"candidate {number}": text for number, text in enumerate(usable, 1)},
        detector_repo=detector_repo,
        tokenizer_repo=tokenizer_repo,
        device=device,
        hf_token=hf_token,
    )
    if result.error:
        # Without scores there is nothing to choose on, so the first candidate is as good as any.
        logger.warning("Candidates could not be scored, keeping the first one: {}", result.error)
        return Selection(text=usable[0], error=result.error)

    best = min(
        range(len(result.stages)),
        key=lambda index: (result.stages[index].score, result.stages[index].z_score),
    )
    logger.info(
        "Kept candidate {} of {} at score {:.4f}, z={:+.2f}",
        best + 1,
        len(usable),
        result.stages[best].score,
        result.stages[best].z_score,
    )
    return Selection(text=usable[best], index=best, scores=result.stages)


def collect_stages(*labelled: tuple[str, Optional[str]]) -> Dict[str, str]:
    """Ordered stage name -> text, dropping empty stages and ones that changed nothing.

    A step that was disabled leaves its stage identical to the previous one, and scoring the
    same text twice only costs time and clutters the table.
    """
    stages: Dict[str, str] = {}
    for name, text in labelled:
        if text and text not in stages.values():
            stages[name] = text
    return stages
