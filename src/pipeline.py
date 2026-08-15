"""End-to-end pipeline: clean -> round-trip translate -> paraphrase -> clean again."""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from loguru import logger

from src.cleaner import CleaningResult, clean_text
from src.config import AppConfig
from src.paraphraser import Paraphraser
from src.translator import RoundTripTranslator


@dataclass
class PipelineResult:
    original: str
    cleaned: str
    translated: str = ""
    back_translated: str = ""
    paraphrased: str = ""
    final: str = ""
    cleaning: Optional[CleaningResult] = None
    final_cleaning: Optional[CleaningResult] = None
    steps: list[str] = field(default_factory=list)


def run_pipeline(
    text: str,
    config: AppConfig,
    progress: Callable[[str], None] = lambda _: None,
) -> PipelineResult:
    """Run the configured processing steps and return every intermediate result."""
    result = PipelineResult(original=text, cleaned=text, final=text)
    started = time.monotonic()
    logger.info("Pipeline started for {} characters of input", len(text))

    if config.cleaning.enabled:
        progress("Removing LLM symbols")
        cleaning = clean_text(text, config.cleaning.normalize_whitespace, config.cleaning.tiers)
        result.cleaning = cleaning
        result.cleaned = cleaning.text
        result.final = cleaning.text
        result.steps.append("cleaning")

    if config.translation.enabled:
        progress(
            f"Translating to '{config.translation.intermediate_language}' and back "
            f"to '{config.translation.source_language}'"
        )
        try:
            translator = RoundTripTranslator(config.translation, config.secrets.translator_api_key)
            result.translated, result.back_translated = translator.round_trip(result.final)
        except Exception:
            logger.exception("Round-trip translation failed, the pipeline is aborted")
            raise
        result.final = result.back_translated
        result.steps.append("translation")

    if config.paraphrase.enabled:
        progress(f"Paraphrasing with '{config.paraphrase.model}' via {config.paraphrase.provider}")
        try:
            paraphraser = Paraphraser(config.paraphrase, config.secrets)
            result.paraphrased = paraphraser.paraphrase(
                result.final, config.translation.source_language
            )
        except Exception:
            logger.exception("Paraphrasing failed, the pipeline is aborted")
            raise
        result.final = result.paraphrased
        result.steps.append("paraphrase")

    if config.cleaning.enabled:
        # The LLM can reintroduce em dashes and curly quotes, so clean the output once more.
        progress("Cleaning the final text")
        final_cleaning = clean_text(
            result.final, config.cleaning.normalize_whitespace, config.cleaning.tiers
        )
        result.final_cleaning = final_cleaning
        result.final = final_cleaning.text
        if final_cleaning.total:
            logger.info(
                "{} LLM symbols were reintroduced downstream and cleaned again",
                final_cleaning.total,
            )

    if not result.steps:
        logger.warning("Pipeline ran with every step disabled, the text is returned unchanged")
    logger.info(
        "Pipeline finished in {:.1f}s, steps: {}, {} characters returned",
        time.monotonic() - started,
        result.steps or "none",
        len(result.final),
    )
    return result
