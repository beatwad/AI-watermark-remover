"""End-to-end pipeline: clean -> round-trip translate -> paraphrase -> clean again."""

from dataclasses import dataclass, field
from typing import Callable, Optional

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

    if config.cleaning.enabled:
        progress("Removing LLM symbols")
        cleaning = clean_text(text, config.cleaning.normalize_whitespace)
        result.cleaning = cleaning
        result.cleaned = cleaning.text
        result.final = cleaning.text
        result.steps.append("cleaning")

    if config.translation.enabled:
        progress(
            f"Translating to '{config.translation.intermediate_language}' and back "
            f"to '{config.translation.source_language}'"
        )
        translator = RoundTripTranslator(config.translation, config.secrets.translator_api_key)
        result.translated, result.back_translated = translator.round_trip(result.final)
        result.final = result.back_translated
        result.steps.append("translation")

    if config.paraphrase.enabled:
        progress(f"Paraphrasing with '{config.paraphrase.model}' via {config.paraphrase.provider}")
        paraphraser = Paraphraser(config.paraphrase, config.secrets)
        result.paraphrased = paraphraser.paraphrase(
            result.final, config.translation.source_language
        )
        result.final = result.paraphrased
        result.steps.append("paraphrase")

    if config.cleaning.enabled:
        # The LLM can reintroduce em dashes and curly quotes, so clean the output once more.
        progress("Cleaning the final text")
        final_cleaning = clean_text(result.final, config.cleaning.normalize_whitespace)
        result.final_cleaning = final_cleaning
        result.final = final_cleaning.text

    return result
