"""Round-trip translation: source language -> intermediate language -> source language."""

import time
from typing import List

from deep_translator import DeeplTranslator, GoogleTranslator, MyMemoryTranslator
from loguru import logger

from src.config import TranslationConfig

FREE_PROVIDERS = {"google", "mymemory"}
# The free endpoints intermittently answer with an empty page, a retry usually succeeds.
MAX_ATTEMPTS = 3


class RoundTripTranslator:
    """Translates text to an intermediate language and back to wash out LLM phrasing."""

    def __init__(self, config: TranslationConfig, api_key: str = "") -> None:
        self.config = config
        self.api_key = api_key
        provider = config.provider.lower()
        if provider not in FREE_PROVIDERS and provider != "deepl":
            logger.error("Unsupported translation provider: {}", config.provider)
            raise ValueError(f"Unsupported translation provider: {config.provider}")
        if provider == "deepl" and not api_key:
            logger.error("TRANSLATOR_API_KEY is missing, the deepl provider cannot be used")
            raise ValueError("TRANSLATOR_API_KEY is required for the deepl provider")
        self.provider = provider
        logger.debug("Translator ready: provider={}, chunk_size={}", provider, config.chunk_size)

    def _translate(self, text: str, source: str, target: str) -> str:
        chunks = self._split(text, self.config.chunk_size)
        logger.info(
            "Translating {} characters '{}' -> '{}' in {} chunk(s) via {}",
            len(text),
            source,
            target,
            len(chunks),
            self.provider,
        )
        started = time.monotonic()
        translated = "".join(self._translate_chunk(chunk, source, target) for chunk in chunks)
        logger.info(
            "Translated '{}' -> '{}' in {:.1f}s, {} characters returned",
            source,
            target,
            time.monotonic() - started,
            len(translated),
        )
        return translated

    def _translate_chunk(self, chunk: str, source: str, target: str) -> str:
        if not chunk.strip():
            return chunk
        # Translators drop surrounding whitespace, so restore it to keep the layout.
        leading = chunk[: len(chunk) - len(chunk.lstrip())]
        trailing = chunk[len(chunk.rstrip()) :]

        if self.provider == "google":
            engine = GoogleTranslator(source=source, target=target)
        elif self.provider == "mymemory":
            engine = MyMemoryTranslator(source=source, target=target)
        else:
            engine = DeeplTranslator(api_key=self.api_key, source=source, target=target)

        for attempt in range(MAX_ATTEMPTS):
            try:
                translated = engine.translate(chunk.strip())
                break
            except Exception as error:
                if attempt == MAX_ATTEMPTS - 1:
                    logger.exception(
                        "{} failed to translate a chunk of {} characters after {} attempts",
                        self.provider,
                        len(chunk),
                        MAX_ATTEMPTS,
                    )
                    raise
                logger.warning(
                    "{} translation attempt {}/{} failed ({}: {}), retrying",
                    self.provider,
                    attempt + 1,
                    MAX_ATTEMPTS,
                    type(error).__name__,
                    error,
                )
                time.sleep(1 + attempt)
        if not translated:
            logger.warning(
                "{} returned an empty translation for a chunk of {} characters",
                self.provider,
                len(chunk),
            )
        return f"{leading}{translated or ''}{trailing}"

    @staticmethod
    def _split(text: str, chunk_size: int) -> List[str]:
        """Split text into chunks below the translator limit, preferring paragraph breaks."""
        if len(text) <= chunk_size:
            return [text]

        chunks: List[str] = []
        current = ""
        for paragraph in text.splitlines(keepends=True):
            while len(paragraph) > chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(paragraph[:chunk_size])
                paragraph = paragraph[chunk_size:]
            if len(current) + len(paragraph) > chunk_size:
                chunks.append(current)
                current = paragraph
            else:
                current += paragraph
        if current:
            chunks.append(current)
        return chunks

    def round_trip(self, text: str) -> tuple[str, str]:
        """Return the intermediate translation and the text translated back."""
        source = self.config.source_language
        intermediate = self.config.intermediate_language
        translated = self._translate(text, source, intermediate)
        back_translated = self._translate(translated, intermediate, source)
        return translated, back_translated
