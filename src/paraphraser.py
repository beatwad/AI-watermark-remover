"""Paraphrasing of the text with an LLM from any supported provider via LangChain."""

import time
from typing import List

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from src import prompts
from src.config import ParaphraseConfig, Secrets

LANGUAGE_NAMES = {
    "en": "English",
    "ru": "Russian",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "tr": "Turkish",
    "zh-CN": "Chinese",
    "ja": "Japanese",
}

# provider -> (attribute of Secrets holding the key, name of the variable in .env)
PROVIDER_KEYS = {
    "openrouter": ("openrouter_api_key", "OPENROUTER_API_KEY"),
    "openai": ("openai_api_key", "OPENAI_API_KEY"),
    "gemini": ("google_api_key", "GOOGLE_API_KEY"),
    "claude": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    "ollama": ("", ""),  # runs locally, needs no key
}

PROVIDERS = list(PROVIDER_KEYS)

DEFAULT_MODELS = {
    "openrouter": "openai/gpt-4o-mini",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "claude": "claude-sonnet-4-5",
    "ollama": "llama3.1",
}


def api_key_for(provider: str, secrets: Secrets) -> tuple[str, str]:
    """Return the API key configured for the provider and the .env variable it comes from."""
    attribute, env_name = PROVIDER_KEYS[provider]
    if not attribute:
        return "", ""
    return getattr(secrets, attribute), env_name


def missing_key_message(provider: str, secrets: Secrets) -> str:
    """Return an error message if the provider needs an API key that is not set."""
    provider = provider.lower()
    if provider not in PROVIDER_KEYS:
        return f"Unsupported paraphrase provider: {provider}"
    key, env_name = api_key_for(provider, secrets)
    if env_name and not key:
        return f"{env_name} is not set in .env"
    return ""


def _openai_extra(model_name: str) -> dict:
    """Reasoning models only accept a temperature of 1 and take a reasoning effort instead."""
    if any(marker in model_name for marker in ("o1", "o3", "o4", "gpt-5")):
        return {"temperature": 1, "reasoning_effort": "minimal"}
    return {}


def create_chat_model(config: ParaphraseConfig, secrets: Secrets) -> BaseChatModel:
    """Build the LangChain chat model for the configured provider."""
    provider = config.provider.lower()
    error = missing_key_message(provider, secrets)
    if error:
        logger.error("Cannot create the chat model: {}", error)
        raise ValueError(error)

    api_key, _ = api_key_for(provider, secrets)
    proxy = secrets.llm_proxy
    logger.info(
        "Creating chat model: provider={}, model={}, temperature={}, timeout={}s, proxy={}",
        provider,
        config.model,
        config.temperature,
        config.timeout,
        "yes" if proxy else "no",
    )

    if provider in ("openrouter", "openai"):
        from langchain_openai import ChatOpenAI

        base_url = config.base_url if provider == "openrouter" else None
        return ChatOpenAI(
            model_name=config.model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=config.temperature,
            timeout=config.timeout,
            http_client=httpx.Client(proxy=proxy) if proxy else None,
            **_openai_extra(config.model),
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory

        return ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=api_key,
            temperature=config.temperature,
            timeout=config.timeout,
            client_args={"proxy": proxy} if proxy else None,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

    if provider == "claude":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model,
            api_key=api_key,
            temperature=config.temperature,
            timeout=config.timeout,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model,
            base_url=config.ollama_api_url or None,
            temperature=config.temperature,
        )

    logger.error("Unsupported paraphrase provider: {}", config.provider)
    raise ValueError(f"Unsupported paraphrase provider: {config.provider}")


class Paraphraser:
    """Rewrites text with an LLM without changing its meaning."""

    def __init__(self, config: ParaphraseConfig, secrets: Secrets) -> None:
        self.config = config
        self.model = create_chat_model(config, secrets)
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=prompts.custom_instructions),
                ("human", prompts.paraphrase_template),
            ]
        )
        self.chain = prompt | self.model | StrOutputParser()

    def paraphrase(self, text: str, language_code: str = "en") -> str:
        return self.paraphrase_candidates(text, language_code, 1)[0]

    def paraphrase_candidates(
        self, text: str, language_code: str = "en", count: int = 1
    ) -> List[str]:
        """Paraphrase the same text `count` times, in one batch of parallel requests.

        The candidates differ only because sampling makes them differ, so a temperature of 0
        returns the same text `count` times and only costs money.
        """
        count = max(1, count)
        language = LANGUAGE_NAMES.get(language_code, language_code)
        if count > 1 and not self.config.temperature:
            logger.warning(
                "Asking for {} candidates at temperature 0, they will all come back identical",
                count,
            )
        logger.info(
            "Paraphrasing {} characters in {} with '{}', {} candidate(s)",
            len(text),
            language,
            self.config.model,
            count,
        )
        started = time.monotonic()
        payload = {"text": text, "language": language}
        try:
            results = [result.strip() for result in self.chain.batch([payload] * count)]
        except Exception:
            # Rate limits, timeouts, proxy and authentication errors all surface here.
            logger.exception(
                "Paraphrasing with '{}' via {} failed after {:.1f}s",
                self.config.model,
                self.config.provider,
                time.monotonic() - started,
            )
            raise

        empty = sum(1 for result in results if not result)
        if empty:
            logger.warning("'{}' returned {} empty paraphrase(s)", self.config.model, empty)
        logger.info(
            "Paraphrased in {:.1f}s, {} characters returned",
            time.monotonic() - started,
            [len(result) for result in results] if count > 1 else len(results[0]),
        )
        return results
