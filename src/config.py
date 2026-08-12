"""Configuration loading: config.yaml for settings, .env for secrets."""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from ruamel.yaml import YAML

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class CleaningConfig:
    enabled: bool = True
    normalize_whitespace: bool = True


@dataclass
class TranslationConfig:
    enabled: bool = True
    provider: str = "google"
    intermediate_language: str = "ru"
    source_language: str = "en"
    chunk_size: int = 4000


@dataclass
class ParaphraseConfig:
    enabled: bool = True
    provider: str = "openrouter"
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.7
    timeout: int = 120
    base_url: str = "https://openrouter.ai/api/v1"
    ollama_api_url: str = "http://localhost:11434"


@dataclass
class Secrets:
    openrouter_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    anthropic_api_key: str = ""
    translator_api_key: str = ""
    llm_proxy: str = ""


@dataclass
class AppConfig:
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    paraphrase: ParaphraseConfig = field(default_factory=ParaphraseConfig)
    secrets: Secrets = field(default_factory=Secrets)


def _section(raw: Dict[str, Any], name: str) -> Dict[str, Any]:
    return raw.get(name) or {}


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    """Read config.yaml and the .env secrets into a single config object."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return AppConfig(
        cleaning=CleaningConfig(**_section(raw, "cleaning")),
        translation=TranslationConfig(**_section(raw, "translation")),
        paraphrase=ParaphraseConfig(**_section(raw, "paraphrase")),
        secrets=Secrets(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            translator_api_key=os.getenv("TRANSLATOR_API_KEY", ""),
            llm_proxy=os.getenv("LLM_PROXY", ""),
        ),
    )


def save_config(config: AppConfig, path: Path = CONFIG_PATH) -> None:
    """Write the settings back to config.yaml, keeping its comments and layout.

    Secrets are never written, they stay in .env.
    """
    sections = {
        "cleaning": asdict(config.cleaning),
        "translation": asdict(config.translation),
        "paraphrase": asdict(config.paraphrase),
    }

    round_trip = YAML()
    round_trip.preserve_quotes = True
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            document = round_trip.load(f) or {}
    else:
        document = {}

    for name, values in sections.items():
        section = document.setdefault(name, {})
        for key, value in values.items():
            section[key] = value

    with open(path, "w", encoding="utf-8") as f:
        round_trip.dump(document, f)
