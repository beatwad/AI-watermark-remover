"""Configuration loading: config.yaml for settings, .env for secrets."""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from loguru import logger
from ruamel.yaml import YAML

from src.cleaner import DEFAULT_TIERS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
# config.yaml is gitignored, so a fresh clone starts from the example.
CONFIG_EXAMPLE_PATH = PROJECT_ROOT / "config_example.yaml"

if not (PROJECT_ROOT / ".env").exists():
    logger.warning("No .env file at {}, secrets are taken from the environment", PROJECT_ROOT)
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class CleaningConfig:
    enabled: bool = True
    normalize_whitespace: bool = True
    tiers: list[str] = field(default_factory=lambda: list(DEFAULT_TIERS))


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
class VerificationConfig:
    enabled: bool = False
    # Defaults repeated from watermark_detector.detector, which cannot be imported here:
    # it pulls in torch, and that is an optional extra.
    detector_repo: str = "joaogante/dummy_synthid_detector"
    tokenizer_repo: str = "unsloth/gemma-2b-it"
    # Empty means cpu, which is the only device that scores correctly. See detector.py.
    device: str = ""


@dataclass
class Secrets:
    openrouter_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    anthropic_api_key: str = ""
    translator_api_key: str = ""
    llm_proxy: str = ""
    hf_token: str = ""


@dataclass
class AppConfig:
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    paraphrase: ParaphraseConfig = field(default_factory=ParaphraseConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    secrets: Secrets = field(default_factory=Secrets)


def _section(raw: Dict[str, Any], name: str) -> Dict[str, Any]:
    return raw.get(name) or {}


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    """Read config.yaml and the .env secrets into a single config object."""
    if not path.exists() and CONFIG_EXAMPLE_PATH.exists():
        logger.warning("{} not found, falling back to {}", path, CONFIG_EXAMPLE_PATH.name)
        path = CONFIG_EXAMPLE_PATH

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        logger.exception("Could not read the config from {}", path)
        raise

    logger.info("Loaded config from {} with sections {}", path, sorted(raw))

    try:
        config = AppConfig(
            cleaning=CleaningConfig(**_section(raw, "cleaning")),
            translation=TranslationConfig(**_section(raw, "translation")),
            paraphrase=ParaphraseConfig(**_section(raw, "paraphrase")),
            verification=VerificationConfig(**_section(raw, "verification")),
            secrets=Secrets(
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
                openai_api_key=os.getenv("OPENAI_API_KEY", ""),
                google_api_key=os.getenv("GOOGLE_API_KEY", ""),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
                translator_api_key=os.getenv("TRANSLATOR_API_KEY", ""),
                llm_proxy=os.getenv("LLM_PROXY", ""),
                hf_token=os.getenv("HF_TOKEN", ""),
            ),
        )
    except TypeError:
        # An unknown or misspelled key in config.yaml.
        logger.exception("Config in {} does not match the expected settings", path)
        raise

    # Never log the values themselves.
    logger.debug(
        "Secrets present: {}",
        [name for name, value in asdict(config.secrets).items() if value] or "none",
    )
    logger.debug(
        "Steps enabled: cleaning={}, translation={}, paraphrase={}",
        config.cleaning.enabled,
        config.translation.enabled,
        config.paraphrase.enabled,
    )
    return config


def save_config(config: AppConfig, path: Path = CONFIG_PATH) -> None:
    """Write the settings back to config.yaml, keeping its comments and layout.

    Secrets are never written, they stay in .env.
    """
    sections = {
        "cleaning": asdict(config.cleaning),
        "translation": asdict(config.translation),
        "paraphrase": asdict(config.paraphrase),
        "verification": asdict(config.verification),
    }

    round_trip = YAML()
    round_trip.preserve_quotes = True
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            document = round_trip.load(f) or {}
    else:
        logger.info("{} does not exist yet, creating it", path)
        document = {}

    for name, values in sections.items():
        section = document.setdefault(name, {})
        for key, value in values.items():
            section[key] = value

    try:
        with open(path, "w", encoding="utf-8") as f:
            round_trip.dump(document, f)
    except OSError:
        logger.exception("Could not write the config to {}", path)
        raise

    logger.info("Saved config to {}", path)
