"""Streamlit GUI for the LLM watermark remover."""

import streamlit as st

from src.cleaner import CleaningResult
from src.config import load_config
from src.paraphraser import DEFAULT_MODELS, LANGUAGE_NAMES, PROVIDERS, missing_key_message
from src.pipeline import run_pipeline

st.set_page_config(page_title="LLM Watermark Remover", page_icon="🧽", layout="wide")


def language_label(code: str) -> str:
    return f"{LANGUAGE_NAMES.get(code, code)} ({code})"


def render_stats(title: str, cleaning: CleaningResult) -> None:
    st.subheader(title)
    if not cleaning.total:
        st.info("No LLM-typical symbols found.")
        return
    st.metric("Symbols removed or replaced", cleaning.total)
    st.dataframe(
        [
            {
                "Symbol": stat.symbol,
                "Code point": stat.codepoint,
                "Name": stat.name,
                "Action": stat.action,
                "Count": stat.count,
            }
            for stat in cleaning.stats
        ],
        hide_index=True,
        width="stretch",
    )


config = load_config()

st.title("LLM Watermark Remover")
st.caption("Strip LLM typography, round-trip translate and paraphrase the text.")

with st.sidebar:
    st.header("Settings")
    st.caption("Defaults come from config.yaml, secrets from .env")

    config.cleaning.enabled = st.checkbox("Remove LLM symbols", config.cleaning.enabled)
    config.cleaning.normalize_whitespace = st.checkbox(
        "Normalize whitespace", config.cleaning.normalize_whitespace
    )

    st.divider()
    config.translation.enabled = st.checkbox("Round-trip translation", config.translation.enabled)
    providers = ["google", "mymemory", "deepl"]
    config.translation.provider = st.selectbox(
        "Translation provider",
        providers,
        index=providers.index(config.translation.provider)
        if config.translation.provider in providers
        else 0,
    )
    languages = sorted(set(LANGUAGE_NAMES) | {config.translation.intermediate_language})
    languages = [code for code in languages if code != config.translation.source_language]
    config.translation.intermediate_language = st.selectbox(
        "Intermediate language",
        languages,
        index=languages.index(config.translation.intermediate_language)
        if config.translation.intermediate_language in languages
        else 0,
        format_func=language_label,
    )

    st.divider()
    config.paraphrase.enabled = st.checkbox("Paraphrase", config.paraphrase.enabled)
    configured_provider = config.paraphrase.provider.lower()
    config.paraphrase.provider = st.selectbox(
        "LLM provider",
        PROVIDERS,
        index=PROVIDERS.index(configured_provider) if configured_provider in PROVIDERS else 0,
    )
    # Keying the field by provider resets the model when another provider is picked.
    default_model = (
        config.paraphrase.model
        if config.paraphrase.provider == configured_provider
        else DEFAULT_MODELS[config.paraphrase.provider]
    )
    config.paraphrase.model = st.text_input(
        "Model", default_model, key=f"model_{config.paraphrase.provider}"
    )
    config.paraphrase.temperature = st.slider(
        "Temperature", 0.0, 1.5, float(config.paraphrase.temperature), 0.1
    )

    if config.paraphrase.enabled:
        key_error = missing_key_message(config.paraphrase.provider, config.secrets)
        if key_error:
            st.error(key_error)

text = st.text_area("Input text", height=280, placeholder="Paste the text here...")

if st.button("Process", type="primary", disabled=not text.strip()):
    status = st.status("Processing...", expanded=True)
    try:
        result = run_pipeline(text, config, progress=status.write)
    except Exception as error:  # surface the failure in the UI instead of a blank page
        status.update(label="Failed", state="error")
        st.error(f"{type(error).__name__}: {error}")
    else:
        status.update(label=f"Done: {', '.join(result.steps)}", state="complete", expanded=False)

        st.subheader("Result")
        st.text_area("Output text", result.final, height=280)

        if result.cleaning:
            render_stats("Symbol statistics", result.cleaning)
        if result.final_cleaning and result.final_cleaning.total:
            render_stats("Symbols reintroduced downstream and cleaned again", result.final_cleaning)

        with st.expander("Intermediate results"):
            if result.cleaning:
                st.text_area("After symbol cleaning", result.cleaned, height=200)
            if result.translated:
                st.text_area(
                    f"Translated to {language_label(config.translation.intermediate_language)}",
                    result.translated,
                    height=200,
                )
                st.text_area("Translated back", result.back_translated, height=200)
            if result.paraphrased:
                st.text_area("Paraphrased", result.paraphrased, height=200)
