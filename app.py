"""Streamlit GUI for the LLM watermark remover."""

import streamlit as st

from src.cleaner import CleaningResult
from src.config import CONFIG_PATH, load_config, save_config
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

def step_label(title: str, key: str, enabled_by_default: bool) -> str:
    """Expander title showing whether the step is on, taken from the previous rerun."""
    return f"{title}" if st.session_state.get(key, enabled_by_default) else f"{title} · off"


with st.sidebar:
    st.caption("Defaults come from config.yaml, secrets from .env")

    with st.expander(step_label("Cleaning", "clean_on", config.cleaning.enabled)):
        config.cleaning.enabled = st.checkbox(
            "Remove LLM symbols", config.cleaning.enabled, key="clean_on"
        )
        config.cleaning.normalize_whitespace = st.checkbox(
            "Normalize whitespace", config.cleaning.normalize_whitespace
        )

    with st.expander(step_label("Translation", "translate_on", config.translation.enabled)):
        config.translation.enabled = st.checkbox(
            "Round-trip translation", config.translation.enabled, key="translate_on"
        )
        providers = ["google", "mymemory", "deepl"]
        config.translation.provider = st.selectbox(
            "Provider",
            providers,
            index=providers.index(config.translation.provider)
            if config.translation.provider in providers
            else 0,
            format_func=lambda code: (
                "deepl (API key required)" if code == "deepl" else code
            ),
        )
        all_languages = sorted(
            set(LANGUAGE_NAMES)
            | {config.translation.source_language, config.translation.intermediate_language}
        )
        source_column, intermediate_column = st.columns(2)
        with source_column:
            config.translation.source_language = st.selectbox(
                "Source",
                all_languages,
                index=all_languages.index(config.translation.source_language),
                format_func=language_label,
                help="Language of the input text. The text is translated back into it.",
            )
        languages = [code for code in all_languages if code != config.translation.source_language]
        with intermediate_column:
            config.translation.intermediate_language = st.selectbox(
                "Intermediate",
                languages,
                index=languages.index(config.translation.intermediate_language)
                if config.translation.intermediate_language in languages
                else 0,
                format_func=language_label,
                help="Language the text is translated through and back.",
            )

    with st.expander(step_label("Paraphrase", "paraphrase_on", config.paraphrase.enabled)):
        config.paraphrase.enabled = st.checkbox(
            "Paraphrase", config.paraphrase.enabled, key="paraphrase_on"
        )
        configured_provider = config.paraphrase.provider.lower()
        provider_column, temperature_column = st.columns(2)
        with provider_column:
            config.paraphrase.provider = st.selectbox(
                "Provider",
                PROVIDERS,
                index=PROVIDERS.index(configured_provider)
                if configured_provider in PROVIDERS
                else 0,
            )
        with temperature_column:
            config.paraphrase.temperature = st.number_input(
                "Temperature", 0.0, 1.5, float(config.paraphrase.temperature), 0.1
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

    # Outside the expander, a collapsed one would hide the warning.
    if config.paraphrase.enabled:
        key_error = missing_key_message(config.paraphrase.provider, config.secrets)
        if key_error:
            st.error(key_error)

    if st.button("Save settings", width="stretch", help=f"Write them to {CONFIG_PATH.name}"):
        try:
            save_config(config)
        except OSError as error:
            st.error(f"Could not write {CONFIG_PATH.name}: {error}")
        else:
            st.success(f"Saved to {CONFIG_PATH.name}")

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
