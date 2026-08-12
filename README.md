# LLM Watermark Remover

Takes text, strips the typographic markers LLMs usually leave behind (em dashes, zero-width
characters, exotic spaces, curly quotes), runs it through a translation round trip
(English -> intermediate language -> English) and paraphrases the result with an LLM,
keeping the meaning, quality and readability intact.

Paraphrasing runs through OpenRouter, OpenAI, Gemini, Claude or a local Ollama model.

## Setup

```bash
uv sync
cp .env_example .env   # then fill in the API key of the provider you use
```

## Run

```bash
uv run streamlit run app.py
```

## Configuration

- `config.yaml` — which steps to run, translation provider, intermediate language, paraphrase
  provider and model.
- `.env` — API keys (see `.env_example`). Only the key of the selected paraphrase provider is
  needed, `ollama` needs none. The `google` and `mymemory` translation providers are free
  and need no key; `deepl` needs `TRANSLATOR_API_KEY`.
- `src/prompts.py` — all prompts.

Every setting in `config.yaml` can also be overridden per run in the sidebar of the GUI. Those
overrides live only in the browser session, until you press **Save settings**, which writes them
back to `config.yaml` (comments included). Secrets are never written there, they stay in `.env`.

## Layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit GUI |
| `src/pipeline.py` | Orchestrates clean -> translate -> paraphrase -> clean |
| `src/cleaner.py` | Symbol map, replacement and per-symbol statistics |
| `src/translator.py` | Round-trip translation with chunking |
| `src/paraphraser.py` | LangChain chain over the selected provider's model |
| `src/prompts.py` | Prompts |
| `src/config.py` | `config.yaml` + `.env` loading |
