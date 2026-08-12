# AI Watermark Remover

An application that uses a multistep pipeline to remove any watermarks from AI generated text

What it does:
- takes text, strips the typographic markers LLMs usually leave behind (em dashes, zero-width characters, exotic spaces, curly quotes)
- runs it through a translation round trip (e.g. English -> intermediate language -> English), can select different translation provide (Google, Mymemory, Deepl)
- paraphrases the result with an LLM, keeping the meaning, quality and readability intact.

Paraphrasing runs through OpenRouter, OpenAI, Gemini, Claude or a local Ollama model.

Any step is optional, so you can configure watermark removing pipeline as you like.

## Setup

```bash
uv sync
cp .env_example .env # then fill in the API key of the provider you use
cp config_example.yaml config.yaml # optional, the example is used until you do
```

## Run

```bash
uv run streamlit run app.py
```

Paste the text into `Input text` field, then press **Ctrl + Enter** to confirm it, then press **Process**.

## Configuration

- `config.yaml` - which steps to run, translation provider, intermediate language, paraphrase
 provider and model. It is gitignored so your settings stay yours; `config_example.yaml` is the
 version in the repository and is read as a fallback while `config.yaml` does not exist.
- `.env` - API keys (see `.env_example`). Only the key of the selected paraphrase provider is
 needed, `ollama` needs none. The `google` and `mymemory` translation providers are free
 and need no key; `deepl` needs `TRANSLATOR_API_KEY`.
- `src/prompts.py` - all prompts.

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