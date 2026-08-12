# SynthID watermark detector

Detects whether a text carries a [SynthID](https://deepmind.google/models/synthid/) text
watermark, using the detection machinery that ships with Hugging Face `transformers`.

## Read this before using it

SynthID text watermarking is **keyed**. During generation a logits processor biases token
sampling with a secret set of key integers, and a detector only recognises the watermark
produced by the very same keys. This has one consequence that decides what this module can
and cannot do for you:

**Text from the Gemini app cannot be detected here.** Google watermarks Gemini output with
production keys it does not publish, and its own SynthID Detector portal is restricted to
journalists, media organisations and researchers. Every publicly downloadable detector,
including the default below, is trained on a *demo* key set. Pointing it at Gemini output
gives you a meaningless score, not a negative result.

What this module is genuinely good for is detecting watermarks you or a partner applied
yourself with `SynthIDTextWatermarkLogitsProcessor`, and for checking whether the removal
pipeline in this repository actually strips a watermark you control end to end.

## Install

```bash
uv sync --extra detector
```

`torch` and `transformers` are an optional extra, so the Streamlit app installs without them.

`transformers` is capped below 5.x on purpose. In 5.x `BayesianDetectorModel.__init__` never
calls `post_init()`, so `from_pretrained` dies with
`AttributeError: 'BayesianDetectorModel' object has no attribute 'all_tied_weights_keys'`.
This is verified working on 4.57.6.

## Hub access

The default detector `joaogante/dummy_synthid_detector` is public, but it was trained
against `google/gemma-2b-it` and must use that model's tokenizer, which is gated. Either
accept the licence at <https://huggingface.co/google/gemma-2b-it> and authenticate:

```bash
huggingface-cli login   # or pass hf_token= / set HF_TOKEN
```

or skip the gate with an ungated mirror of the same tokenizer:

```python
from watermark_detector import UNGATED_GEMMA_TOKENIZER, SynthIDDetector

detector = SynthIDDetector(tokenizer_repo=UNGATED_GEMMA_TOKENIZER)
```

`tokenizer_repo` exists only for this. The g-values are computed over token ids, so a
tokenizer with a different vocabulary silently turns every score into noise.

## Use

```python
from watermark_detector import SynthIDDetector

detector = SynthIDDetector()
result = detector.detect(text)

result.score          # posterior probability that the text is watermarked
result.verdict        # "watermarked" | "not watermarked" | "uncertain"
result.is_watermarked # True | False | None when uncertain
result.token_count    # tokens scored
result.reliable       # False when the text is too short to conclude anything
```

`detect_batch(texts)` scores several texts in one pass.

To use a detector of your own, or one you have been given access to, pass its repo id or a
local path:

```python
detector = SynthIDDetector(detector_repo="google/synthid-spaces-demo-detector")
```

The detector carries the key set and the model it was trained on in its config, so the
tokenizer and the g-function are always taken from it and never configured separately.

## Calibration and limits

- **Thresholds.** `watermarked_threshold=0.95` and `not_watermarked_threshold=0.60` are a
  starting point, not a calibrated setting. Score your own watermarked and unwatermarked
  samples and move them until the false positive rate is where you want it.
- **Length.** Detection is statistical. Under roughly 200 tokens the posterior stays near
  the prior; `result.reliable` is `False` there and the module logs a warning.
- **Fragility.** The watermark degrades under heavy paraphrasing and translation, which is
  exactly what this repository's pipeline does. A "not watermarked" verdict on processed
  text is evidence the pipeline worked, not evidence the text was human written.
- **Not an AI-text detector.** A negative verdict means "not watermarked with these keys".
  It says nothing about whether a machine wrote the text.

## Verified behaviour

Text generated with `SynthIDTextWatermarkLogitsProcessor` under the demo key set scores
`1.0000`, the same prompts generated without it score `0.0000`. Ordinary unwatermarked text
scores between `0.0000` and `0.0383`. So the machinery separates cleanly on the key set the
detector was trained for, which is the demo key set and not Gemini's.

## References

- [SynthID Text on Hugging Face](https://huggingface.co/blog/synthid-text)
- [Google's SynthID text guide](https://ai.google.dev/responsible/docs/safeguards/synthid)
- [google-deepmind/synthid-text](https://github.com/google-deepmind/synthid-text)
