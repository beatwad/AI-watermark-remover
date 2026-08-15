"""Detection of SynthID text watermarks.

SynthID is keyed. A detector only recognises watermarks that were produced with the
exact same key set, and the keys Google uses for Gemini in production are not public,
so no detector you can download will flag real Gemini output. What this module runs is
the genuine SynthID detection machinery from `transformers`, pointed at whichever
Bayesian detector you have access to. The default is a public demo detector trained on
the demo key set published with the SynthID example code, so it only recognises text
generated with those same demo keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from loguru import logger
from transformers import (
    AutoTokenizer,
    BayesianDetectorModel,
    SynthIDTextWatermarkLogitsProcessor,
)

# Public detector trained on the demo key set, uses the google/gemma-2b-it tokenizer.
DEFAULT_DETECTOR_REPO = "joaogante/dummy_synthid_detector"

# Google's own demo detector, same demo key set, gated behind a licence on the Hub.
GOOGLE_DEMO_DETECTOR_REPO = "google/synthid-spaces-demo-detector"

# The default detector names google/gemma-2b-it as its tokenizer, which is gated. This is a
# mirror of the same tokenizer that needs no licence, for use as tokenizer_repo.
UNGATED_GEMMA_TOKENIZER = "unsloth/gemma-2b-it"

# Under this many tokens the posterior stays close to the prior and says very little.
MIN_RELIABLE_TOKENS = 200

WATERMARKED = "watermarked"
NOT_WATERMARKED = "not watermarked"
UNCERTAIN = "uncertain"


@dataclass
class DetectionResult:
    """The verdict for one text."""

    score: float
    verdict: str
    token_count: int
    z_score: float = 0.0
    mean_g_value: float = 0.5

    @property
    def is_watermarked(self) -> Optional[bool]:
        """True, False, or None when the score falls between the two thresholds."""
        return {WATERMARKED: True, NOT_WATERMARKED: False}.get(self.verdict)

    @property
    def reliable(self) -> bool:
        """Whether the text was long enough for the score to mean anything."""
        return self.token_count >= MIN_RELIABLE_TOKENS


class SynthIDDetector:
    """Scores text with a trained SynthID Bayesian detector.

    The score is the posterior probability that the text carries the watermark the
    detector was trained on. It is turned into one of three verdicts by two thresholds,
    which you should calibrate against your own watermarked and unwatermarked samples
    to hit the false positive rate you want; the defaults are only a starting point.
    """

    def __init__(
        self,
        detector_repo: str = DEFAULT_DETECTOR_REPO,
        tokenizer_repo: Optional[str] = None,
        device: Optional[str] = None,  # see the note in __init__ before setting this to cuda
        hf_token: Optional[str] = None,
        watermarked_threshold: float = 0.95,
        not_watermarked_threshold: float = 0.60,
    ):
        if not not_watermarked_threshold <= watermarked_threshold:
            raise ValueError(
                f"not_watermarked_threshold ({not_watermarked_threshold}) must not be above "
                f"watermarked_threshold ({watermarked_threshold})"
            )
        self.watermarked_threshold = watermarked_threshold
        self.not_watermarked_threshold = not_watermarked_threshold
        # cpu on purpose, and not "cuda if available". The g-function is a sampling table built
        # by torch.randint from a seeded generator, and torch RNG is not reproducible across
        # devices, so the table and every g-value that comes out of it differ on cuda. The
        # detector's weights were trained against the cpu table, which makes a cuda score noise
        # rather than a wrong number you could correct for. Measured on the notebook's
        # watermarked sample: cpu 1.0000 with mean g 0.5599, cuda 0.0000 with mean g 0.4963.
        self.device = device or "cpu"
        if self.device.startswith("cuda"):
            logger.warning(
                "Running the detector on {} scores against a g-function this detector was not "
                "trained on, so the result is noise. Use cpu unless you also generated the "
                "watermark on this device.",
                self.device,
            )

        logger.info("Loading the SynthID detector {} on {}", detector_repo, self.device)
        try:
            detector_module = BayesianDetectorModel.from_pretrained(
                detector_repo, token=hf_token
            ).to(self.device)
            # The detector carries the key set it was trained on and the model it was
            # trained against, so the tokenizer and the g-function must come from it.
            # tokenizer_repo only exists to point at an ungated mirror of that tokenizer;
            # a tokenizer with a different vocabulary makes every score meaningless.
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_repo or detector_module.config.model_name, token=hf_token
            )
        except OSError:
            logger.exception(
                "Could not load {} or its tokenizer. Either may be gated on the Hub, in which "
                "case accept the licence and pass hf_token, or pass tokenizer_repo={!r}",
                detector_repo,
                UNGATED_GEMMA_TOKENIZER,
            )
            raise

        detector_module.eval()
        # Padding must be with the EOS token on the right: the detector stops scoring at
        # the first EOS, so padding on the left would mask every text out entirely.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.logits_processor = SynthIDTextWatermarkLogitsProcessor(
            **detector_module.config.watermarking_config, device=self.device
        )
        self.detector_module = detector_module
        self.ngram_len = self.logits_processor.ngram_len
        logger.info(
            "Detector ready: {} keys, ngram_len {}",
            len(detector_module.config.watermarking_config["keys"]),
            self.ngram_len,
        )

    def detect(self, text: str) -> DetectionResult:
        """Score a single text."""
        return self.detect_batch([text])[0]

    def detect_batch(self, texts: List[str]) -> List[DetectionResult]:
        """Score several texts in one pass."""
        # Special tokens are not part of what the model generated, so they are left out.
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        )
        input_ids = encoded.input_ids.to(self.device)
        token_counts = encoded.attention_mask.sum(dim=1).tolist()

        if input_ids.shape[1] < self.ngram_len:
            logger.warning(
                "The longest text is {} tokens, below the {}-token n-gram the detector scores "
                "over, so there is nothing to score",
                input_ids.shape[1],
                self.ngram_len,
            )
            return [
                DetectionResult(score=0.5, verdict=UNCERTAIN, token_count=count)
                for count in token_counts
            ]

        with torch.no_grad():
            g_values, mask = self._g_values_and_mask(input_ids)
            scores = self.detector_module(g_values, mask)[0]
        mean_g_values, z_scores = self._mean_and_z(g_values, mask)

        results = [
            DetectionResult(
                score=float(score),
                verdict=self._verdict(float(score)),
                token_count=count,
                z_score=float(z),
                mean_g_value=float(mean_g),
            )
            for score, count, z, mean_g in zip(scores, token_counts, z_scores, mean_g_values)
        ]
        for result in results:
            if not result.reliable:
                logger.warning(
                    "Only {} tokens, under the {} that make a score meaningful, so '{}' at "
                    "{:.3f} is weak evidence",
                    result.token_count,
                    MIN_RELIABLE_TOKENS,
                    result.verdict,
                    result.score,
                )
            else:
                logger.info(
                    "Scored {} tokens: {} at {:.3f}, z={:.2f}",
                    result.token_count,
                    result.verdict,
                    result.score,
                    result.z_score,
                )
        return results

    def _g_values_and_mask(self, input_ids):
        """Compute the g-values and the mask saying which of them count.

        This is what SynthIDTextWatermarkDetector.__call__ does internally before handing
        them to the Bayesian model. It is repeated here because that wrapper returns only
        the posterior and throws the g-values away, and the z-score is computed from them.
        """
        eos_token_mask = self.logits_processor.compute_eos_token_mask(
            input_ids=input_ids, eos_token_id=self.tokenizer.eos_token_id
        )[:, self.ngram_len - 1 :]
        context_repetition_mask = self.logits_processor.compute_context_repetition_mask(
            input_ids=input_ids
        )
        g_values = self.logits_processor.compute_g_values(input_ids=input_ids)
        return g_values, context_repetition_mask * eos_token_mask

    @staticmethod
    def _mean_and_z(g_values, mask):
        """Mean g-value per text and its z-score against the unwatermarked null.

        Without a watermark each g-value is a fair coin flip, so the mean of n of them has
        mean 0.5 and standard deviation 0.5/sqrt(n), which gives z = 2*sqrt(n)*(mean - 0.5).
        Watermarking pushes g-values towards 1, so a watermark shows up as a large positive z.
        """
        counted = mask.unsqueeze(-1).expand_as(g_values)
        n = counted.sum(dim=(1, 2))
        safe_n = n.clamp(min=1)
        mean_g = (g_values * counted).sum(dim=(1, 2)) / safe_n
        z = 2.0 * safe_n.float().sqrt() * (mean_g - 0.5)
        # A text with nothing left to score is not evidence either way.
        return torch.where(n > 0, mean_g, torch.full_like(mean_g, 0.5)), torch.where(
            n > 0, z, torch.zeros_like(z)
        )

    def _verdict(self, score: float) -> str:
        if score >= self.watermarked_threshold:
            return WATERMARKED
        if score <= self.not_watermarked_threshold:
            return NOT_WATERMARKED
        return UNCERTAIN
