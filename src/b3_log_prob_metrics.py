"""
Text-wide log-probability and surprisal metrics (no IO).

Input (build_metrics_for_text):
{
  "text": "<raw text string>",
  "filename": "book1.txt",
  "window_size": 3,
  "sentence_spans": [[0, 42], [43, 88], ...]  # optional; list of [start_char, end_char]
}

Output:
{
  "meta": {
    "filename": "book1.txt",
    "window_size": 3,
    "num_sentences": 120,
    "model": "gpt2",
    "avg_log_prob": -2.31
  },
  "sentences": [
    {
      "sentence_id": 0,
      "sentence_log_probs": [-3.1, -2.7, ...],
      "sentence_log_prob_metrics": {
        "sum_log_prob": -23.1,
        "mean_log_prob": -2.3,
        "perplexity": 9.97,
        "num_tokens": 10
      },
      "sentence_surprisal_metrics": {
        "mean_surprisal": 2.31,
        "surprisal_variance": 0.12,
        "num_tokens": 10
      }
    }
  ],
  "windows": [
    {
      "start_sentence": 0,
      "end_sentence": 2,
      "sentence_log_prob_metrics": {
        "sum_log_prob": -69.3,
        "mean_log_prob": -2.31,
        "perplexity": 10.08,
        "num_tokens": 30
      },
      "sentence_surprisal_metrics": {
        "mean_surprisal": 2.28,
        "surprisal_variance": 0.08,
        "num_tokens": 30
      },
      "token_weighted_mean_log_prob": -2.31,
      "token_weighted_perplexity": 10.08,
      "token_weighted_mean_surprisal": 2.28,
      "token_weighted_surprisal_variance": 0.08,
      "mean_log_prob_per_token": -2.31,
      "perplexity_per_token": 10.08,
      "mean_surprisal_per_token": 2.28,
      "surprisal_variance_per_token": 0.08,
      "token_count": 30
    }
  ]
}
"""

import math
import statistics
import warnings
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .x_configs import DEFAULT_WINDOW_SIZE, MODEL_CONFIGS
from .z_utils import aggregate_windows, load_spacy_model


class WholeTextMetrics:
    """
    Compute LLM token log-probabilities and derived corpus-level metrics for a text.
    Produces per-sentence scores and windowed aggregates in the shared meta/sentences/windows schema; no file IO.
    """

    def __init__(self, lm_model=MODEL_CONFIGS["causal_lm"], device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model)
        # Disable tokenizer max-length enforcement so we can tokenize whole documents; we chunk before the model call.
        self.tokenizer.model_max_length = int(1e9)
        self.model = AutoModelForCausalLM.from_pretrained(lm_model).to(self.device)
        self.model.eval()

    def compute_log_probs_per_sentence(
        self,
        text,
        nlp=None,
        chunk_size=2048,
        stride=256,
        sentence_spans=None,
    ):
        # Ensure we respect the model's maximum position embeddings (e.g., GPT-2 = 1024).
        max_len = (
            getattr(self.model.config, "n_positions", None)
            or getattr(self.model.config, "max_position_embeddings", None)
            or getattr(self.tokenizer, "model_max_length", None)
        )
        if max_len is None or max_len <= 0:
            max_len = 1024
        effective_chunk = min(chunk_size, max_len)
        if effective_chunk <= 0:
            raise ValueError("chunk_size must be positive")
        if stride >= effective_chunk:
            raise ValueError("stride must be smaller than chunk_size")

        if sentence_spans is None:
            nlp = nlp or load_spacy_model()
            doc = nlp(text)
            sentence_spans = [(sent.start_char, sent.end_char) for sent in doc.sents]
        if not sentence_spans:
            return []

        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            max_length=None,  # avoid HF max-length errors; we handle chunking ourselves
            truncation=False,
        )
        tokens = tokenized["input_ids"]
        offsets = tokenized["offset_mapping"]

        if not tokens:
            return [[] for _ in sentence_spans]

        token_log_probs = [None] * len(tokens)
        for i in range(0, len(tokens), effective_chunk - stride):
            chunk_tokens = tokens[i : i + effective_chunk]
            if len(chunk_tokens) < 2:
                continue

            inputs = torch.tensor([chunk_tokens]).to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                logits = outputs.logits[:, :-1]
                target_tokens = inputs[:, 1:]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                chunk_log_probs = (
                    log_probs.gather(2, target_tokens.unsqueeze(-1))
                    .squeeze(-1)[0]
                    .tolist()
                )

            if stride > 0 and i > 0:
                scored_start = max(stride - 1, 0)
            else:
                scored_start = 0
            for offset_idx, lp in enumerate(
                chunk_log_probs[scored_start:], start=scored_start
            ):
                token_index = i + 1 + offset_idx
                if token_index < len(token_log_probs) and token_log_probs[token_index] is None:
                    token_log_probs[token_index] = float(lp)

        sentence_log_probs = [[] for _ in sentence_spans]
        sent_idx = 0
        for token_idx, (start_char, end_char) in enumerate(offsets):
            while sent_idx < len(sentence_spans) and start_char >= sentence_spans[sent_idx][1]:
                sent_idx += 1
            if sent_idx >= len(sentence_spans):
                break
            sent_start, sent_end = sentence_spans[sent_idx]
            if start_char < sent_start or end_char > sent_end:
                continue
            log_prob = token_log_probs[token_idx]
            if log_prob is not None:
                sentence_log_probs[sent_idx].append(log_prob)

        return sentence_log_probs

    @staticmethod
    def summarize_sentence_log_probs(sentence_log_probs):
        metrics = []
        for log_probs in sentence_log_probs:
            if not log_probs:
                metrics.append(
                    {
                        "sum_log_prob": 0.0,
                        "mean_log_prob": 0.0,
                        "perplexity": 0.0,
                        "num_tokens": 0,
                    }
                )
                continue
            sum_log_prob = float(sum(log_probs))
            mean_log_prob = sum_log_prob / len(log_probs)
            perplexity = float(np.exp(-mean_log_prob))
            metrics.append(
                {
                    "sum_log_prob": round(sum_log_prob, 6),
                    "mean_log_prob": round(mean_log_prob, 6),
                    "perplexity": round(perplexity, 6),
                    "num_tokens": len(log_probs),
                }
            )
        return metrics

    @staticmethod
    def compute_sentence_surprisal_metrics(sentence_log_probs):
        """
        Compute surprisal-based metrics from sentence log-probs.
        """
        sent_metrics = []
        for log_probs in sentence_log_probs:
            if not log_probs:
                sent_metrics.append(
                    {
                        "mean_surprisal": 0.0,
                        "surprisal_variance": 0.0,
                        "num_tokens": 0,
                    }
                )
                continue

            surprisals = [-lp for lp in log_probs]
            mean_surprisal = statistics.mean(surprisals)
            surprisal_variance = statistics.pvariance(surprisals) if surprisals else 0.0

            sent_metrics.append(
                {
                    "mean_surprisal": round(mean_surprisal, 6),
                    "surprisal_variance": round(surprisal_variance, 6),
                    "num_tokens": len(surprisals),
                }
            )

        return sent_metrics

    @staticmethod
    def _distribution_tokens(text, nlp=None, lowercase=True, alpha_only=True):
        nlp = nlp or load_spacy_model()
        doc = nlp.make_doc(text or "")
        tokens = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if alpha_only and not token.is_alpha:
                continue
            token_text = token.text.lower() if lowercase else token.text
            if token_text:
                tokens.append(token_text)
        return tokens

    @classmethod
    def compare_text_distributions(
        cls,
        reference_text,
        comparison_text,
        *,
        nlp=None,
        lowercase=True,
        alpha_only=True,
        smoothing=1e-9,
    ):
        """Compare two texts via cross-entropy and KL divergence over token distributions."""
        reference_tokens = cls._distribution_tokens(
            reference_text,
            nlp=nlp,
            lowercase=lowercase,
            alpha_only=alpha_only,
        )
        comparison_tokens = cls._distribution_tokens(
            comparison_text,
            nlp=nlp,
            lowercase=lowercase,
            alpha_only=alpha_only,
        )

        if not reference_tokens or not comparison_tokens:
            return {
                "cross_entropy": 0.0,
                "kl_divergence": 0.0,
                "reference_entropy": 0.0,
                "vocabulary_size": 0,
                "reference_token_count": len(reference_tokens),
                "comparison_token_count": len(comparison_tokens),
            }

        reference_counts = Counter(reference_tokens)
        comparison_counts = Counter(comparison_tokens)
        vocabulary = sorted(set(reference_counts) | set(comparison_counts))
        if not vocabulary:
            return {
                "cross_entropy": 0.0,
                "kl_divergence": 0.0,
                "reference_entropy": 0.0,
                "vocabulary_size": 0,
                "reference_token_count": len(reference_tokens),
                "comparison_token_count": len(comparison_tokens),
            }

        reference_total = float(sum(reference_counts.values()))
        comparison_total = float(sum(comparison_counts.values()))
        epsilon = max(float(smoothing), 1e-12)
        smoothing_mass = epsilon * len(vocabulary)

        reference_entropy = 0.0
        cross_entropy = 0.0
        kl_divergence = 0.0
        for token in vocabulary:
            p = (reference_counts.get(token, 0.0) + epsilon) / (reference_total + smoothing_mass)
            q = (comparison_counts.get(token, 0.0) + epsilon) / (comparison_total + smoothing_mass)
            reference_entropy -= p * math.log2(p)
            cross_entropy -= p * math.log2(q)
            kl_divergence += p * math.log2(p / q)

        return {
            "cross_entropy": round(cross_entropy, 6),
            "kl_divergence": round(max(kl_divergence, 0.0), 6),
            "reference_entropy": round(reference_entropy, 6),
            "vocabulary_size": len(vocabulary),
            "reference_token_count": len(reference_tokens),
            "comparison_token_count": len(comparison_tokens),
        }

    def compute_corpus_frequencies(self, texts, nlp=None, lowercase=True, min_freq=1):
        """
        Computes corpus-level word frequencies from a list of texts.
        Tokenization is aligned with spaCy's token.is_alpha when available.
        """
        nlp = nlp or load_spacy_model()

        word_counter = Counter()
        total_tokens = 0
        for text in texts:
            doc = nlp.make_doc(text)
            words = [
                (token.text.lower() if lowercase else token.text)
                for token in doc
                if token.is_alpha
            ]
            total_tokens += len(words)
            word_counter.update(words)

        corpus_freqs = {w: freq for w, freq in word_counter.items() if freq >= min_freq}
        return {
            "meta": {
                "lowercase": lowercase,
                "min_freq": min_freq,
                "num_texts": len(texts),
                "total_tokens": total_tokens,
                "vocab_size": len(corpus_freqs),
            },
            "word_frequencies": corpus_freqs,
        }

    def build_metrics_for_text(
        self,
        text,
        filename,
        nlp=None,
        window_size=DEFAULT_WINDOW_SIZE,
        sentence_spans=None,
    ):
        """
        Compute corpus-level and windowed log-prob/surprisal metrics for a single text.

        Returns:
            dict with meta, sentences (per-sentence metrics), windows (aggregated), and optional heavy payloads.
            Window rows are produced via aggregate_windows: contiguous spans of `window_size` sentences are
            averaged for numeric values (including nested dicts) and tagged with inclusive start/end indices;
            raw sentence strings are excluded.

        Example:
            >>> metrics = WholeTextMetrics().build_metrics_for_text("Short text.", "demo.txt")
            >>> metrics["sentences"][0]["mean_surprisal"]
            2.1
        """
        nlp = nlp or load_spacy_model()

        sentence_log_probs = self.compute_log_probs_per_sentence(
            text,
            nlp=nlp,
            chunk_size=2048,
            sentence_spans=sentence_spans,
        )
        sentence_log_prob_metrics = self.summarize_sentence_log_probs(sentence_log_probs)

        num_sentences = len(sentence_log_probs)
        surprisal_metrics = self.compute_sentence_surprisal_metrics(sentence_log_probs)

        avg_log_prob = None
        total_scored_tokens = sum(len(lp_list) for lp_list in sentence_log_probs)
        if total_scored_tokens > 0:
            total_log_prob = sum(sum(lp_list) for lp_list in sentence_log_probs)
            avg_log_prob = total_log_prob / total_scored_tokens
        else:
            warnings.warn(
                f"No scored tokens for {filename}; check tokenization offsets/sentence spans.",
                RuntimeWarning,
            )

        # Build aligned sentence-level payloads with nested metrics and raw log-probs
        sentences = []
        for idx, (lp_metrics, sp_metrics, log_probs) in enumerate(
            zip(sentence_log_prob_metrics, surprisal_metrics, sentence_log_probs)
        ):
            sentences.append(
                {
                    "sentence_id": idx,
                    "sentence_log_probs": log_probs,
                    "sentence_log_prob_metrics": lp_metrics,
                    "sentence_surprisal_metrics": sp_metrics,
                }
            )

        # Aggregate window-level metrics aligned with the window payloads
        window_inputs = [
            {
                "sentence_log_prob_metrics": sent.get("sentence_log_prob_metrics"),
                "sentence_surprisal_metrics": sent.get("sentence_surprisal_metrics"),
            }
            for sent in sentences
        ]
        windows = aggregate_windows(window_inputs, window_size) if window_inputs else []

        if windows:
            def _prefix(values):
                acc = [0.0]
                running = 0.0
                for value in values:
                    running += float(value)
                    acc.append(running)
                return acc

            def _range_sum(prefix, start, end):
                if start < 0 or end < start:
                    return 0.0
                return prefix[end + 1] - prefix[start]

            token_counts = [
                metrics.get("num_tokens", 0) for metrics in sentence_log_prob_metrics
            ]
            sum_log_probs = [
                metrics.get("sum_log_prob", 0.0) for metrics in sentence_log_prob_metrics
            ]
            mean_surprisals = [
                metrics.get("mean_surprisal", 0.0) for metrics in surprisal_metrics
            ]
            surprisal_variances = [
                metrics.get("surprisal_variance", 0.0) for metrics in surprisal_metrics
            ]
            max_surprisal_per_sentence = []
            for sent_log_probs in sentence_log_probs:
                if not isinstance(sent_log_probs, list) or not sent_log_probs:
                    max_surprisal_per_sentence.append(0.0)
                    continue
                max_surprisal = 0.0
                for lp in sent_log_probs:
                    if isinstance(lp, (int, float)) and not (
                        isinstance(lp, float) and math.isnan(lp)
                    ):
                        max_surprisal = max(max_surprisal, -float(lp))
                max_surprisal_per_sentence.append(max_surprisal)

            token_prefix = _prefix(token_counts)
            sum_log_prob_prefix = _prefix(sum_log_probs)
            surprisal_sum_prefix = _prefix(
                [
                    mean * count for mean, count in zip(mean_surprisals, token_counts)
                ]
            )
            surprisal_sq_prefix = _prefix(
                [
                    (mean ** 2) * count for mean, count in zip(mean_surprisals, token_counts)
                ]
            )
            surprisal_var_prefix = _prefix(
                [
                    var * count for var, count in zip(surprisal_variances, token_counts)
                ]
            )

            for window_idx, window in enumerate(windows):
                start = int(window.get("start_sentence", 0))
                end = int(window.get("end_sentence", start))
                total_tokens = _range_sum(token_prefix, start, end)
                sum_log_prob = _range_sum(sum_log_prob_prefix, start, end)
                max_token_surprisal = (
                    max(max_surprisal_per_sentence[start : end + 1])
                    if start <= end
                    else 0.0
                )

                if total_tokens > 0:
                    token_weighted_mean_log_prob = sum_log_prob / total_tokens
                    token_weighted_perplexity = math.exp(-token_weighted_mean_log_prob)
                    sum_surprisal = _range_sum(surprisal_sum_prefix, start, end)
                    token_weighted_mean_surprisal = sum_surprisal / total_tokens
                    pooled_variance = _range_sum(surprisal_var_prefix, start, end)
                    mean_sq_sum = _range_sum(surprisal_sq_prefix, start, end)
                    pooled_variance += (
                        mean_sq_sum
                        - 2 * token_weighted_mean_surprisal * sum_surprisal
                        + (token_weighted_mean_surprisal ** 2) * total_tokens
                    )
                    token_weighted_surprisal_variance = pooled_variance / total_tokens
                else:
                    token_weighted_mean_log_prob = 0.0
                    token_weighted_perplexity = 0.0
                    token_weighted_mean_surprisal = 0.0
                    token_weighted_surprisal_variance = 0.0

                windows[window_idx]["token_weighted_mean_log_prob"] = round(
                    token_weighted_mean_log_prob, 6
                )
                windows[window_idx]["token_weighted_perplexity"] = round(
                    token_weighted_perplexity, 6
                )
                windows[window_idx]["token_weighted_mean_surprisal"] = round(
                    token_weighted_mean_surprisal, 6
                )
                windows[window_idx]["token_weighted_surprisal_variance"] = round(
                    token_weighted_surprisal_variance, 6
                )
                windows[window_idx]["max_token_surprisal"] = round(max_token_surprisal, 6)
                windows[window_idx]["sentence_log_prob_metrics"] = {
                    "sum_log_prob": round(sum_log_prob, 6),
                    "mean_log_prob": round(token_weighted_mean_log_prob, 6),
                    "perplexity": round(token_weighted_perplexity, 6),
                    "num_tokens": int(total_tokens),
                }
                windows[window_idx]["sentence_surprisal_metrics"] = {
                    "mean_surprisal": round(token_weighted_mean_surprisal, 6),
                    "surprisal_variance": round(token_weighted_surprisal_variance, 6),
                    "num_tokens": int(total_tokens),
                }
                windows[window_idx]["mean_log_prob_per_token"] = windows[window_idx][
                    "token_weighted_mean_log_prob"
                ]
                windows[window_idx]["perplexity_per_token"] = windows[window_idx][
                    "token_weighted_perplexity"
                ]
                windows[window_idx]["mean_surprisal_per_token"] = windows[window_idx][
                    "token_weighted_mean_surprisal"
                ]
                windows[window_idx]["surprisal_variance_per_token"] = windows[window_idx][
                    "token_weighted_surprisal_variance"
                ]
                windows[window_idx]["token_count"] = int(total_tokens)

        return {
            "meta": {
                "filename": filename,
                "window_size": window_size,
                "num_sentences": num_sentences,
                "model": MODEL_CONFIGS["causal_lm"],
                "avg_log_prob": avg_log_prob,
            },
            "sentences": sentences,
            "windows": windows,
        }
