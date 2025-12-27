import statistics
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from x_configs import DEFAULT_WINDOW_SIZE, load_spacy_model, model
from .z_utils import aggregate_windows

"""
Text-wide log-probability and surprisal metrics (no IO).

What it produces (example dict for a single text):
{
  "filename": "book1.txt",
  "window_size": 3,
  "num_sentences": 120,
  "avg_log_prob": -2.31,
  "num_tokens": 18420,
  "top_words": [["the", 621], ["and", 402], ...],
  "sentence_log_probs": [[-3.1, -2.7, ...], ...],
  "sentence_log_prob_metrics": [
    {"sum_log_prob": -23.1, "mean_log_prob": -2.3, "perplexity": 9.97, "num_tokens": 10},
    ...
  ],
  "sentence_log_prob_metrics_windowed": [
    {"sum_log_prob": -69.3, "mean_log_prob": -2.31, "perplexity": 10.08, "num_tokens": 30,
     "start_sentence": 0, "end_sentence": 2, "sentences": [...]},
    ...
  ],
  "sentence_surprisal_metrics": [
    {"sentence_text": "First sentence.", "mean_surprisal": 2.31, "surprisal_variance": 0.12, "num_tokens": 10},
    ...
  ],
  "sentence_surprisal_metrics_windowed": [
    {"mean_surprisal": 2.28, "surprisal_variance": 0.08, "num_tokens": 30,
     "start_sentence": 0, "end_sentence": 2, "sentences": [...]},
    ...
  ],
  "window_size": 3
}
Use this module from the d-layer orchestrator to save outputs.
"""


class WholeTextMetrics:
    """
    Compute LLM token log-probabilities and derived corpus-level metrics for a text.
    Produces per-sentence scores and windowed aggregates; no file IO is performed here.
    """

    def __init__(self, lm_model=model, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model)
        self.model = AutoModelForCausalLM.from_pretrained(lm_model).to(self.device)
        self.model.eval()

    def compute_log_probs_per_sentence(
        self,
        text,
        nlp=None,
        chunk_size=2048,
        stride=0,
    ):
        if stride >= chunk_size:
            raise ValueError("stride must be smaller than chunk_size")

        nlp = nlp or load_spacy_model()
        doc = nlp(text)
        sentence_spans = [(sent.start_char, sent.end_char) for sent in doc.sents]
        if not sentence_spans:
            return []

        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        tokens = tokenized["input_ids"]
        offsets = tokenized["offset_mapping"]

        if not tokens:
            return [[] for _ in sentence_spans]

        token_log_probs = [None] * len(tokens)
        for i in range(0, len(tokens), chunk_size - stride):
            chunk_tokens = tokens[i : i + chunk_size]
            if len(chunk_tokens) < 2:
                continue

            inputs = torch.tensor([chunk_tokens]).to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                logits = outputs.logits[:, :-1]
                target_tokens = inputs[:, 1:]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                chunk_log_probs = (
                    log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)[0].tolist()
                )

            scored_start = stride if stride > 0 and i > 0 else 0
            for offset_idx, lp in enumerate(chunk_log_probs[scored_start:], start=scored_start):
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
    def compute_sentence_surprisal_metrics(sentence_log_probs, window_size=None, sentence_texts=None):
        """
        Compute surprisal-based metrics from sentence log-probs, with optional window aggregation.
        """
        sent_metrics = []
        for idx, log_probs in enumerate(sentence_log_probs):
            sent_text = None
            if sentence_texts and idx < len(sentence_texts):
                sent_text = sentence_texts[idx]

            if not log_probs:
                sent_metrics.append(
                    {
                        "sentence_text": sent_text,
                        "mean_surprisal": 0.0,
                        "surprisal_variance": 0.0,
                        "num_tokens": 0,
                    }
                )
                continue

            surprisals = [-lp for lp in log_probs]
            mean_surprisal = statistics.mean(surprisals)
            surprisal_variance = statistics.variance(surprisals) if len(surprisals) > 1 else 0.0

            sent_metrics.append(
                {
                    "sentence_text": sent_text,
                    "mean_surprisal": round(mean_surprisal, 6),
                    "surprisal_variance": round(surprisal_variance, 6),
                    "num_tokens": len(surprisals),
                }
            )

        windowed = aggregate_windows(sent_metrics, window_size) if window_size and window_size > 1 else []
        return sent_metrics, windowed

    def compute_corpus_frequencies(self, texts, lowercase=True, min_freq=1):
        """
        Computes corpus-level word frequencies from a list of texts.
        """
        word_counter = Counter()
        for text in texts:
            words = [w.lower() if lowercase else w for w in text.split() if w.isalpha()]
            word_counter.update(words)

        corpus_freqs = {w: freq for w, freq in word_counter.items() if freq >= min_freq}
        return corpus_freqs

    def build_metrics_for_text(self, text, filename, nlp=None, window_size=DEFAULT_WINDOW_SIZE):
        """
        Compute corpus-level and windowed log-prob/surprisal metrics for a single text.

        Returns:
            dict shaped like the module example (filename/window_size/num_sentences/model/word stats + per-sentence and windowed metrics).

        Example:
            >>> metrics = WholeTextMetrics().build_metrics_for_text("Short text.", "demo.txt")
            >>> metrics["sentence_surprisal_metrics"][0]["mean_surprisal"]
            2.1
        """
        nlp = nlp or load_spacy_model()

        sentence_log_probs = self.compute_log_probs_per_sentence(
            text,
            nlp=nlp,
            chunk_size=2048,
        )
        sentence_log_prob_metrics = self.summarize_sentence_log_probs(sentence_log_probs)
        windowed_sentence_log_prob_metrics = aggregate_windows(
            sentence_log_prob_metrics, window_size
        ) if sentence_log_prob_metrics else []

        sentence_texts = [sent.text for sent in nlp(text).sents] if text else None
        num_sentences = len(sentence_texts) if sentence_texts else 0
        surprisal_metrics, windowed_surprisal_metrics = self.compute_sentence_surprisal_metrics(
            sentence_log_probs,
            window_size=window_size,
            sentence_texts=sentence_texts,
        )

        avg_log_prob = None
        total_scored_tokens = sum(len(lp_list) for lp_list in sentence_log_probs)
        if total_scored_tokens > 0:
            total_log_prob = sum(sum(lp_list) for lp_list in sentence_log_probs)
            avg_log_prob = total_log_prob / total_scored_tokens

        corpus_freq = self.compute_corpus_frequencies([text], min_freq=2)

        return {
            "filename": filename,
            "window_size": window_size,
            "num_sentences": num_sentences,
            "text": text,
            "model": model,
            "avg_log_prob": avg_log_prob,
            "num_tokens": total_scored_tokens,
            "top_words": sorted(corpus_freq.items(), key=lambda x: x[1], reverse=True)[:50],
            "sentence_log_probs": sentence_log_probs,
            "sentence_log_prob_metrics": sentence_log_prob_metrics,
            "sentence_log_prob_metrics_windowed": windowed_sentence_log_prob_metrics,
            "sentence_surprisal_metrics": surprisal_metrics,
            "sentence_surprisal_metrics_windowed": windowed_surprisal_metrics,
            "window_size": window_size,
        }
