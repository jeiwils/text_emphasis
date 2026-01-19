"""
Embedding-based topic modeling for long-form text.

Input (run_topic_modelling / load_segmented_topic_mentions):
JSONL lines with:
{
  "sentence_id": 0,
  "text": "Sentence text.",
  "start_char": 0,
  "end_char": 23
}

Output (topics file):
{
  "meta": {
    "filename": "book_normalised_segmented.jsonl",
    "num_sentences": 120
  },
  "params": {
    "base_window_size": 3,
    "window_multiple": 5,
    "model_window_size": 15,
    "window_stride": 6,
    "min_cluster_size": 5,
    "min_samples": null,
    "soft_score_threshold": 0.5,
    "soft_top_k_topics": 3,
    "use_pca": true,
    "pca_components": 50
  },
  "topics": {
    "items": [
      {
        "topic_id": 0,
        "keywords": ["term1", "term2", "..."],
        "stats": {
          "prevalence": 0.12,
          "persistence": 2.3,
          "coherence": 0.45,
          "exclusivity": 0.21,
          "top10_mean": 0.67
        },
        "mentions": [
          {
            "sentence_index": 10,
            "start_sentence": 10,
            "end_sentence": 12,
            "window_index": 4,
            "start_char": 500,
            "end_char": 620,
            "text": "Window text ..."
          }
        ]
      }
    ]
  },
  "windows": {
    "items": [
      {
        "window_index": 0,
        "start_sentence": 0,
        "end_sentence": 14,
        "topic_scores": [{"topic_id": 0, "score": 0.71}, {"topic_id": 1, "score": 0.42}],
        "is_noise": false
      }
    ]
  }
}
"""

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .z_utils import (
    analytics_path,
    encode_texts,
    hdbscan_cluster_labels,
    iter_dirs,
    l2_normalize_embeddings,
    sliding_windows,
    text_path,
)
from .x_configs import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_TOPIC_WINDOW_MULTIPLE,
    DEFAULT_TOPIC_WINDOW_STRIDE_MULTIPLE,
    GENRES,
    MODEL_CONFIGS,
    DEFAULT_SOFT_SCORE_THRESHOLD,
    DEFAULT_SOFT_TOP_K,
    DEFAULT_RNG_SEED,
    DEFAULT_CENTRALITY_TOP_SCORE_FRACTION,
)

@dataclass
class TopicMention:
    sentence_index: int
    text: str
    start_char: int
    end_char: int
    end_sentence: Optional[int] = None
    window_index: Optional[int] = None
    topic_scores: Optional[List[Dict[str, float]]] = None


@dataclass
class TopicResult:
    topic_id: int
    keywords: List[str]
    mentions: List[TopicMention]
    stats: Optional[Dict[str, float]] = None


def load_segmented_topic_mentions(jsonl_path: Path) -> List[TopicMention]:
    """
    Load pre-segmented sentences with offsets from a JSONL file into TopicMention objects.
    Raises if offsets are absent or invalid; downstream expects offsets to be present.
    """
    mentions: List[TopicMention] = []
    try:
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Unable to read segmented file: {jsonl_path}") from exc

    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        start_char = entry.get("start_char")
        end_char = entry.get("end_char")
        sentence_index = entry.get("sentence_id", entry.get("sentence_index", idx))
        if text is None or start_char is None or end_char is None:
            raise ValueError(
                f"Segmented file lacks offsets; regenerate preprocessing outputs for {jsonl_path.name}"
            )
        try:
            sentence_index = int(sentence_index)
            start_char = int(start_char)
            end_char = int(end_char)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid offsets in {jsonl_path.name} (line {idx + 1})")
        mentions.append(
            TopicMention(
                sentence_index=sentence_index,
                text=str(text),
                start_char=start_char,
                end_char=end_char,
            )
        )
    if not mentions:
        raise ValueError(f"No segmented sentences found in {jsonl_path}")
    return mentions


def _is_subphrase(tokens: List[str], longer_tokens: List[str]) -> bool:
    if len(longer_tokens) <= len(tokens):
        return False
    max_start = len(longer_tokens) - len(tokens)
    for start in range(max_start + 1):
        if longer_tokens[start : start + len(tokens)] == tokens:
            return True
    return False


def _drop_subphrase_keywords(
    keywords: List[str],
    scores: np.ndarray,
    *,
    max_terms: Optional[int] = None,
) -> List[str]:
    if not keywords:
        return []
    if max_terms is not None and max_terms <= 0:
        return []
    tokenized = [term.split() for term in keywords]
    order = list(range(len(keywords)))
    order.sort(key=lambda idx: (scores[idx], len(tokenized[idx])), reverse=True)
    kept: List[int] = []
    for idx in order:
        tokens = tokenized[idx]
        if any(
            _is_subphrase(tokens, tokenized[kept_idx])
            or _is_subphrase(tokenized[kept_idx], tokens)
            for kept_idx in kept
        ):
            continue
        kept.append(idx)
        if max_terms is not None and len(kept) >= max_terms:
            break
    return [keywords[idx] for idx in kept]


class EmbeddingTopicModeler:
    """
    Clusters sentence embeddings, extracts keywords, and returns
    localized mentions (sentence index + char offsets).
    """

    def __init__(
        self,
        model_name: str = MODEL_CONFIGS["sentence_embedding"],
        stop_words: str = "english",
        encoder: SentenceTransformer | None = None,
    ):
        self.encoder = encoder or SentenceTransformer(model_name)
        self.stop_words = stop_words

    def build_windows(
        self, sentences: List[TopicMention], window_size: int, stride: int = 1
    ) -> List[TopicMention]:
        """
        Build sliding windows (stride 1) of sentences.
        Each window is represented as a TopicMention with start/end sentence indices.
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        windows: List[TopicMention] = []
        for idx, window_sents in enumerate(sliding_windows(sentences, window_size, step=stride)):
            if not window_sents:
                continue
            start = window_sents[0]
            end = window_sents[-1]
            window_text = " ".join(s.text for s in window_sents)
            windows.append(
                TopicMention(
                    sentence_index=start.sentence_index,
                    end_sentence=end.sentence_index,
                    window_index=idx,
                    text=window_text,
                    start_char=start.start_char,
                    end_char=end.end_char,
                )
            )
        return windows

    def _build_topic_keywords(
        self,
        cluster_docs: List[str],
        labels: List[int],
        top_n: int = 8,
        ngram_range: Tuple[int, int] = (1, 3),
    ) -> Tuple[Dict[int, List[str]], TfidfVectorizer, np.ndarray]:
        """Build TF-IDF keywords for each topic cluster."""
        vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            ngram_range=ngram_range,
        )
        tfidf = vectorizer.fit_transform(cluster_docs)
        feature_names = np.array(vectorizer.get_feature_names_out())
        topic_term_presence = (tfidf > 0).sum(axis=0)
        term_topic_counts = np.asarray(topic_term_presence).ravel()
        topic_count = max(1, tfidf.shape[0])
        overlap_penalty = np.log((topic_count + 1) / (term_topic_counts + 1))

        keywords: Dict[int, List[str]] = {}
        for row_idx, label in enumerate(labels):
            row = tfidf[row_idx]
            if row.nnz == 0:
                keywords[label] = []
                continue
            indices = row.indices
            scores = row.data
            adjusted_scores = scores * overlap_penalty[indices]
            candidate_terms = feature_names[indices].tolist()
            keywords[label] = _drop_subphrase_keywords(
                candidate_terms,
                adjusted_scores,
                max_terms=top_n,
            )
        return keywords, vectorizer, term_topic_counts

    def _topic_prevalence_persistence(
        self,
        window_topics: List[Dict[str, object]],
        topic_ids: List[int],
    ) -> Dict[int, Dict[str, float]]:
        topic_windows: Dict[int, List[int]] = {topic_id: [] for topic_id in topic_ids}
        topic_score_sums: Dict[int, float] = {topic_id: 0.0 for topic_id in topic_ids}
        scored_windows = []
        for window in window_topics:
            if window.get("is_noise"):
                continue
            scores = window.get("topic_scores") or []
            scored_windows.append(window)
            for entry in scores:
                if not isinstance(entry, dict):
                    continue
                topic_id = entry.get("topic_id")
                score = entry.get("score")
                if topic_id in topic_score_sums and isinstance(score, (int, float)):
                    topic_score_sums[topic_id] += float(score)

        total_windows = max(1, len(scored_windows))
        for window in scored_windows:
            window_idx = window.get("window_index")
            if window_idx is None:
                continue
            scores = window.get("topic_scores") or []
            for entry in scores:
                if not isinstance(entry, dict):
                    continue
                topic_id = entry.get("topic_id")
                if topic_id in topic_windows:
                    topic_windows[topic_id].append(int(window_idx))

        stats: Dict[int, Dict[str, float]] = {}
        for topic_id, indices in topic_windows.items():
            indices.sort()
            prevalence = topic_score_sums.get(topic_id, 0.0) / total_windows if total_windows else 0.0
            if not indices:
                stats[topic_id] = {"prevalence": 0.0, "persistence": 0.0}
                continue
            run_lengths = []
            run = 1
            for prev, curr in zip(indices, indices[1:]):
                if curr == prev + 1:
                    run += 1
                else:
                    run_lengths.append(run)
                    run = 1
            run_lengths.append(run)
            persistence = float(np.mean(run_lengths)) if run_lengths else 0.0
            stats[topic_id] = {
                "prevalence": float(prevalence),
                "persistence": persistence,
            }
        return stats

    def _topic_top_fraction_means(
        self,
        window_topics: List[Dict[str, object]],
        topic_ids: List[int],
        *,
        fraction: float,
    ) -> Dict[int, float]:
        if fraction <= 0 or fraction > 1:
            raise ValueError("top score fraction must be in (0, 1]")
        scores_by_topic: Dict[int, List[float]] = {topic_id: [] for topic_id in topic_ids}
        for window in window_topics:
            if window.get("is_noise"):
                continue
            scores = window.get("topic_scores") or []
            for entry in scores:
                if not isinstance(entry, dict):
                    continue
                topic_id = entry.get("topic_id")
                score = entry.get("score")
                if topic_id in scores_by_topic and isinstance(score, (int, float)):
                    scores_by_topic[topic_id].append(float(score))
        means: Dict[int, float] = {}
        for topic_id, scores in scores_by_topic.items():
            if not scores:
                means[topic_id] = 0.0
                continue
            top_count = max(1, math.ceil(len(scores) * fraction))
            scores.sort()
            means[topic_id] = sum(scores[-top_count:]) / top_count
        return means

    def _topic_coherence_exclusivity(
        self,
        keywords: Dict[int, List[str]],
        vectorizer: TfidfVectorizer,
        term_topic_counts: np.ndarray,
        topic_window_texts: Dict[int, List[str]],
    ) -> Dict[int, Dict[str, float]]:
        if not keywords:
            return {}
        vocab = vectorizer.vocabulary_
        if not vocab:
            return {topic_id: {"coherence": 0.0, "exclusivity": 0.0} for topic_id in keywords}
        stats: Dict[int, Dict[str, float]] = {}
        num_topics = len(keywords)
        for topic_id, terms in keywords.items():
            topic_texts = topic_window_texts.get(topic_id, [])
            if not topic_texts:
                stats[topic_id] = {"coherence": 0.0, "exclusivity": 0.0}
                continue
            dtm = vectorizer.transform(topic_texts)
            dtm = (dtm > 0).astype(int)
            doc_count = max(1, dtm.shape[0])
            df = np.asarray(dtm.sum(axis=0)).ravel()
            term_indices = [vocab[t] for t in terms if t in vocab]
            if not term_indices:
                stats[topic_id] = {"coherence": 0.0, "exclusivity": 0.0}
                continue
            pair_scores = []
            for i, j in combinations(term_indices, 2):
                cooc = int(dtm[:, i].multiply(dtm[:, j]).sum())
                if cooc == 0:
                    continue
                p_xy = cooc / doc_count
                p_x = df[i] / doc_count
                p_y = df[j] / doc_count
                if p_x <= 0 or p_y <= 0 or p_xy <= 0 or p_xy >= 1:
                    continue
                pmi = np.log(p_xy / (p_x * p_y))
                denom = -np.log(p_xy)
                if denom <= 0:
                    continue
                npmi = pmi / denom
                pair_scores.append(npmi)
            coherence = float(np.mean(pair_scores)) if pair_scores else 0.0
            exclusivity_scores = []
            for idx in term_indices:
                count = term_topic_counts[idx] if idx < len(term_topic_counts) else 1
                count = max(1.0, float(count))
                if num_topics <= 1:
                    exclusivity_scores.append(1.0)
                else:
                    score = 1.0 - ((count - 1.0) / (num_topics - 1.0))
                    exclusivity_scores.append(max(0.0, min(1.0, score)))
            exclusivity = float(np.mean(exclusivity_scores)) if exclusivity_scores else 0.0
            stats[topic_id] = {
                "coherence": coherence,
                "exclusivity": exclusivity,
            }
        return stats

    def _build_topic_scores(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        cluster_labels: List[int],
        top_k_topics: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[List[Dict[str, float]]]:
        """
        Compute cosine similarity between each window embedding and topic centroids.
        Returns per-window lists of {"topic_id": int, "score": float}, optionally filtered.
        """
        if embeddings.size == 0 or not cluster_labels:
            return [[] for _ in range(len(embeddings))]

        centroid_vectors = []
        for label in cluster_labels:
            idxs = np.where(labels == label)[0]
            if len(idxs) == 0:
                centroid_vectors.append(np.zeros(embeddings.shape[1], dtype=float))
                continue
            if len(idxs) > 50:
                step = max(1, len(idxs) // 50)
                idxs = idxs[::step]
            centroid_vectors.append(embeddings[idxs].mean(axis=0))

        centroid_matrix = np.vstack(centroid_vectors)
        centroid_matrix = l2_normalize_embeddings(centroid_matrix)
        sim_matrix = cosine_similarity(embeddings, centroid_matrix)

        topic_scores: List[List[Dict[str, float]]] = []
        for idx, row in enumerate(sim_matrix):
            if labels[idx] == -1:
                topic_scores.append([])
                continue
            scores = [
                {"topic_id": int(label), "score": float(score)}
                for label, score in zip(cluster_labels, row)
            ]
            if score_threshold is not None:
                scores = [entry for entry in scores if entry["score"] >= score_threshold]
            if top_k_topics is not None and top_k_topics > 0:
                scores = sorted(scores, key=lambda kv: kv["score"], reverse=True)[:top_k_topics]
            topic_scores.append(scores)
        return topic_scores

    def extract_topics(
        self,
        sentences: List[TopicMention],
        min_cluster_size: Optional[int] = None,
        min_samples: Optional[int] = None,
        top_n: int = 8,
        ngram_range: Tuple[int, int] = (1, 3),
        window_multiple: int = DEFAULT_TOPIC_WINDOW_MULTIPLE,
        base_window_size: int = DEFAULT_WINDOW_SIZE,
        window_size: Optional[int] = None,
        window_stride: Optional[int] = None,
        top_k_topics: Optional[int] = DEFAULT_SOFT_TOP_K,
        score_threshold: Optional[float] = DEFAULT_SOFT_SCORE_THRESHOLD,
        use_pca: bool = True,
        pca_components: int = 50,
    ) -> Tuple[List[TopicResult], List[Dict[str, object]]]:
        """
        Main entrypoint: consumes pre-segmented sentences with offsets and returns clustered topics.

        Sentences are grouped into sliding windows of size
        `base_window_size * window_multiple` (default DEFAULT_WINDOW_SIZE * DEFAULT_TOPIC_WINDOW_MULTIPLE = 15)
        with stride `base_window_size * DEFAULT_TOPIC_WINDOW_STRIDE_MULTIPLE` (default 6) so that topic windows
        align to the sentence-scale metrics built on window size 3.
        """
        if not sentences:
            return [], []

        model_window_size = window_size or max(1, base_window_size * max(1, window_multiple))
        stride = window_stride or max(1, base_window_size * DEFAULT_TOPIC_WINDOW_STRIDE_MULTIPLE)
        windows = self.build_windows(sentences, model_window_size, stride=stride)
        window_texts = [w.text for w in windows]

        embeddings = encode_texts(self.encoder, window_texts)
        reduced_embeddings = embeddings
        if use_pca and embeddings.size and embeddings.shape[0] > 1:
            max_components = min(pca_components, embeddings.shape[0], embeddings.shape[1])
            if max_components > 0 and max_components < embeddings.shape[1]:
                pca = PCA(n_components=max_components, random_state=DEFAULT_RNG_SEED)
                reduced_embeddings = pca.fit_transform(embeddings)
        if len(window_texts) < min_cluster_size:
            labels = np.zeros(len(window_texts), dtype=int)
        else:
            labels = hdbscan_cluster_labels(
                reduced_embeddings,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )

        topic_docs: Dict[int, List[str]] = {}
        topic_mentions: Dict[int, List[TopicMention]] = {}
        for window, label in zip(windows, labels):
            if label == -1:
                continue
            label_int = int(label)
            topic_docs.setdefault(label_int, []).append(window.text)
            topic_mentions.setdefault(label_int, []).append(window)

        # Order topics by prominence (number of window mentions), then by label for stability.
        cluster_labels = sorted(
            topic_docs.keys(),
            key=lambda label: (-len(topic_mentions.get(label, [])), label),
        )
        cluster_docs = [" ".join(topic_docs[label]) for label in cluster_labels]
        keywords = {}
        vectorizer = None
        term_topic_counts = None
        if cluster_labels:
            keywords, vectorizer, term_topic_counts = self._build_topic_keywords(
                cluster_docs,
                cluster_labels,
                top_n=top_n,
                ngram_range=ngram_range,
            )
        topic_scores = self._build_topic_scores(
            embeddings=embeddings,
            labels=labels,
            cluster_labels=cluster_labels,
            top_k_topics=top_k_topics,
            score_threshold=score_threshold,
        )
        window_topics = []
        for idx, window in enumerate(windows):
            scores = topic_scores[idx] if topic_scores else []
            window.topic_scores = scores
            window_topics.append(
                {
                    "window_index": window.window_index
                    if window.window_index is not None
                    else idx,
                    "start_sentence": window.sentence_index,
                    "end_sentence": window.end_sentence,
                    "topic_scores": scores,
                    "is_noise": bool(labels[idx] == -1),
                }
            )

        prevalence_stats = self._topic_prevalence_persistence(window_topics, cluster_labels)
        top_fraction_means = self._topic_top_fraction_means(
            window_topics,
            cluster_labels,
            fraction=DEFAULT_CENTRALITY_TOP_SCORE_FRACTION,
        )
        topic_window_texts = {topic_id: [] for topic_id in cluster_labels}
        for idx, window in enumerate(window_topics):
            if window.get("is_noise"):
                continue
            scores = window.get("topic_scores") or []
            for entry in scores:
                if not isinstance(entry, dict):
                    continue
                topic_id = entry.get("topic_id")
                if topic_id in topic_window_texts:
                    topic_window_texts[topic_id].append(window_texts[idx])
        coherence_exclusivity = {}
        if vectorizer is not None and term_topic_counts is not None:
            coherence_exclusivity = self._topic_coherence_exclusivity(
                keywords,
                vectorizer,
                term_topic_counts,
                topic_window_texts,
            )

        results = []
        for label in cluster_labels:
            stats = {}
            stats.update(prevalence_stats.get(label, {}))
            stats.update(coherence_exclusivity.get(label, {}))
            stats["top10_mean"] = top_fraction_means.get(label, 0.0)
            results.append(
                TopicResult(
                    topic_id=int(label),
                    keywords=keywords.get(label, []),
                    mentions=topic_mentions.get(label, []),
                    stats=stats if stats else None,
                )
            )
        return results, window_topics


def serialize_topic_results(topic_results: List[TopicResult]) -> List[Dict[str, Any]]:
    """Convert TopicResult objects to plain dicts for JSON output."""
    return [
        {
            "topic_id": int(result.topic_id),
            "keywords": result.keywords,
            "stats": result.stats,
            "mentions": [
                {
                    "sentence_index": int(mention.sentence_index),
                    "end_sentence": int(mention.end_sentence)
                    if mention.end_sentence is not None
                    else None,
                    "window_index": int(mention.window_index)
                    if mention.window_index is not None
                    else None,
                    "start_char": int(mention.start_char),
                    "end_char": int(mention.end_char),
                    "text": mention.text,
                }
                for mention in result.mentions
            ],
        }
        for result in topic_results
    ]


def _topic_debug_stats(
    topic_results: List[TopicResult], window_topics: List[Dict[str, object]]
) -> Dict[str, object]:
    """Lightweight debug stats about clustering outcomes."""
    noise_windows = sum(1 for w in window_topics if w.get("is_noise"))
    hard_label_counts: Dict[str, int] = {}
    for window in window_topics:
        if window.get("is_noise"):
            continue
        scores = window.get("topic_scores") or []
        if not scores:
            continue
        best_topic = max(scores, key=lambda kv: kv.get("score", 0.0)).get("topic_id")
        if best_topic is None:
            continue
        hard_label_counts[str(best_topic)] = hard_label_counts.get(str(best_topic), 0) + 1

    score_entropies = []
    for window in window_topics:
        if window.get("is_noise"):
            continue
        scores = window.get("topic_scores") or []
        if not scores:
            continue
        values = np.array([entry.get("score", 0.0) for entry in scores], dtype=float)
        if values.size == 0:
            continue
        values = values - np.max(values)
        probs = np.exp(values)
        probs_sum = probs.sum()
        if probs_sum <= 0:
            continue
        probs = probs / probs_sum
        if probs.size == 1:
            norm_entropy = 0.0
        else:
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
            norm_entropy = float(entropy / np.log(probs.size))
        score_entropies.append(norm_entropy)

    return {
        "topic_count": len(topic_results),
        "window_count": len(window_topics),
        "noise_window_count": noise_windows,
        "hard_label_counts": hard_label_counts,
        "score_entropy_mean": float(np.mean(score_entropies)) if score_entropies else None,
        "score_entropy_median": float(np.median(score_entropies)) if score_entropies else None,
        "score_stability_mean": float(1 - np.mean(score_entropies)) if score_entropies else None,
        "score_stability_median": float(1 - np.median(score_entropies)) if score_entropies else None,
    }


def collect_soft_topic_mentions(topics_data: Optional[object]):
    """Build per-sentence topic mentions from window-level soft scores; skips noise windows."""
    if not topics_data or not isinstance(topics_data, dict):
        return []

    windows_section = topics_data.get("windows")
    if not isinstance(windows_section, dict):
        return []
    windows = windows_section.get("items")
    if not isinstance(windows, list):
        return []
    mentions = []
    for window in windows:
        if window.get("is_noise"):
            continue
        scores = window.get("topic_scores") or []
        items = []
        for entry in scores:
            if not isinstance(entry, dict):
                continue
            topic_id = entry.get("topic_id")
            score = entry.get("score")
            if isinstance(topic_id, int) and isinstance(score, (int, float)):
                items.append((topic_id, float(score)))
        items.sort(key=lambda kv: kv[1], reverse=True)
        try:
            start_sentence = int(window.get("start_sentence", 0))
            end_sentence = int(window.get("end_sentence", start_sentence))
        except (TypeError, ValueError):
            start_sentence = 0
            end_sentence = start_sentence
        for topic_id, score in items:
            mentions.append(
                {
                    "topic_id": topic_id,
                    "start_sentence": start_sentence,
                    "end_sentence": end_sentence,
                    "score": score,
                }
            )
    return mentions


def build_topic_window_metrics(topic_mentions, window_entries):
    """Aggregate topic mention counts for each sentence window."""
    metrics = []
    for window in window_entries:
        start_sentence = window.get("start_sentence", 0)
        end_sentence = window.get("end_sentence", 0)
        token_count = window.get("token_count", 0)
        window_mentions = [
            mention
            for mention in topic_mentions
            if (
                mention.get("start_sentence", mention.get("sentence_index", 0))
                <= end_sentence
                and mention.get("end_sentence", mention.get("sentence_index", 0))
                >= start_sentence
            )
        ]
        topic_counts = {}
        for mention in window_mentions:
            topic_id = mention["topic_id"]
            if topic_id is None:
                continue
            topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
        if isinstance(token_count, (int, float)) and token_count > 0:
            mention_count_per_token = round(len(window_mentions) / token_count, 6)
            unique_topic_count_per_token = round(len(topic_counts) / token_count, 6)
            topic_counts_per_token = {
                topic_id: round(count / token_count, 6) for topic_id, count in topic_counts.items()
            }
        else:
            mention_count_per_token = 0.0
            unique_topic_count_per_token = 0.0
            topic_counts_per_token = {}
        sorted_topics = sorted(
            topic_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        top_topic_ids = [topic_id for topic_id, _ in sorted_topics]
        metrics.append(
            {
                "start_sentence": start_sentence,
                "end_sentence": end_sentence,
                "topic_mention_count": len(window_mentions),
                "topic_mention_count_per_token": mention_count_per_token,
                "unique_topic_count": len(topic_counts),
                "unique_topic_count_per_token": unique_topic_count_per_token,
                "top_topic_ids": top_topic_ids,
                "topic_counts": topic_counts,
                "topic_counts_per_token": topic_counts_per_token,
            }
        )
    return metrics


def _flatten_numeric_metrics(entry: Dict[str, object], prefix: str) -> Dict[str, float]:
    """Flatten a window metric entry into dot-notated numeric fields."""
    metrics = {}
    for key, value in entry.items():
        if key in {"start_sentence", "end_sentence"}:
            continue
        metric_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, (int, float)):
                    metrics[f"{metric_key}.{sub_key}"] = float(sub_val)
        elif isinstance(value, (int, float)):
            metrics[metric_key] = float(value)
    return metrics


def build_window_metric_table(window_metrics_by_name: Dict[str, List[Dict[str, object]]]):
    """Create a table of flattened window metrics aligned by window index."""
    if not window_metrics_by_name:
        return []

    lengths = {name: len(entries) for name, entries in window_metrics_by_name.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Window metric lengths differ: {lengths}")

    window_count = next(iter(lengths.values()))
    table = []
    for idx in range(window_count):
        row = {}
        for name, entries in window_metrics_by_name.items():
            row.update(_flatten_numeric_metrics(entries[idx], name))
        table.append(row)
    return table


def select_metric_groups(
    window_result: Dict[str, object],
    metric_group_names: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, object]]]:
    """Select named metric groups from a windowed analysis result."""
    if metric_group_names is None:
        metric_group_names = [
            "clause_metrics",
            "clause_embedding_metrics",
            "dependency_complexity_metrics",
            "avg_word_freq_metrics",
            "lexical_information_content",
            "information_content_metrics",
            "semantic_structures",
            "lexical_density_metrics",
            "lexical_diversity_windowed",
            "cohesion_metrics",
            "semantic_role_metrics",
            "discourse_metrics",
        ]

    selected = {}
    for name in metric_group_names:
        metrics = window_result.get(name)
        if metrics:
            selected[name] = metrics
    return selected


def _window_count(num_sentences: int, window_size: int, stride: int) -> int:
    """Compute how many sliding windows a text yields."""
    if num_sentences < window_size:
        return 1
    return ((num_sentences - window_size) // stride) + 1


def _auto_cluster_params(window_count: int) -> Tuple[int, int]:
    """Choose HDBSCAN params based on window count with short-text guardrails."""
    if window_count < 40:
        return 3, 2
    min_cluster_size = max(3, round(0.03 * window_count))
    min_samples = max(2, min(min_cluster_size - 1, 4))
    return min_cluster_size, min_samples


def run_topic_modelling(
    use_existing: bool = True,
    window_multiple: int = DEFAULT_TOPIC_WINDOW_MULTIPLE,
    base_window_size: int = DEFAULT_WINDOW_SIZE,
    window_stride: Optional[int] = None,
    authors: Optional[List[str]] = None,
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    soft_score_threshold: Optional[float] = DEFAULT_SOFT_SCORE_THRESHOLD,
    soft_top_k_topics: Optional[int] = DEFAULT_SOFT_TOP_K,
    use_pca: bool = True,
    pca_components: int = 50,
    encoder: SentenceTransformer | None = None,
):
    """
    Batch topic modelling across normalised, segmented text files.

    Defaults: window size DEFAULT_WINDOW_SIZE * DEFAULT_TOPIC_WINDOW_MULTIPLE (15) with stride
    DEFAULT_WINDOW_SIZE * DEFAULT_TOPIC_WINDOW_STRIDE_MULTIPLE (6)
    so topic windows align to the sentence-level metrics that use window size 3.
    Output shape: data/analytics/topic_modelling/<category>/<name>/<name>_clustered_topics.json
    """
    modeler = EmbeddingTopicModeler(encoder=encoder)
    normalised_root = text_path("processed", "normalised_segmented_texts")
    output_root = analytics_path("topic")
    output_root.mkdir(parents=True, exist_ok=True)
    stride = window_stride or (base_window_size * DEFAULT_TOPIC_WINDOW_STRIDE_MULTIPLE)

    categories = list(iter_dirs(normalised_root, genres=GENRES, authors=authors, depth=2))
    processed = 0
    skipped = 0
    for category_key, subdir in tqdm(categories, desc="Topic modelling", ascii=True):
        genre, author = category_key.split("/", 1)
        out_subdir = output_root / genre / author
        out_subdir.mkdir(parents=True, exist_ok=True)

        files = sorted(subdir.glob("*.jsonl"))
        for file in tqdm(files, desc=f"Topic modelling: {genre}/{author}", leave=False, ascii=True):
            base_name = file.stem.replace("_normalised_segmented", "")
            text_dir = out_subdir / base_name
            text_dir.mkdir(parents=True, exist_ok=True)
            clustered_output_file = text_dir / f"{base_name}_clustered_topics.json"
            debug_file = text_dir / f"{base_name}_topic_debug.json"
            if use_existing and clustered_output_file.exists():
                skipped += 1
                continue

            segmented_mentions = load_segmented_topic_mentions(file)

            sentence_count = len(segmented_mentions)
            book_window_multiple = window_multiple
            book_window_stride = stride
            effective_window_size = base_window_size * book_window_multiple
            window_count = _window_count(
                sentence_count, effective_window_size, book_window_stride
            )
            auto_min_cluster_size, auto_min_samples = _auto_cluster_params(window_count)
            book_min_cluster_size = (
                auto_min_cluster_size if min_cluster_size is None else min_cluster_size
            )
            book_min_samples = auto_min_samples if min_samples is None else min_samples
            if book_min_samples > book_min_cluster_size:
                book_min_samples = book_min_cluster_size
            book_soft_score_threshold = soft_score_threshold
            book_soft_top_k_topics = soft_top_k_topics
            book_use_pca = use_pca
            book_pca_components = pca_components

            topic_results, window_topics = modeler.extract_topics(
                segmented_mentions,
                min_cluster_size=book_min_cluster_size,
                min_samples=book_min_samples,
                window_multiple=book_window_multiple,
                base_window_size=base_window_size,
                window_size=effective_window_size,
                window_stride=book_window_stride,
                top_k_topics=book_soft_top_k_topics,
                score_threshold=book_soft_score_threshold,
                use_pca=book_use_pca,
                pca_components=book_pca_components,
            )
            num_sentences = sentence_count
            result = {
                "meta": {
                    "filename": file.name,
                    "num_sentences": num_sentences,
                },
                "params": {
                    "base_window_size": base_window_size,
                    "window_multiple": book_window_multiple,
                    "model_window_size": effective_window_size,
                    "window_stride": book_window_stride,
                    "min_cluster_size": book_min_cluster_size,
                    "min_samples": book_min_samples,
                    "soft_score_threshold": book_soft_score_threshold,
                    "soft_top_k_topics": book_soft_top_k_topics,
                    "use_pca": book_use_pca,
                    "pca_components": book_pca_components,
                },
                "topics": {"items": serialize_topic_results(topic_results)},
                "windows": {"items": window_topics},
            }
            debug_stats = _topic_debug_stats(topic_results, window_topics)
            debug_stats["meta"] = {
                "filename": file.name,
                "window_count": len(window_topics),
                "model_window_size": effective_window_size,
                "window_stride": book_window_stride,
                "min_cluster_size": book_min_cluster_size,
                "min_samples": book_min_samples,
                "soft_score_threshold": book_soft_score_threshold,
                "soft_top_k_topics": book_soft_top_k_topics,
                "use_pca": book_use_pca,
                "pca_components": book_pca_components,
            }

            with open(clustered_output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(debug_stats, f, indent=2)

            processed += 1

    tqdm.write(f"Topic modelling complete: {processed} processed, {skipped} skipped.")


if __name__ == "__main__":
    run_topic_modelling()
