"""

Neural topic modeling for long-form text.

Input:
- pre-segmented sentences with char offsets (normalised_segmented JSONL)

Output:
- list of TopicResult objects:
  - topic_id (int)
  - keywords (top TF-IDF terms for the cluster)
  - mentions (sentence spans w/ character offsets for localisation)


15 window
3 stride? won't that make sentences that don't overlap weaker signal? 

multiple themes can be present in a single window, not necessarily one dominant topic

topic-window similarity matrix - join topics 

"""

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from z_utils import (
    analytics_path,
    encode_texts,
    hdbscan_cluster_labels,
    iter_genre_author_dirs,
    l2_normalize_embeddings,
    sliding_windows,
    text_path,
)
from x_configs import (
    DEFAULT_WINDOW_SIZE,
    GENRES,
    MODEL_CONFIGS,
    TOPIC_BOOK_OVERRIDES,
)

@dataclass
class TopicMention:
    sentence_index: int
    text: str
    start_char: int
    end_char: int
    end_sentence: Optional[int] = None
    window_index: Optional[int] = None
    topic_scores: Optional[Dict[int, float]] = None


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


class NeuralTopicModeler:
    """
    Clusters sentence embeddings, extracts keywords, and returns
    localized mentions (sentence index + char offsets).
    """

    def __init__(
        self,
        model_name: str = MODEL_CONFIGS["sentence_embedding"],
        stop_words: str = "english",
    ):
        self.encoder = SentenceTransformer(model_name)
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
            scores = row.toarray().ravel()
            adjusted_scores = scores * overlap_penalty
            top_indices = adjusted_scores.argsort()[::-1][:top_n]
            keywords[label] = feature_names[top_indices].tolist()
        return keywords, vectorizer, term_topic_counts

    def _topic_prevalence_persistence(
        self,
        window_topics: List[Dict[str, object]],
        topic_ids: List[int],
    ) -> Dict[int, Dict[str, float]]:
        topic_windows: Dict[int, List[int]] = {topic_id: [] for topic_id in topic_ids}
        total_windows = sum(1 for window in window_topics if not window.get("is_noise"))
        total_windows = max(1, total_windows)
        for window in window_topics:
            if window.get("is_noise"):
                continue
            window_idx = window.get("window_index")
            if window_idx is None:
                continue
            scores = window.get("topic_scores") or {}
            for topic_id in topic_ids:
                if topic_id in scores:
                    topic_windows[topic_id].append(int(window_idx))

        stats: Dict[int, Dict[str, float]] = {}
        for topic_id, indices in topic_windows.items():
            indices.sort()
            prevalence = len(indices) / total_windows if total_windows else 0.0
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
                exclusivity_scores.append(1.0 / max(1.0, float(count)))
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
    ) -> List[Dict[int, float]]:
        """
        Compute cosine similarity between each window embedding and topic centroids.
        Returns per-window dicts of topic_id -> similarity, optionally filtered.
        """
        if embeddings.size == 0 or not cluster_labels:
            return [{} for _ in range(len(embeddings))]

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

        topic_scores: List[Dict[int, float]] = []
        for idx, row in enumerate(sim_matrix):
            if labels[idx] == -1:
                topic_scores.append({})
                continue
            scores = {int(label): float(score) for label, score in zip(cluster_labels, row)}
            if score_threshold is not None:
                scores = {k: v for k, v in scores.items() if v >= score_threshold}
            if top_k_topics is not None and top_k_topics > 0:
                scores = dict(
                    sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[
                        :top_k_topics
                    ]
                )
            topic_scores.append(scores)
        return topic_scores

    def extract_topics(
        self,
        sentences: List[TopicMention],
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        top_n: int = 8,
        ngram_range: Tuple[int, int] = (1, 3),
        window_multiple: int = 5,
        base_window_size: int = DEFAULT_WINDOW_SIZE,
        window_size: Optional[int] = None,
        window_stride: Optional[int] = None,
        top_k_topics: Optional[int] = 5,
        score_threshold: Optional[float] = None,
        use_pca: bool = True,
        pca_components: int = 50,
    ) -> Tuple[List[TopicResult], List[Dict[str, object]]]:
        """
        Main entrypoint: consumes pre-segmented sentences with offsets and returns clustered topics.

        Sentences are grouped into sliding windows of size
        `base_window_size * window_multiple` (default 3 * 5 = 15)
        with stride `base_window_size * 2` (default 6) so that topic windows
        align to the sentence-scale metrics built on window size 3.
        """
        if not sentences:
            return [], []

        model_window_size = window_size or max(1, base_window_size * max(1, window_multiple))
        stride = window_stride or max(1, base_window_size * 2)
        windows = self.build_windows(sentences, model_window_size, stride=stride)
        window_texts = [w.text for w in windows]

        embeddings = encode_texts(self.encoder, window_texts)
        reduced_embeddings = embeddings
        if use_pca and embeddings.size and embeddings.shape[0] > 1:
            max_components = min(pca_components, embeddings.shape[0], embeddings.shape[1])
            if max_components > 0 and max_components < embeddings.shape[1]:
                pca = PCA(n_components=max_components, random_state=42)
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
            scores = topic_scores[idx] if topic_scores else {}
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
        topic_window_texts = {topic_id: [] for topic_id in cluster_labels}
        for idx, window in enumerate(window_topics):
            if window.get("is_noise"):
                continue
            scores = window.get("topic_scores") or {}
            for topic_id in scores.keys():
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
        scores = window.get("topic_scores") or {}
        if not scores:
            continue
        best_topic = max(scores.items(), key=lambda kv: kv[1])[0]
        hard_label_counts[str(best_topic)] = hard_label_counts.get(str(best_topic), 0) + 1

    score_entropies = []
    for window in window_topics:
        if window.get("is_noise"):
            continue
        scores = window.get("topic_scores") or {}
        if not scores:
            continue
        values = np.array(list(scores.values()), dtype=float)
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

def _iter_sentence_span(mention: TopicMention):
    start = mention.sentence_index
    end = mention.end_sentence if mention.end_sentence is not None else mention.sentence_index
    return range(start, end + 1)


def count_mentions_per_sentence(topic_results: List[TopicResult]) -> Dict[int, int]:
    """
    Count how many topic mentions overlap each sentence index.
    If a mention spans multiple sentences, every sentence in the span is counted.
    """
    counts: Dict[int, int] = {}
    for result in topic_results:
        for mention in result.mentions:
            for idx in _iter_sentence_span(mention):
                counts[idx] = counts.get(idx, 0) + 1
    return counts





def collect_topic_mentions(topics_data):
    """Extract per-sentence topic mentions from a topic model result."""
    if not topics_data:
        return []

    if isinstance(topics_data, dict):
        topics = (
            topics_data.get("topics")
            or topics_data.get("topic_results")
            or topics_data.get("results")
            or []
        )
    elif isinstance(topics_data, list):
        topics = topics_data
    else:
        topics = []

    mentions = []
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        topic_id = topic.get("topic_id", topic.get("id"))
        for mention in topic.get("mentions", []):
            if not isinstance(mention, dict):
                continue
            sentence_index = mention.get("sentence_index")
            if sentence_index is None:
                continue
            start_sentence = mention.get("start_sentence", sentence_index)
            end_sentence = mention.get("end_sentence", sentence_index)
            mentions.append(
                {
                    "topic_id": topic_id,
                    "sentence_index": sentence_index,
                    "start_sentence": start_sentence,
                    "end_sentence": end_sentence,
                }
            )
    return mentions


def collect_soft_topic_mentions(
    topics_data: Optional[object],
    score_threshold: Optional[float] = 0.5,
    top_k: Optional[int] = None,
):
    """
    Build per-sentence topic mentions from window-level soft scores.
    Filters by optional top_k and/or score_threshold; skips noise windows.
    """
    if not topics_data or not isinstance(topics_data, dict):
        return []

    windows = topics_data.get("windows") or []
    mentions = []
    for window in windows:
        if window.get("is_noise"):
            continue
        scores = window.get("topic_scores") or {}
        items = []
        for k, v in scores.items():
            try:
                topic_id = int(k)
                score = float(v)
            except (TypeError, ValueError):
                continue
            items.append((topic_id, score))
        items.sort(key=lambda kv: kv[1], reverse=True)
        if top_k is not None and top_k > 0:
            items = items[:top_k]
        if score_threshold is not None:
            items = [(tid, s) for tid, s in items if s >= score_threshold]
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


def compute_topic_metric_report(
    topic_metrics: List[Dict[str, object]],
    window_metrics_by_name: Dict[str, List[Dict[str, object]]],
    min_topic_mentions: int = 1,
    min_windows: int = 2,
):
    """Compute per-topic comparisons across windowed metric values."""
    window_table = build_window_metric_table(window_metrics_by_name)
    metric_names = sorted(window_table[0].keys()) if window_table else []

    topics = set()
    for entry in topic_metrics:
        for topic_id in entry.get("topic_counts", {}).keys():
            topics.add(topic_id)

    report = {
        "window_count": len(window_table),
        "metric_names": metric_names,
        "topics": {},
    }

    for topic_id in sorted(topics):
        windows_with_topic = []
        windows_without_topic = []
        for idx, window_row in enumerate(window_table):
            topic_count = topic_metrics[idx].get("topic_counts", {}).get(topic_id, 0)
            if topic_count >= min_topic_mentions:
                windows_with_topic.append(window_row)
            else:
                windows_without_topic.append(window_row)

        topic_entry = {
            "window_count_with_topic": len(windows_with_topic),
            "window_count_without_topic": len(windows_without_topic),
            "metrics": {},
        }

        for metric in metric_names:
            values_with = [row[metric] for row in windows_with_topic if metric in row]
            values_without = [row[metric] for row in windows_without_topic if metric in row]

            variance_with = (
                float(np.var(values_with, ddof=1)) if len(values_with) >= min_windows else None
            )
            variance_without = (
                float(np.var(values_without, ddof=1)) if len(values_without) >= min_windows else None
            )
            variance_delta = (
                variance_with - variance_without
                if variance_with is not None and variance_without is not None
                else None
            )
            variance_ratio = (
                variance_with / variance_without
                if variance_with is not None and variance_without not in (None, 0)
                else None
            )

            topic_entry["metrics"][metric] = {
                "variance_with_topic": variance_with,
                "variance_without_topic": variance_without,
                "variance_delta": variance_delta,
                "variance_ratio": variance_ratio,
                "n_with_topic": len(values_with),
                "n_without_topic": len(values_without),
            }

        report["topics"][topic_id] = topic_entry

    return report


def compute_topic_metric_report_from_window_result(
    window_result: Dict[str, object],
    topics_data: Optional[object] = None,
    metric_group_names: Optional[List[str]] = None,
    min_topic_mentions: int = 1,
    min_windows: int = 2,
    use_soft_topic_scores: bool = False,
    soft_score_threshold: Optional[float] = 0.5,
    soft_top_k: Optional[int] = None,
):
    """Compute a topic/metric report from a window result and topic model output."""
    topic_mentions = (
        collect_soft_topic_mentions(
            topics_data,
            score_threshold=soft_score_threshold,
            top_k=soft_top_k,
        )
        if use_soft_topic_scores
        else collect_topic_mentions(topics_data)
    )
    window_entries = window_result.get("syntax", {}).get("windows", [])
    topic_metrics = build_topic_window_metrics(topic_mentions, window_entries)
    window_metrics_by_name = select_metric_groups(window_result, metric_group_names)
    return compute_topic_metric_report(
        topic_metrics=topic_metrics,
        window_metrics_by_name=window_metrics_by_name,
        min_topic_mentions=min_topic_mentions,
        min_windows=min_windows,
    )


def run_topic_modelling(
    use_existing: bool = True,
    window_multiple: int = 5,
    base_window_size: int = DEFAULT_WINDOW_SIZE,
    window_stride: Optional[int] = None,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    soft_score_threshold: Optional[float] = 0.5,
    soft_top_k_topics: Optional[int] = 3,
    use_pca: bool = True,
    pca_components: int = 50,
):
    """
    Batch topic modelling across all normalised, segmented text files.

    Defaults: window size 15 (DEFAULT_WINDOW_SIZE * 5) with stride 6 (DEFAULT_WINDOW_SIZE * 2)
    so topic windows align to the sentence-level metrics that use window size 3.
    Output shape: data/analytics/topic_modelling/<category>/<name>/<name>_topics.json
    """
    modeler = NeuralTopicModeler()
    normalised_root = text_path("processed", "normalised_segmented_texts")
    output_root = analytics_path("topic")
    output_root.mkdir(parents=True, exist_ok=True)
    stride = window_stride or (base_window_size * 2)

    for genre, author, subdir in iter_genre_author_dirs(normalised_root, GENRES):
        print(f"Processing category: {genre}/{author}")
        category_window_multiple = window_multiple
        category_min_cluster_size = min_cluster_size
        category_min_samples = min_samples

        out_subdir = output_root / genre / author
        out_subdir.mkdir(parents=True, exist_ok=True)

        for file in subdir.glob("*.jsonl"):
            base_name = file.stem.replace("_normalised_segmented", "")
            text_dir = out_subdir / base_name
            text_dir.mkdir(parents=True, exist_ok=True)
            clustered_output_file = text_dir / f"{base_name}_clustered_topics.json"
            debug_file = text_dir / f"{base_name}_topic_debug.json"
            if use_existing and clustered_output_file.exists():
                print(f"Skipping {file.name} (exists)")
                continue

            segmented_mentions = load_segmented_topic_mentions(file)

            book_overrides = TOPIC_BOOK_OVERRIDES.get(genre, {}).get(base_name, {})
            book_window_multiple = book_overrides.get("window_multiple", category_window_multiple)
            book_window_stride = book_overrides.get("window_stride", stride)
            book_min_cluster_size = book_overrides.get("min_cluster_size", category_min_cluster_size)
            book_min_samples = book_overrides.get("min_samples", category_min_samples)
            book_soft_score_threshold = book_overrides.get("soft_score_threshold", soft_score_threshold)
            book_soft_top_k_topics = book_overrides.get("soft_top_k_topics", soft_top_k_topics)
            book_use_pca = book_overrides.get("use_pca", use_pca)
            book_pca_components = book_overrides.get("pca_components", pca_components)
            effective_window_size = base_window_size * book_window_multiple

            print(f"Extracting topics for {file.name}...")
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
            num_sentences = len(segmented_mentions)
            result = {
                "meta": {
                    "filename": file.name,
                    "base_window_size": base_window_size,
                    "window_multiple": book_window_multiple,
                    "model_window_size": effective_window_size,
                    "window_stride": book_window_stride,
                    "num_sentences": num_sentences,
                    "min_cluster_size": book_min_cluster_size,
                    "min_samples": book_min_samples,
                    "soft_score_threshold": book_soft_score_threshold,
                    "soft_top_k_topics": book_soft_top_k_topics,
                    "use_pca": book_use_pca,
                    "pca_components": book_pca_components,
                },
                "topics": serialize_topic_results(topic_results),
                "windows": window_topics,
                "mentions_per_sentence": count_mentions_per_sentence(topic_results),
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

            print(f"Saved clustered topics to {clustered_output_file.name}")
            print(f"Saved topic debug to {debug_file.name}")

    print("All done.")


if __name__ == "__main__":
    run_topic_modelling()
