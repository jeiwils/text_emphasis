"""

Neural topic modeling for long-form text.

Input:
- a raw text string (document, chapter, article, etc.)

Output:
- list of TopicResult objects:
  - topic_id (int)
  - keywords (top TF-IDF terms for the cluster)
  - mentions (sentence spans w/ character offsets for localisation)


"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from .z_utils import processed_text_path, topic_modelling_path, sliding_windows
from x_configs import DEFAULT_WINDOW_SIZE, load_spacy_model


@dataclass
class TopicMention:
    sentence_index: int
    text: str
    start_char: int
    end_char: int
    end_sentence: Optional[int] = None
    window_index: Optional[int] = None


@dataclass
class TopicResult:
    topic_id: int
    keywords: List[str]
    mentions: List[TopicMention]


class NeuralTopicModeler:
    """
    Clusters sentence embeddings, extracts keywords, and returns
    localized mentions (sentence index + char offsets).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        language: str = "en_core_web_sm",
        stop_words: str = "english",
    ):
        self.encoder = SentenceTransformer(model_name)
        self.nlp = load_spacy_model(language)
        self.stop_words = stop_words

    def segment_sentences(self, text: str) -> List[TopicMention]:
        """Split text into sentences with char offsets."""
        doc = self.nlp(text)
        sentences = []
        for idx, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            sentences.append(
                TopicMention(
                    sentence_index=idx,
                    text=sent_text,
                    start_char=sent.start_char,
                    end_char=sent.end_char,
                    end_sentence=idx,
                    window_index=idx,
                )
            )
        return sentences

    def build_windows(
        self, sentences: List[TopicMention], window_size: int
    ) -> List[TopicMention]:
        """
        Build sliding windows (stride 1) of sentences.
        Each window is represented as a TopicMention with start/end sentence indices.
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        windows: List[TopicMention] = []
        for window_sents in sliding_windows(sentences, window_size):
            if not window_sents:
                continue
            start = window_sents[0]
            end = window_sents[-1]
            window_text = " ".join(s.text for s in window_sents)
            windows.append(
                TopicMention(
                    sentence_index=start.sentence_index,
                    end_sentence=end.sentence_index,
                    window_index=start.sentence_index,
                    text=window_text,
                    start_char=start.start_char,
                    end_char=end.end_char,
                )
            )
        return windows

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """Encode sentences into embeddings."""
        return self.encoder.encode(sentences)

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Cluster embeddings with HDBSCAN."""
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        return clusterer.fit_predict(embeddings)

    def _build_topic_keywords(
        self,
        cluster_docs: List[str],
        labels: List[int],
        top_n: int = 8,
        ngram_range: Tuple[int, int] = (1, 2),
    ) -> Dict[int, List[str]]:
        """Build TF-IDF keywords for each topic cluster."""
        vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            ngram_range=ngram_range,
        )
        tfidf = vectorizer.fit_transform(cluster_docs)
        feature_names = np.array(vectorizer.get_feature_names_out())

        keywords: Dict[int, List[str]] = {}
        for row_idx, label in enumerate(labels):
            row = tfidf[row_idx]
            if row.nnz == 0:
                keywords[label] = []
                continue
            scores = row.toarray().ravel()
            top_indices = scores.argsort()[::-1][:top_n]
            keywords[label] = feature_names[top_indices].tolist()
        return keywords

    def extract_topics(
        self,
        text: str,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        top_n: int = 8,
        ngram_range: Tuple[int, int] = (1, 2),
        window_multiple: int = 2,
        base_window_size: int = DEFAULT_WINDOW_SIZE,
    ) -> List[TopicResult]:
        """
        Main entrypoint: returns clustered topics + localized mentions.

        Sentences are grouped into sliding windows of size
        `base_window_size * window_multiple` with stride 1 to align with
        other windowed metrics (default base window size is 3).
        """
        sentences = self.segment_sentences(text)
        if not sentences:
            return []

        model_window_size = max(1, base_window_size * max(1, window_multiple))
        windows = self.build_windows(sentences, model_window_size)
        window_texts = [w.text for w in windows]

        if len(window_texts) < min_cluster_size:
            labels = np.zeros(len(window_texts), dtype=int)
        else:
            embeddings = self.embed_sentences(window_texts)
            labels = self.cluster_embeddings(
                embeddings,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )

        topic_docs: Dict[int, List[str]] = {}
        topic_mentions: Dict[int, List[TopicMention]] = {}
        for window, label in zip(windows, labels):
            if label == -1:
                continue
            topic_docs.setdefault(label, []).append(window.text)
            topic_mentions.setdefault(label, []).append(window)

        if not topic_docs:
            return []

        cluster_labels = sorted(topic_docs.keys())
        cluster_docs = [" ".join(topic_docs[label]) for label in cluster_labels]
        keywords = self._build_topic_keywords(
            cluster_docs,
            cluster_labels,
            top_n=top_n,
            ngram_range=ngram_range,
        )

        results = []
        for label in cluster_labels:
            results.append(
                TopicResult(
                    topic_id=label,
                    keywords=keywords.get(label, []),
                    mentions=topic_mentions.get(label, []),
                )
            )
        return results


def serialize_topic_results(topic_results: List[TopicResult]) -> List[Dict[str, Any]]:
    """Convert TopicResult objects to plain dicts for JSON output."""
    return [
        {
            "topic_id": result.topic_id,
            "keywords": result.keywords,
            "mentions": [
                {
                    "sentence_index": mention.sentence_index,
                    "end_sentence": mention.end_sentence,
                    "window_index": mention.window_index,
                    "start_char": mention.start_char,
                    "end_char": mention.end_char,
                    "text": mention.text,
                }
                for mention in result.mentions
            ],
        }
        for result in topic_results
    ]


def count_mentions_per_sentence(topic_results: List[TopicResult]) -> Dict[int, int]:
    """
    Count how many topic mentions overlap each sentence index.
    If a mention spans multiple sentences, every sentence in the span is counted.
    """
    counts: Dict[int, int] = {}
    for result in topic_results:
        for mention in result.mentions:
            start = mention.sentence_index
            end = mention.end_sentence if mention.end_sentence is not None else mention.sentence_index
            for idx in range(start, end + 1):
                counts[idx] = counts.get(idx, 0) + 1
    return counts


def load_topics_json(topic_file: Path):
    """Locate a topics JSON file and load it."""
    candidate_paths = [
        topic_file.with_name(f"{topic_file.stem}_topics.json"),
        topic_file.with_name(f"{topic_file.stem}.topics.json"),
        topic_file.with_name(f"{topic_file.stem}-topics.json"),
    ]
    candidate_paths.extend(sorted(topic_file.parent.glob(f"{topic_file.stem}*topic*.json")))

    seen = set()
    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


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
            mentions.append(
                {
                    "topic_id": topic_id,
                    "sentence_index": sentence_index,
                }
            )
    return mentions


def build_topic_window_metrics(topic_mentions, window_entries):
    """Aggregate topic mention counts for each sentence window."""
    metrics = []
    for window in window_entries:
        start_sentence = window.get("start_sentence", 0)
        end_sentence = window.get("end_sentence", 0)
        window_mentions = [
            mention
            for mention in topic_mentions
            if start_sentence <= mention["sentence_index"] <= end_sentence
        ]
        topic_counts = {}
        for mention in window_mentions:
            topic_id = mention["topic_id"]
            if topic_id is None:
                continue
            topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
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
                "unique_topic_count": len(topic_counts),
                "top_topic_ids": top_topic_ids,
                "topic_counts": topic_counts,
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


def compute_topic_metric_report_from_window_result(
    window_result: Dict[str, object],
    topics_data: Optional[object] = None,
    metric_group_names: Optional[List[str]] = None,
    min_topic_mentions: int = 1,
    min_windows: int = 2,
):
    """Compute a topic/metric report from a window result and topic model output."""
    topic_mentions = collect_topic_mentions(topics_data)
    window_entries = window_result.get("syntax", {}).get("windows", [])
    topic_metrics = build_topic_window_metrics(topic_mentions, window_entries)
    window_metrics_by_name = select_metric_groups(window_result, metric_group_names)
    return compute_topic_metric_report(
        topic_metrics=topic_metrics,
        window_metrics_by_name=window_metrics_by_name,
        min_topic_mentions=min_topic_mentions,
        min_windows=min_windows,
    )


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


def run_topic_modelling(
    use_existing: bool = True,
    window_multiple: int = 2, ############ REMOVE THIS AT SOME POINT??? 
    base_window_size: int = DEFAULT_WINDOW_SIZE,
):
    """Batch topic modelling across all normalised, segmented text files."""
    modeler = NeuralTopicModeler()
    normalised_root = processed_text_path("normalised_segmented")
    output_root = topic_modelling_path()
    output_root.mkdir(parents=True, exist_ok=True)

    for subdir in normalised_root.iterdir():
        if not subdir.is_dir():
            continue
        print(f"Processing category: {subdir.name}")

        out_subdir = output_root / subdir.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        for file in subdir.glob("*.jsonl"):
            output_file = out_subdir / f"{file.stem}_topics.json"
            if use_existing and output_file.exists():
                print(f"Skipping {file.name} (exists)")
                continue

            lines = file.read_text(encoding="utf-8").splitlines()
            sentences = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = entry.get("text") if isinstance(entry, dict) else None
                if text:
                    sentences.append(str(text))
            text = "\n".join(sentences)
            print(f"Extracting topics for {file.name}...")

            topic_results = modeler.extract_topics(
                text,
                window_multiple=window_multiple,
                base_window_size=base_window_size,
            )
            result = {
                "meta": {
                    "filename": file.name,
                    "base_window_size": base_window_size,
                    "window_multiple": window_multiple,
                    "model_window_size": base_window_size * max(1, window_multiple),
                    "num_sentences": len(modeler.segment_sentences(text)),
                },
                "topics": serialize_topic_results(topic_results),
                "mentions_per_sentence": count_mentions_per_sentence(topic_results),
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(f"Saved topic modelling to {output_file.name}")

    print("All done.")


if __name__ == "__main__":
    run_topic_modelling()
