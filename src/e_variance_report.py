import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_topics_json(corpus_file: Path):
    """Locate a topics JSON file adjacent to a corpus file and load it."""
    candidate_paths = [
        corpus_file.with_name(f"{corpus_file.stem}_topics.json"),
        corpus_file.with_name(f"{corpus_file.stem}.topics.json"),
        corpus_file.with_name(f"{corpus_file.stem}-topics.json"),
    ]
    candidate_paths.extend(sorted(corpus_file.parent.glob(f"{corpus_file.stem}*topic*.json")))

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


def compute_topic_variance_from_window_result(
    window_result: Dict[str, object],
    metric_group_names: Optional[List[str]] = None,
    min_topic_mentions: int = 1,
    min_windows: int = 2,
):
    """Compute topic-variance report directly from a saved window result."""
    topic_metrics = window_result.get("topic_metrics", [])
    window_metrics_by_name = select_metric_groups(window_result, metric_group_names)
    return compute_topic_variance_report(
        topic_metrics=topic_metrics,
        window_metrics_by_name=window_metrics_by_name,
        min_topic_mentions=min_topic_mentions,
        min_windows=min_windows,
    )


def compute_topic_variance_report(
    topic_metrics: List[Dict[str, object]],
    window_metrics_by_name: Dict[str, List[Dict[str, object]]],
    min_topic_mentions: int = 1,
    min_windows: int = 2,
):
    """Compute per-topic variance deltas across windowed metric values."""
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


def plot_topic_variance_heatmap(
    report: Dict[str, object],
    output_path,
    value_key: str = "variance_delta",
    min_windows: int = 2,
    top_n: Optional[int] = None,
):
    """Render a heatmap for a selected variance statistic across topics."""
    metric_names: List[str] = report.get("metric_names", [])
    topics: Dict[int, Dict[str, object]] = report.get("topics", {})

    sorted_topics = sorted(
        topics.items(),
        key=lambda item: item[1].get("window_count_with_topic", 0),
        reverse=True,
    )
    if top_n:
        sorted_topics = sorted_topics[:top_n]

    topic_labels = []
    values = []
    for topic_id, topic_entry in sorted_topics:
        if topic_entry.get("window_count_with_topic", 0) < min_windows:
            continue
        topic_labels.append(str(topic_id))
        row = []
        for metric in metric_names:
            metric_entry = topic_entry["metrics"].get(metric, {})
            value = metric_entry.get(value_key)
            row.append(np.nan if value is None else value)
        values.append(row)

    if not values:
        raise ValueError("No topic/metric data available to plot.")

    data = np.array(values, dtype=float)

    fig, ax = plt.subplots(
        figsize=(max(8, len(metric_names) * 0.5), max(6, len(topic_labels) * 0.4))
    )
    cmap = plt.cm.viridis
    cmap.set_bad(color="lightgrey")
    im = ax.imshow(data, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.set_yticks(range(len(topic_labels)))
    ax.set_yticklabels(topic_labels)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Topic ID")
    ax.set_title(f"Topic variance heatmap ({value_key})")
    fig.colorbar(im, ax=ax)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
