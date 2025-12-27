import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from x_configs import load_spacy_model
from .z_utils import processed_text_path
from .c1_syntactics import SyntaxAnalyzer
from .c2_lexico_semantics import LexicoSemanticsAnalyzer




def load_topics_json(corpus_file):
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


def _tokenize_words(text: str, lowercase: bool = True):
    """Lightweight tokenizer for lexical diversity (MATTR)."""
    tokens = re.findall(r"[A-Za-z0-9']+", text)
    if lowercase:
        tokens = [t.lower() for t in tokens]
    return tokens


def _moving_average_type_token_ratio(tokens, window_size: int = 50) -> float:
    """Compute Moving Average Type-Token Ratio (MATTR) over a sliding window."""
    tokens = [t for t in tokens if t]
    total_tokens = len(tokens)

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if total_tokens == 0:
        return 0.0
    if total_tokens < window_size:
        return round(len(set(tokens)) / total_tokens, 3)

    ttr_values = []
    for i in range(total_tokens - window_size + 1):
        window = tokens[i : i + window_size]
        ttr_values.append(len(set(window)) / window_size)

    return round(statistics.mean(ttr_values), 3)


def compute_mattr_metrics(text: str, window_size: int = 50, lowercase: bool = True):
    """Compute MATTR over the whole text for inclusion in window metrics."""
    words = _tokenize_words(text, lowercase=lowercase)
    mattr = _moving_average_type_token_ratio(words, window_size=window_size)
    return {
        "mattr_score": mattr,
        "window_size": min(window_size, len(words)),
        "total_tokens": len(words),
    }


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
            "cohesion_metrics",
            "semantic_role_metrics",
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

    fig, ax = plt.subplots(figsize=(max(8, len(metric_names) * 0.5), max(6, len(topic_labels) * 0.4)))
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



def _flatten_numeric_metrics(entry: Dict[str, object], prefix: str) -> Dict[str, float]:
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


def compute_topic_variance_report(
    topic_metrics: List[Dict[str, object]],
    window_metrics_by_name: Dict[str, List[Dict[str, object]]],
    min_topic_mentions: int = 1,
    min_windows: int = 2,
):
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

    fig, ax = plt.subplots(figsize=(max(8, len(metric_names) * 0.5), max(6, len(topic_labels) * 0.4)))
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




def run_windowed_metrics(window_size=3, mattr_window_size=50, use_existing=True):
    """
    Computes sentence/window-level metrics for all texts
    using precomputed corpus-level metrics.
    
    Reads from: processed_text_paths('corpus')
    Saves to:  processed_text_paths('window')
    """
    corpus_root = processed_text_path("corpus")
    output_root = processed_text_path("window")
    output_root.mkdir(parents=True, exist_ok=True)
    nlp = load_spacy_model()
    syntax_analyzer = SyntaxAnalyzer(nlp)

    for subdir in corpus_root.iterdir():
        if not subdir.is_dir():
            continue
        print(f"Processing category: {subdir.name}")
        out_subdir = output_root / subdir.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        for file in subdir.glob("*.json"):
            output_file = out_subdir / file.name
            if use_existing and output_file.exists():
                print(f"Skipping {file.name} (exists)")
                continue

            # Load precomputed corpus metrics
            data = json.load(file.open("r", encoding="utf-8"))
            text_content = data.get("text")  # make sure you saved raw text or chunks
            if not text_content:
                # Fallback: stitch chunk text if raw text was not stored
                chunks = data.get("chunks", [])
                text_content = " ".join(chunk.get("text", "") for chunk in chunks if isinstance(chunk, dict))
            mattr_metrics = compute_mattr_metrics(text_content or "", window_size=mattr_window_size)

            corpus_word_freqs = data.get("word_frequencies") or {w: f for w, f in data.get("top_words", [])}

            # Initialize analyzers
            lex_analyzer = LexicoSemanticsAnalyzer(nlp, corpus_freqs=corpus_word_freqs)
            doc = nlp(text_content or "")
            sentence_texts = [sent.text for sent in doc.sents]

            # ------------------------
            # Syntax metrics
            # ------------------------
            clause_metrics = syntax_analyzer.compute_clause_metrics(doc, window_size=window_size)
            clause_embed_metrics = syntax_analyzer.compute_clause_embedding_depth(doc, window_size=window_size)
            dep_complexity_metrics = syntax_analyzer.compute_dependency_complexity(doc, window_size=window_size)

            # ------------------------
            # Lexico-semantic metrics
            # ------------------------
            avg_word_freq_metrics = lex_analyzer.compute_avg_word_frequency(doc, window_size=window_size)
            lexical_density_metrics = lex_analyzer.analyze_lexical_density(doc, window_size=window_size)
            lexical_information_content = lex_analyzer.analyze_information_content(
                doc, word_frequencies=corpus_word_freqs, window_size=window_size
            )
            cohesion_metrics = lex_analyzer.analyze_cohesion(doc, window_size=window_size)
            semantic_role_metrics = lex_analyzer.analyze_semantic_roles(doc, window_size=window_size)
            
            # Use token log-probs if available
            log_probs_list = []
            for chunk in data.get("chunks", []):
                log_probs_list.append(chunk.get("log_probs", []))
            info_content_metrics = lex_analyzer.compute_information_content(
                log_probs_list,
                window_size=window_size,
                sentence_texts=sentence_texts,
            )

            semantic_structures = lex_analyzer.extract_semantic_structures(doc, window_size=window_size)

            topics_data = load_topics_json(file)
            topic_mentions = collect_topic_mentions(topics_data)
            topic_metrics = build_topic_window_metrics(topic_mentions, clause_metrics)

            # Combine into result
            result = {
                "filename": data["filename"],
                "model": data.get("model", ""),
                "clause_metrics": clause_metrics,
                "clause_embedding_metrics": clause_embed_metrics,
                "dependency_complexity_metrics": dep_complexity_metrics,
                "avg_word_freq_metrics": avg_word_freq_metrics,
                "lexical_density_metrics": lexical_density_metrics,
                "lexical_information_content": lexical_information_content,
                "cohesion_metrics": cohesion_metrics,
                "semantic_role_metrics": semantic_role_metrics,
                "information_content_metrics": info_content_metrics,
                "semantic_structures": semantic_structures,
                "topic_metrics": topic_metrics,
                "lexical_diversity": mattr_metrics,
            }

            # Save
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(f"✅ Saved windowed metrics for {file.name}")
    print("🎉 All done.")
