"""
Dashboard metric computations and topic correlation reports.

Input (build_dashboard_row):
{
  "window_data": {
    "meta": {"filename": "book.txt", "window_size": 3, "num_sentences": 120},
    "syntax": {"windows": [...]},
    "lexico_semantics": {"windows": [...]},
    "discourse": {"windows": [...]},
    "log_prob": {"sentences": [...]}
  }
}

Output (build_dashboard_row):
{
  "filename": "book.txt",
  "window_size": 3,
  "num_sentences": 120,
  "metrics": [
    {
      "unexpectedness": [
        {"avg_token_surprisal": 2.31},
        {"max_token_surprisal": 3.02},
        {"surprisal_variance": 0.08}
      ]
    },
    {
      "lexical": [
        {"lexical_density": 0.61},
        {"lexical_diversity_mattr": 0.68},
        {"avg_word_freq": 12.0},
        {"normalized_freq": 0.76},
        {"information_content": 1.1}
      ]
    },
    {
      "structure": [
        {"mean_dependency_depth": 1.9},
        {"max_dependency_depth": 3.3},
        {"clause_density": 0.25},
        {"avg_dependents_per_head": 2.1},
        {"clause_ratios": {"subordination_ratio": 0.25, "coordination_ratio": 0.1}},
        {"avg_mean_dependency_distance": 1.5},
        {"avg_median_depth": 1.8},
        {"depth_skew": 0.2}
      ]
    },
    {
      "discourse": [
        {"explicit_connectives_per_token": 0.05},
        {"connective_counts_per_token": {"Temporal": 0.05, "Contingency": 0.0, "Comparison": 0.0, "Expansion": 0.0}},
        {"tense_shift": 0.1},
        {"entity_overlap_rate": 0.12}
      ]
    }
  ]
}

Input (build_topic_correlation_report / build_central_report):
{
  "window_data": {"syntax": {"windows": [...]}, "lexico_semantics": {"windows": [...]}, ...},
  "topics_data": {"topics": [...], "windows": [...]}
}

Output (build_topic_correlation_report):
{
  "window_count": 40,
  "metric_names": ["syntax.mean_depth", "lexico_semantics.lexical_density", "..."],
  "topics": {
    "0": {
      "keywords": ["term1", "..."],
      "windows_with_topic": 12,
      "windows_without_topic": 28,
      "correlations": {
        "syntax.mean_depth": {
          "pearson_r": 0.32,
          "p_value": 0.04,
          "pearson_r_binary": 0.28,
          "p_value_binary": 0.05,
          "n": 40,
          "mean_with_topic": 2.1,
          "mean_without_topic": 1.8
        }
      }
    }
  },
  "params": {
    "use_soft_topic_scores": true,
    "soft_score_threshold": 0.5,
    "soft_top_k": 3,
    "central_top_n": 5,
    "block_size": 5,
    "permutations": 2000
  }
}
"""

import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from c4_topic_modeling import (
    build_topic_window_metrics,
    collect_soft_topic_mentions,
    collect_topic_mentions,
)


def load_window_metrics(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict payload in {path}")
    return data


def _require_numbers(values: List[float], *, label: str) -> List[float]:
    if not values:
        raise ValueError(f"Expected non-empty numeric values for {label}")
    for value in values:
        if not isinstance(value, (int, float)):
            raise ValueError(f"Expected numeric value in {label}; got {type(value).__name__}")
        if isinstance(value, float) and math.isnan(value):
            raise ValueError(f"NaN encountered in {label}")
    return [float(v) for v in values]


def flatten_numeric_metrics(entry: Dict[str, object], prefix: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in entry.items():
        if key in {"start_sentence", "end_sentence"}:
            continue
        metric_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if not isinstance(sub_val, (int, float)):
                    raise ValueError(f"Non-numeric value for {metric_key}.{sub_key}")
                if isinstance(sub_val, float) and math.isnan(sub_val):
                    raise ValueError(f"NaN value for {metric_key}.{sub_key}")
                metrics[f"{metric_key}.{sub_key}"] = float(sub_val)
        elif isinstance(value, (int, float)):
            if isinstance(value, float) and math.isnan(value):
                raise ValueError(f"NaN value for {metric_key}")
            metrics[metric_key] = float(value)
        else:
            raise ValueError(f"Non-numeric value for {metric_key}")
    return metrics


def collect_window_tables(window_data: Dict[str, object]) -> List[Dict[str, float]]:
    """
    Flatten per-window metrics from the window analysis result so they can be
    correlated against topic presence.
    """
    metrics_by_name: Dict[str, List[Dict[str, object]]] = {}

    syntax_windows = window_data.get("syntax", {}).get("windows", [])
    if syntax_windows:
        metrics_by_name["syntax"] = syntax_windows

    lex_windows = window_data.get("lexico_semantics", {}).get("windows", [])
    if lex_windows:
        metrics_by_name["lexico_semantics"] = lex_windows

    discourse_windows = window_data.get("discourse", {}).get("windows", [])
    if discourse_windows:
        metrics_by_name["discourse"] = discourse_windows

    log_prob_windows = window_data.get("log_prob", {}).get("windows", [])
    if log_prob_windows:
        metrics_by_name["log_prob"] = log_prob_windows

    if not metrics_by_name:
        return []

    lengths = {name: len(entries) for name, entries in metrics_by_name.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Window metric lengths differ: {lengths}")
    window_count = next(iter(lengths.values()))
    if window_count <= 0:
        return []
    table: List[Dict[str, float]] = []
    for idx in range(window_count):
        row: Dict[str, float] = {}
        for name, entries in metrics_by_name.items():
            row.update(flatten_numeric_metrics(entries[idx], name))
        table.append(row)
    return table


def _is_length_metric(name: str) -> bool:
    if name == "syntax.avg_tokens_per_sentence":
        return False
    if "avg_tokens_per_sentence" in name:
        return False
    if "lexical_diversity_mattr.token_count" in name:
        return True
    if "lexical_diversity_mattr.window_token_span" in name:
        return True
    if name.endswith(".token_count") or name == "token_count":
        return True
    return False


def _is_excluded_dashboard_metric(name: str) -> bool:
    if "discourse.pronoun_ratio" in name:
        return True
    if "lexico_semantics.num_agents_" in name:
        return True
    if "lexico_semantics.num_patients_" in name:
        return True
    if "lexico_semantics.role_count_" in name:
        return True
    if "lexico_semantics.role_counts_per_token." in name:
        return True
    return False


def collect_window_topic_scores(
    topics_data: Dict[str, object],
    *,
    soft_score_threshold: Optional[float],
    soft_top_k: Optional[int],
    window_entries: List[Dict[str, object]],
) -> List[Dict[int, float]]:
    if not topics_data or not isinstance(topics_data, dict):
        raise ValueError("Expected topics_data dict for topic scores")

    windows = topics_data.get("windows") or []
    if not window_entries:
        return []

    topic_windows = []
    for window in windows:
        if window.get("is_noise"):
            continue
        try:
            start_sentence = int(window.get("start_sentence", 0))
            end_sentence = int(window.get("end_sentence", start_sentence))
        except (TypeError, ValueError):
            continue

        scores = window.get("topic_scores") or []
        items = []
        for entry in scores:
            if not isinstance(entry, dict):
                raise ValueError("Expected topic_scores entries to be dicts")
            topic_id = entry.get("topic_id")
            score = entry.get("score")
            if not isinstance(topic_id, int) or not isinstance(score, (int, float)):
                raise ValueError("Invalid topic_scores entry; expected int topic_id and float score")
            items.append((topic_id, float(score)))
        items.sort(key=lambda kv: kv[1], reverse=True)
        if soft_top_k is not None and soft_top_k > 0:
            items = items[:soft_top_k]
        if soft_score_threshold is not None:
            items = [(topic_id, score) for topic_id, score in items if score >= soft_score_threshold]
        if items:
            topic_windows.append((start_sentence, end_sentence, items))

    scores_by_window: List[Dict[int, float]] = []
    for window in window_entries:
        try:
            metric_start = int(window.get("start_sentence", 0))
            metric_end = int(window.get("end_sentence", metric_start))
        except (TypeError, ValueError):
            scores_by_window.append({})
            continue

        score_sums: Dict[int, float] = {}
        weight_sums: Dict[int, float] = {}
        for start_sentence, end_sentence, items in topic_windows:
            if end_sentence < metric_start or start_sentence > metric_end:
                continue
            overlap = min(metric_end, end_sentence) - max(metric_start, start_sentence) + 1
            if overlap <= 0:
                continue
            for topic_id, score in items:
                score_sums[topic_id] = score_sums.get(topic_id, 0.0) + score * overlap
                weight_sums[topic_id] = weight_sums.get(topic_id, 0.0) + overlap

        window_scores = {
            topic_id: score_sums[topic_id] / weight_sums[topic_id]
            for topic_id in score_sums
            if weight_sums.get(topic_id)
        }
        scores_by_window.append(window_scores)

    return scores_by_window


def pearson_correlation(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2 or len(y) < 2:
        return None
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if x_arr.std() == 0 or y_arr.std() == 0:
        return None
    corr = float(np.corrcoef(x_arr, y_arr)[0, 1])
    if math.isnan(corr):
        return None
    return corr


def block_permutation_p_value(
    x: List[float],
    y: List[float],
    *,
    block_size: int,
    permutations: int,
    rng: np.random.Generator,
) -> Optional[float]:
    if len(x) < 2 or len(y) < 2:
        return None
    if block_size <= 0 or permutations <= 0:
        return None
    observed = pearson_correlation(x, y)
    if observed is None:
        return None
    n = len(x)
    block_size = min(block_size, n)
    blocks = [x[i : i + block_size] for i in range(0, n, block_size)]
    if not blocks:
        return None
    counts = 0
    for _ in range(permutations):
        perm_blocks = rng.permutation(len(blocks))
        shuffled = [val for idx in perm_blocks for val in blocks[idx]]
        shuffled = shuffled[:n]
        corr = pearson_correlation(shuffled, y)
        if corr is None:
            continue
        if abs(corr) >= abs(observed):
            counts += 1
    return (counts + 1) / (permutations + 1)


def topic_keywords(topics_data: Dict[str, object]) -> Dict[int, List[str]]:
    topics = topics_data.get("topics", []) if isinstance(topics_data, dict) else []
    mapping: Dict[int, List[str]] = {}
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        topic_id = topic.get("topic_id")
        keywords = topic.get("keywords", [])
        if isinstance(topic_id, int):
            mapping[topic_id] = keywords if isinstance(keywords, list) else []
    return mapping


def central_topics(
    topics_data: Dict[str, object],
    top_n: int = 5,
) -> List[Dict[str, object]]:
    topics = topics_data.get("topics", []) if isinstance(topics_data, dict) else []
    metrics = ["coherence", "exclusivity", "prevalence", "persistence"]
    values: Dict[str, List[float]] = {metric: [] for metric in metrics}
    entries: List[Tuple[int, List[str], Dict[str, float]]] = []
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        topic_id = topic.get("topic_id")
        stats = topic.get("stats") or {}
        if not isinstance(topic_id, int) or not isinstance(stats, dict):
            continue
        keywords = topic.get("keywords", [])
        stats_clean: Dict[str, float] = {}
        for metric in metrics:
            val = stats.get(metric)
            if not isinstance(val, (int, float)):
                continue
            if isinstance(val, float) and math.isnan(val):
                continue
            stats_clean[metric] = float(val)
            values[metric].append(float(val))
        if stats_clean:
            entries.append((topic_id, keywords if isinstance(keywords, list) else [], stats_clean))

    if not entries:
        return []

    mins = {metric: min(values[metric]) for metric in metrics if values[metric]}
    maxs = {metric: max(values[metric]) for metric in metrics if values[metric]}

    def _norm(metric: str, val: float) -> float:
        lo = mins.get(metric)
        hi = maxs.get(metric)
        if lo is None or hi is None or hi == lo:
            return 0.0
        return (val - lo) / (hi - lo)

    ranked = []
    for topic_id, keywords, stats in entries:
        score = 0.0
        used = 0
        for metric, val in stats.items():
            score += _norm(metric, val)
            used += 1
        if used == 0:
            continue
        ranked.append(
            {
                "topic_id": topic_id,
                "score": score,
                "keywords": keywords,
                "stats": stats,
            }
        )

    ranked.sort(key=lambda row: row["score"], reverse=True)
    scores = [row["score"] for row in ranked]
    if scores:
        threshold = float(np.percentile(scores, 60))
        filtered = [row for row in ranked if row["score"] >= threshold]
        min_count = min(3, len(ranked))
        if len(filtered) < min_count:
            filtered = ranked[:min_count]
        ranked = filtered
    return ranked[:top_n]


def build_topic_correlation_report(
    window_data: Dict[str, object],
    topics_data: Dict[str, object],
    *,
    soft_score_threshold: Optional[float] = 0.5,
    soft_top_k: Optional[int] = 3,
    central_top_n: int = 5,
    topics_key: str = "topics",
    use_soft_scores: Optional[bool] = None,
    block_size: int = 5,
    permutations: int = 2000,
) -> Dict[str, object]:
    is_topic_report = topics_key == "topics"
    if use_soft_scores is None:
        use_soft_scores = is_topic_report
    if use_soft_scores:
        score_threshold = None if is_topic_report else soft_score_threshold
        score_top_k = None if is_topic_report else soft_top_k
    else:
        score_threshold = None
        score_top_k = None
    topic_mentions = (
        collect_soft_topic_mentions(
            topics_data,
            score_threshold=score_threshold,
            top_k=score_top_k,
        )
        if use_soft_scores
        else collect_topic_mentions(topics_data)
    )
    window_entries = window_data.get("syntax", {}).get("windows", [])
    if not topic_mentions or not window_entries:
        return {}

    topic_metrics = build_topic_window_metrics(topic_mentions, window_entries)
    window_table = collect_window_tables(window_data)
    if not window_table or len(window_table) != len(topic_metrics):
        return {}

    metric_names = sorted(window_table[0].keys()) if window_table else []
    metric_names = [
        name
        for name in metric_names
        if not any(token in name for token in ("sentence_index", "start_sentence", "end_sentence"))
    ]
    metric_names = [
        name for name in metric_names if not _is_length_metric(name) and not _is_excluded_dashboard_metric(name)
    ]
    topics = set()
    for entry in topic_metrics:
        topics.update(entry.get("topic_counts", {}).keys())

    central_ranked = central_topics(topics_data, top_n=central_top_n)
    central_topic_ids = {entry["topic_id"] for entry in central_ranked}
    topics = central_topic_ids

    keyword_map = topic_keywords(topics_data)
    topic_reports: Dict[int, Dict[str, object]] = {}

    rng = np.random.default_rng(42)

    topic_score_windows = (
        collect_window_topic_scores(
            topics_data,
            soft_score_threshold=score_threshold,
            soft_top_k=score_top_k,
            window_entries=window_entries,
        )
        if use_soft_scores
        else [{} for _ in range(len(window_table))]
    )

    for topic_id in sorted(topics):
        values_by_metric: Dict[str, List[Tuple[float, float]]] = {name: [] for name in metric_names}
        with_topic_count = 0
        without_topic_count = 0

        for idx, window_row in enumerate(window_table):
            if use_soft_scores:
                score = 0.0
                if idx < len(topic_score_windows):
                    score = topic_score_windows[idx].get(topic_id, 0.0)
                present = float(score)
                has_topic = score > 0
            else:
                topic_count = topic_metrics[idx].get("topic_counts", {}).get(topic_id, 0)
                present = 1.0 if topic_count >= 1 else 0.0
                has_topic = present >= 1
            with_topic_count += 1 if has_topic else 0
            without_topic_count += 0 if has_topic else 1
            for metric in metric_names:
                val = window_row.get(metric)
                if not isinstance(val, (int, float)):
                    raise ValueError(f"Non-numeric metric value for {metric}")
                if isinstance(val, float) and math.isnan(val):
                    raise ValueError(f"NaN metric value for {metric}")
                values_by_metric[metric].append((present, float(val)))

        metric_correlations: Dict[str, object] = {}
        for metric, pairs in values_by_metric.items():
            presences = [p for p, _ in pairs]
            values = [v for _, v in pairs]
            values_with = [v for p, v in pairs if p > 0]
            values_without = [v for p, v in pairs if p <= 0]
            corr = pearson_correlation(presences, values)
            if corr is None:
                continue
            p_value = block_permutation_p_value(
                presences,
                values,
                block_size=block_size,
                permutations=permutations,
                rng=rng,
            )
            binary_presences = [1.0 if p > 0 else 0.0 for p in presences]
            corr_binary = pearson_correlation(binary_presences, values)
            p_value_binary = (
                block_permutation_p_value(
                    binary_presences,
                    values,
                    block_size=block_size,
                    permutations=permutations,
                    rng=rng,
                )
                if corr_binary is not None
                else None
            )
            mean_with = (
                statistics.mean(_require_numbers(values_with, label=f"{metric}.with_topic"))
                if values_with
                else None
            )
            mean_without = (
                statistics.mean(_require_numbers(values_without, label=f"{metric}.without_topic"))
                if values_without
                else None
            )
            metric_correlations[metric] = {
                "pearson_r": corr,
                "p_value": p_value,
                "pearson_r_binary": corr_binary,
                "p_value_binary": p_value_binary,
                "n": len(values),
                "mean_with_topic": mean_with,
                "mean_without_topic": mean_without,
            }

        if metric_correlations:
            topic_reports[topic_id] = {
                "keywords": keyword_map.get(topic_id, []),
                "windows_with_topic": with_topic_count,
                "windows_without_topic": without_topic_count,
                "correlations": metric_correlations,
            }

    if not topic_reports:
        return {}

    return {
        "window_count": len(window_table),
        "metric_names": metric_names,
        topics_key: topic_reports,
        "params": {
            "use_soft_topic_scores": use_soft_scores,
            "soft_score_threshold": score_threshold,
            "soft_top_k": score_top_k,
            "central_top_n": central_top_n,
            "block_size": block_size,
            "permutations": permutations,
        },
    }


def build_central_report(
    window_data: Dict[str, object],
    topics_data: Dict[str, object],
    *,
    soft_score_threshold: Optional[float],
    soft_top_k: Optional[int],
    central_top_n: int,
    block_size: int,
    permutations: int,
) -> Dict[str, object]:
    central_ranked = central_topics(topics_data, top_n=central_top_n)
    central_topic_ids = {
        row["topic_id"] for row in central_ranked if isinstance(row.get("topic_id"), int)
    }

    weighted_report = build_topic_correlation_report(
        window_data,
        topics_data,
        soft_score_threshold=soft_score_threshold,
        soft_top_k=soft_top_k,
        central_top_n=central_top_n,
        topics_key="central_topics",
        use_soft_scores=True,
        block_size=block_size,
        permutations=permutations,
    )

    report: Dict[str, object] = weighted_report if weighted_report else {}
    if len(central_topic_ids) > 1:
        window_entries = window_data.get("syntax", {}).get("windows", [])
        topic_mentions = collect_topic_mentions(topics_data)
        topic_metrics = build_topic_window_metrics(topic_mentions, window_entries)
        window_table = collect_window_tables(window_data)
        if window_table and len(window_table) == len(topic_metrics):
            metric_names = sorted(window_table[0].keys()) if window_table else []
            metric_names = [
                name
                for name in metric_names
                if not any(token in name for token in ("sentence_index", "start_sentence", "end_sentence"))
            ]
            metric_names = [
                name
                for name in metric_names
                if not _is_length_metric(name) and not _is_excluded_dashboard_metric(name)
            ]
            presence_rows = []
            with_central = 0
            without_central = 0
            for idx, window_row in enumerate(window_table):
                topic_counts = topic_metrics[idx].get("topic_counts", {})
                has_central = any(topic_counts.get(topic_id, 0) >= 1 for topic_id in central_topic_ids)
                presence = 1.0 if has_central else 0.0
                with_central += 1 if has_central else 0
                without_central += 0 if has_central else 1
                presence_rows.append((presence, window_row))

            correlations: Dict[str, object] = {}
            rng = np.random.default_rng(42)
            for metric in metric_names:
                pairs = []
                for presence, row in presence_rows:
                    value = row.get(metric)
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"Non-numeric metric value for {metric}")
                    if isinstance(value, float) and math.isnan(value):
                        raise ValueError(f"NaN metric value for {metric}")
                    pairs.append((presence, float(value)))
                if not pairs:
                    continue
                presences = [p for p, _ in pairs]
                values = [v for _, v in pairs]
                corr = pearson_correlation(presences, values)
                if corr is None:
                    continue
                p_value = block_permutation_p_value(
                    presences,
                    values,
                    block_size=block_size,
                    permutations=permutations,
                    rng=rng,
                )
                values_with = [v for p, v in pairs if p > 0]
                values_without = [v for p, v in pairs if p <= 0]
                correlations[metric] = {
                    "pearson_r": corr,
                    "p_value": p_value,
                    "n": len(values),
                    "mean_with_topic": statistics.mean(_require_numbers(values_with, label=f"{metric}.with_topic")),
                    "mean_without_topic": statistics.mean(_require_numbers(values_without, label=f"{metric}.without_topic")),
                }

            report["central_topic_presence"] = {
                "windows_with_central_topic": with_central,
                "windows_without_central_topic": without_central,
                "correlations": correlations,
            }

    ordered_topics = []
    for row in central_ranked:
        topic_id = row.get("topic_id")
        if not isinstance(topic_id, int):
            continue
        correlation_entry = report.get("central_topics", {}).get(topic_id, {})
        ordered_topics.append(
            {
                "topic_id": topic_id,
                "score": row.get("score"),
                "keywords": row.get("keywords", []),
                "stats": row.get("stats", {}),
                "correlations": correlation_entry.get("correlations", {}),
                "windows_with_topic": correlation_entry.get("windows_with_topic"),
                "windows_without_topic": correlation_entry.get("windows_without_topic"),
            }
        )

    report["central_topics_ordered"] = ordered_topics
    return report


def build_dashboard_row(
    window_data: Dict[str, object],
    *,
    filename_fallback: Optional[str] = None,
) -> Dict[str, object]:
    meta = window_data.get("meta", {}) if isinstance(window_data, dict) else {}
    filename = meta.get("filename") or filename_fallback
    row = {
        "filename": filename,
        "window_size": meta.get("window_size"),
        "num_sentences": meta.get("num_sentences"),
    }
    discourse = window_data.get("discourse", {}) if isinstance(window_data, dict) else {}
    discourse_windows = discourse.get("windows", []) if isinstance(discourse, dict) else []
    overlap_rates = [
        window.get("entity_overlap_ratio")
        for window in discourse_windows
        if isinstance(window, dict)
    ]
    overlap_rates = _require_numbers(overlap_rates, label="entity_overlap_ratio")
    entity_overlap_rate = statistics.mean(overlap_rates)

    log_prob = window_data.get("log_prob", {}) if isinstance(window_data, dict) else {}
    sentences = log_prob.get("sentences", []) if isinstance(log_prob, dict) else []
    mean_surprisals = []
    variance_values = []
    for sent in sentences if isinstance(sentences, list) else []:
        if not isinstance(sent, dict):
            continue
        sent_surprisal = sent.get("sentence_surprisal_metrics", {})
        if not isinstance(sent_surprisal, dict):
            continue
        mean_surprisal = sent_surprisal.get("mean_surprisal")
        variance = sent_surprisal.get("surprisal_variance")
        num_tokens = sent_surprisal.get("num_tokens")
        if not isinstance(mean_surprisal, (int, float)):
            raise ValueError("Non-numeric mean_surprisal")
        if not isinstance(num_tokens, (int, float)):
            raise ValueError("Non-numeric num_tokens in sentence_surprisal_metrics")
        if num_tokens > 0:
            if isinstance(mean_surprisal, float) and math.isnan(mean_surprisal):
                raise ValueError("NaN mean_surprisal")
            mean_surprisals.append(float(mean_surprisal))
        if variance is not None:
            if not isinstance(variance, (int, float)):
                raise ValueError("Non-numeric surprisal_variance")
            if isinstance(variance, float) and math.isnan(variance):
                raise ValueError("NaN surprisal_variance")
            variance_values.append(float(variance))
    avg_token_surprisal = statistics.mean(_require_numbers(mean_surprisals, label="mean_surprisal"))
    max_token_surprisal = max(_require_numbers(mean_surprisals, label="mean_surprisal"))
    surprisal_variance = statistics.mean(_require_numbers(variance_values, label="surprisal_variance"))

    lexico = window_data.get("lexico_semantics", {}) if isinstance(window_data, dict) else {}
    lexico_windows = lexico.get("windows", []) if isinstance(lexico, dict) else []
    densities = []
    mattr_scores = []
    avg_word_freqs = []
    normalized_freqs = []
    information_contents = []
    for window in lexico_windows:
        if not isinstance(window, dict):
            continue
        density = window.get("lexical_density_per_token")
        if density is None:
            density = window.get("lexical_density")
        if density is not None:
            densities.append(density)
        mattr = window.get("lexical_diversity_mattr", {})
        if isinstance(mattr, dict):
            mattr_score = mattr.get("mattr_score")
            if mattr_score is not None:
                mattr_scores.append(mattr_score)
        avg_word_freq = window.get("avg_word_freq")
        if avg_word_freq is not None:
            avg_word_freqs.append(avg_word_freq)
        normalized_freq = window.get("normalized_freq")
        if normalized_freq is not None:
            normalized_freqs.append(normalized_freq)
        information_content = window.get("information_content")
        if information_content is not None:
            information_contents.append(information_content)
    lexical_density = statistics.mean(_require_numbers(densities, label="lexical_density"))
    lexical_diversity_mattr = statistics.mean(
        _require_numbers(mattr_scores, label="lexical_diversity_mattr")
    )
    avg_word_freq = statistics.mean(_require_numbers(avg_word_freqs, label="avg_word_freq"))
    normalized_freq = statistics.mean(_require_numbers(normalized_freqs, label="normalized_freq"))
    information_content = statistics.mean(
        _require_numbers(information_contents, label="information_content")
    )

    syntax = window_data.get("syntax", {}) if isinstance(window_data, dict) else {}
    syntax_windows = syntax.get("windows", []) if isinstance(syntax, dict) else []
    mean_depths = []
    max_depths = []
    clauses = []
    dependents = []
    subordination_ratios = []
    coordination_ratios = []
    avg_mean_dependency_distances = []
    avg_median_depths = []
    depth_skews = []
    for window in syntax_windows:
        if not isinstance(window, dict):
            continue
        mean_depths.append(window.get("mean_depth"))
        max_depths.append(window.get("max_depth"))
        clause_counts_per_token = window.get("clause_counts_per_token", {})
        if not isinstance(clause_counts_per_token, dict):
            raise ValueError("Expected clause_counts_per_token dict in syntax windows")
        clause_total = sum(_require_numbers(list(clause_counts_per_token.values()), label="clause_counts_per_token"))
        clauses.append(clause_total)
        dep_per_head = window.get("avg_dependents_per_head", {})
        if not isinstance(dep_per_head, dict):
            raise ValueError("Expected avg_dependents_per_head dict in syntax windows")
        dep_vals = _require_numbers(list(dep_per_head.values()), label="avg_dependents_per_head")
        dependents.append(statistics.mean(dep_vals))
        clause_ratios = window.get("clause_ratios", {})
        if isinstance(clause_ratios, dict):
            subordination_ratio = clause_ratios.get("subordination_ratio")
            if subordination_ratio is not None:
                subordination_ratios.append(subordination_ratio)
            coordination_ratio = clause_ratios.get("coordination_ratio")
            if coordination_ratio is not None:
                coordination_ratios.append(coordination_ratio)
        avg_mean_dependency_distance = window.get("avg_mean_dependency_distance")
        if avg_mean_dependency_distance is not None:
            avg_mean_dependency_distances.append(avg_mean_dependency_distance)
        avg_median_depth = window.get("avg_median_depth", window.get("median_depth"))
        if avg_median_depth is not None:
            avg_median_depths.append(avg_median_depth)
        depth_skew = window.get("avg_depth_skew", window.get("depth_skew"))
        if depth_skew is not None:
            depth_skews.append(depth_skew)
    mean_dependency_depth = statistics.mean(_require_numbers(mean_depths, label="mean_depth"))
    max_dependency_depth = max(_require_numbers(max_depths, label="max_depth"))
    clause_density = statistics.mean(_require_numbers(clauses, label="clause_counts_per_token"))
    avg_dependents_per_head = statistics.mean(_require_numbers(dependents, label="avg_dependents_per_head"))
    clause_ratios = {
        "subordination_ratio": statistics.mean(
            _require_numbers(subordination_ratios, label="subordination_ratio")
        ),
        "coordination_ratio": statistics.mean(
            _require_numbers(coordination_ratios, label="coordination_ratio")
        ),
    }
    avg_mean_dependency_distance = statistics.mean(
        _require_numbers(avg_mean_dependency_distances, label="avg_mean_dependency_distance")
    )
    avg_median_depth = statistics.mean(
        _require_numbers(avg_median_depths, label="avg_median_depth")
    )
    depth_skew = statistics.mean(_require_numbers(depth_skews, label="depth_skew"))

    explicit_connectives_per_token = []
    connective_counts_per_token: Dict[str, List[float]] = {}
    tense_shifts = []
    for window in discourse_windows:
        if not isinstance(window, dict):
            continue
        explicit_connectives = window.get("explicit_connectives_per_token")
        if explicit_connectives is not None:
            explicit_connectives_per_token.append(explicit_connectives)
        connective_counts = window.get("connective_counts_per_token", {})
        if isinstance(connective_counts, dict):
            for key, value in connective_counts.items():
                if isinstance(value, (int, float)):
                    connective_counts_per_token.setdefault(key, []).append(value)
        tense_shift = window.get("tense_shift")
        if tense_shift is not None:
            tense_shifts.append(tense_shift)
    explicit_connectives_per_token = statistics.mean(
        _require_numbers(explicit_connectives_per_token, label="explicit_connectives_per_token")
    )
    connective_counts_per_token = {
        key: statistics.mean(_require_numbers(values, label=f"connective_counts_per_token.{key}"))
        for key, values in connective_counts_per_token.items()
    }
    tense_shift = statistics.mean(_require_numbers(tense_shifts, label="tense_shift"))

    row["metrics"] = [
        {
            "unexpectedness": [
                {"avg_token_surprisal": avg_token_surprisal},
                {"max_token_surprisal": max_token_surprisal},
                {"surprisal_variance": surprisal_variance},
            ]
        },
        {
            "lexical": [
                {"lexical_density": lexical_density},
                {"lexical_diversity_mattr": lexical_diversity_mattr},
                {"avg_word_freq": avg_word_freq},
                {"normalized_freq": normalized_freq},
                {"information_content": information_content},
            ]
        },
        {
            "structure": [
                {"mean_dependency_depth": mean_dependency_depth},
                {"max_dependency_depth": max_dependency_depth},
                {"clause_density": clause_density},
                {"avg_dependents_per_head": avg_dependents_per_head},
                {"clause_ratios": clause_ratios},
                {"avg_mean_dependency_distance": avg_mean_dependency_distance},
                {"avg_median_depth": avg_median_depth},
                {"depth_skew": depth_skew},
            ]
        },
        {
            "discourse": [
                {"explicit_connectives_per_token": explicit_connectives_per_token},
                {"connective_counts_per_token": connective_counts_per_token},
                {"tense_shift": tense_shift},
                {"entity_overlap_rate": entity_overlap_rate},
            ]
        },
    ]
    return row
