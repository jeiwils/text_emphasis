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
        {"lexical_density_per_token": 0.61},
        {"lexical_diversity_mattr": 0.68},
        {"avg_word_freq": 12.0},
        {"normalized_freq": 0.76},
        {"information_content": 1.1}
      ]
    },
    {
      "structure": [
        {"max_dependency_depth": 3.3},
        {"clause_density": 0.25},
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

Input (build_topic_correlation_report / build_central_topic_correlations / build_central_presence_report):
{
  "window_data": {"syntax": {"windows": [...]}, "lexico_semantics": {"windows": [...]}, ...},
  "topics_data": {"topics": {"items": [...]}, "windows": {"items": [...]}, "meta": {...}, "params": {...}}
}

Output (build_topic_correlation_report):
{
  "total_topics": 12,
  "window_count": 40,
  "metric_names": ["syntax.median_depth", "lexico_semantics.lexical_density_per_token", "..."],
  "topics": {
    "0": {
      "keywords": ["term1", "..."],
      "correlations": {
        "syntax.median_depth": {
          "pearson_r": 0.32,
          "p_value": 0.04,
          "n_windows": 40
        }
      }
    }
  },
  "params": {
    "near_top_alpha": 0.85,
    "block_size": 5,
    "permutations": 1000
  }
}
"""

import json
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import bisect

from .x_configs import (
    DASHBOARD_WINDOW_CONFIG,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CENTRAL_PRESENCE_NORMALIZE,
    DEFAULT_CENTRAL_PRESENCE_P,
    DEFAULT_CENTRAL_TOPIC_SELECTION_CONFIG,
    DEFAULT_DASHBOARD_PERMUTATIONS,
)

from .b2_topic_modeling import (
    build_topic_window_metrics,
    collect_soft_topic_mentions,
)

# Dashboard correlations use a curated subset of window metrics; full windows remain in analytics.
CENTRALITY_METRICS = ("coherence", "exclusivity", "prevalence", "persistence")
WINDOW_METRIC_DOMAINS = ("syntax", "lexico_semantics", "discourse", "log_prob")


def _window_metrics_base_name(path: Path) -> Optional[str]:
    name = path.name
    for domain in WINDOW_METRIC_DOMAINS:
        suffix = f"_window_metrics.{domain}.json"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def load_window_metrics(path: Path) -> Dict[str, object]:
    base_name = _window_metrics_base_name(path)
    if base_name is None:
        raise ValueError(
            f"Expected window metrics file with suffix _window_metrics.<domain>.json; got {path}"
        )

    window_data: Dict[str, object] = {}
    base_meta: Optional[Dict[str, object]] = None
    for domain in WINDOW_METRIC_DOMAINS:
        domain_path = path.with_name(f"{base_name}_window_metrics.{domain}.json")
        if not domain_path.exists():
            raise FileNotFoundError(f"Missing window metrics file: {domain_path}")
        with open(domain_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict payload in {domain_path}")
        window_data[domain] = payload
        if base_meta is None:
            meta = payload.get("meta")
            base_meta = meta if isinstance(meta, dict) else {}

    window_data["meta"] = base_meta or {}
    return window_data


def _prune_window_metrics(
    metrics: dict,
    keep_keys: set,
    *,
    nested_keys: Optional[set] = None,
    nested_subkeys: Optional[Dict[str, set]] = None,
) -> dict:
    if not isinstance(metrics, dict):
        return {}
    pruned = {}
    for key in ("start_sentence", "end_sentence"):
        if key in metrics:
            pruned[key] = metrics.get(key)
    for key in keep_keys:
        value = metrics.get(key)
        if key in (nested_keys or set()) and isinstance(value, dict):
            if nested_subkeys and key in nested_subkeys:
                allowed = nested_subkeys[key]
                filtered = {k: value[k] for k in allowed if k in value}
                if filtered:
                    pruned[key] = filtered
            else:
                pruned[key] = value
        elif value is not None:
            pruned[key] = value
    return pruned


def _filter_dashboard_windows(
    windows: List[Dict[str, object]],
    *,
    domain: str,
) -> List[Dict[str, object]]:
    config = DASHBOARD_WINDOW_CONFIG.get(domain)
    if not config:
        return windows
    return [
        _prune_window_metrics(
            window,
            config.get("keep_keys", set()),
            nested_keys=config.get("nested_keys"),
            nested_subkeys=config.get("nested_subkeys"),
        )
        for window in windows
    ]


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
        metrics_by_name["syntax"] = _filter_dashboard_windows(syntax_windows, domain="syntax")

    lex_windows = window_data.get("lexico_semantics", {}).get("windows", [])
    if lex_windows:
        metrics_by_name["lexico_semantics"] = _filter_dashboard_windows(
            lex_windows,
            domain="lexico_semantics",
        )

    discourse_windows = window_data.get("discourse", {}).get("windows", [])
    if discourse_windows:
        metrics_by_name["discourse"] = _filter_dashboard_windows(
            discourse_windows,
            domain="discourse",
        )

    log_prob_windows = window_data.get("log_prob", {}).get("windows", [])
    if log_prob_windows:
        metrics_by_name["log_prob"] = _filter_dashboard_windows(log_prob_windows, domain="log_prob")

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


def dashboard_metric_names(window_table: List[Dict[str, float]]) -> List[str]:
    if not window_table:
        return []
    metric_names = sorted(window_table[0].keys())
    metric_names = [
        name
        for name in metric_names
        if not any(token in name for token in ("sentence_index", "start_sentence", "end_sentence"))
    ]
    return metric_names


def _topic_items(topics_data: Dict[str, object]) -> List[Dict[str, object]]:
    if not isinstance(topics_data, dict):
        return []
    topics_section = topics_data.get("topics")
    if not isinstance(topics_section, dict):
        return []
    items = topics_section.get("items")
    return items if isinstance(items, list) else []


def _window_items(topics_data: Dict[str, object]) -> List[Dict[str, object]]:
    if not isinstance(topics_data, dict):
        return []
    windows_section = topics_data.get("windows")
    if not isinstance(windows_section, dict):
        return []
    items = windows_section.get("items")
    return items if isinstance(items, list) else []


def collect_window_topic_scores(
    topics_data: Dict[str, object],
    *,
    window_entries: List[Dict[str, object]],
) -> List[Dict[int, float]]:
    if not topics_data or not isinstance(topics_data, dict):
        raise ValueError("Expected topics_data dict for topic scores")

    windows = _window_items(topics_data)
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
    topics = _topic_items(topics_data)
    mapping: Dict[int, List[str]] = {}
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        topic_id = topic.get("topic_id")
        keywords = topic.get("keywords", [])
        if isinstance(topic_id, int):
            mapping[topic_id] = keywords if isinstance(keywords, list) else []
    return mapping


def _collect_centrality_rows(topics_data: Dict[str, object]) -> List[Dict[str, object]]:
    topics = _topic_items(topics_data)
    values: Dict[str, List[float]] = {metric: [] for metric in CENTRALITY_METRICS}
    entries: List[Dict[str, object]] = []
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        topic_id = topic.get("topic_id")
        stats = topic.get("stats") or {}
        if not isinstance(topic_id, int) or not isinstance(stats, dict):
            continue
        keywords = topic.get("keywords", [])
        raw_stats: Dict[str, float] = {}
        for metric in CENTRALITY_METRICS:
            val = stats.get(metric)
            if not isinstance(val, (int, float)):
                continue
            if isinstance(val, float) and math.isnan(val):
                continue
            raw_stats[metric] = float(val)
            values[metric].append(float(val))
        if raw_stats:
            entries.append(
                {
                    "topic_id": topic_id,
                    "keywords": keywords if isinstance(keywords, list) else [],
                    "raw_stats": raw_stats,
                }
            )

    if not entries:
        return []

    sorted_values = {
        metric: sorted(values[metric]) for metric in CENTRALITY_METRICS if values[metric]
    }

    def _percentile_rank(metric: str, val: float) -> Optional[float]:
        buckets = sorted_values.get(metric)
        if not buckets:
            return None
        pos = bisect.bisect_right(buckets, val)
        return pos / len(buckets)

    rows: List[Dict[str, object]] = []
    for entry in entries:
        raw_stats = entry["raw_stats"]
        raw_score: List[Optional[float]] = []
        percentile_score: List[Optional[float]] = []
        available: List[float] = []
        for metric in CENTRALITY_METRICS:
            val = raw_stats.get(metric)
            raw_score.append(val if val is not None else None)
            if val is None:
                percentile_score.append(None)
                continue
            rank = _percentile_rank(metric, val)
            if rank is None:
                percentile_score.append(None)
                continue
            percentile_score.append(rank)
            available.append(rank)
        if not available:
            continue
        score = sum(available) / len(available)
        rows.append(
            {
                "topic_id": entry["topic_id"],
                "score": score,
                "keywords": entry["keywords"],
                "raw_score": raw_score,
                "percentile_score": percentile_score,
            }
        )

    return rows


def central_topics(
    topics_data: Dict[str, object],
    top_n: Optional[int] = None,
    *,
    near_top_alpha: Optional[float] = None,
) -> List[Dict[str, object]]:
    config = DEFAULT_CENTRAL_TOPIC_SELECTION_CONFIG
    if near_top_alpha is None:
        near_top_alpha = config.near_top_alpha
    if top_n is None:
        top_n = config.max_topics
    if top_n is not None and top_n <= 0:
        top_n = None

    ranked = _collect_centrality_rows(topics_data)
    if not ranked:
        return []

    ranked.sort(key=lambda row: row["score"], reverse=True)
    scores = [row["score"] for row in ranked]
    if scores:
        max_score = max(scores)
        near_top = near_top_alpha * max_score
        filtered = [row for row in ranked if row["score"] >= near_top]
        ranked = filtered
    if top_n is None or top_n <= 0:
        return ranked
    return ranked[:top_n]


def build_topic_correlation_report(
    window_data: Dict[str, object],
    topics_data: Dict[str, object],
    *,
    topics_key: str = "topics",
    block_size: int = DEFAULT_BLOCK_SIZE,
    permutations: int = DEFAULT_DASHBOARD_PERMUTATIONS,
) -> Dict[str, object]:
    topic_mentions = collect_soft_topic_mentions(topics_data)
    window_entries = window_data.get("syntax", {}).get("windows", [])
    if not topic_mentions or not window_entries:
        return {}

    topic_metrics = build_topic_window_metrics(topic_mentions, window_entries)
    window_table = collect_window_tables(window_data)
    if not window_table or len(window_table) != len(topic_metrics):
        return {}

    metric_names = dashboard_metric_names(window_table)
    topics = set()
    for entry in topic_metrics:
        topics.update(entry.get("topic_counts", {}).keys())

    central_ranked = central_topics(topics_data)
    central_topic_ids = {entry["topic_id"] for entry in central_ranked}
    if topics_key != "topics":
        topics = central_topic_ids

    keyword_map = topic_keywords(topics_data)
    topic_reports: Dict[int, Dict[str, object]] = {}

    rng = np.random.default_rng(42)

    topic_score_windows = collect_window_topic_scores(
        topics_data,
        window_entries=window_entries,
    )

    for topic_id in sorted(topics):
        values_by_metric: Dict[str, List[Tuple[float, float]]] = {name: [] for name in metric_names}

        for idx, window_row in enumerate(window_table):
            score = 0.0
            if idx < len(topic_score_windows):
                score = topic_score_windows[idx].get(topic_id, 0.0)
            present = float(score)
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
            metric_correlations[metric] = {
                "pearson_r": corr,
                "p_value": p_value,
                "n_windows": len(values),
            }

        if metric_correlations:
            topic_reports[topic_id] = {
                "keywords": keyword_map.get(topic_id, []),
                "correlations": metric_correlations,
            }

    if not topic_reports:
        return {}

    topics_list = _topic_items(topics_data)
    total_topics = len(topics_list) if isinstance(topics_list, list) else 0

    return {
        "total_topics": total_topics,
        "window_count": len(window_table),
        "metric_names": metric_names,
        topics_key: topic_reports,
        "params": {
            "near_top_alpha": DEFAULT_CENTRAL_TOPIC_SELECTION_CONFIG.near_top_alpha,
            "block_size": block_size,
            "permutations": permutations,
        },
    }


def build_central_topic_correlations(
    window_data: Dict[str, object],
    topics_data: Dict[str, object],
    *,
    block_size: int,
    permutations: int,
) -> Dict[str, object]:
    central_ranked = central_topics(topics_data)
    if not central_ranked:
        return {}

    weighted_report = build_topic_correlation_report(
        window_data,
        topics_data,
        topics_key="central_topics",
        block_size=block_size,
        permutations=permutations,
    )
    if not weighted_report:
        return {}

    window_count = weighted_report.get("window_count")
    metric_names = weighted_report.get("metric_names", [])
    params = weighted_report.get("params", {})

    central_topics_output = []
    for row in central_ranked:
        topic_id = row.get("topic_id")
        if not isinstance(topic_id, int):
            continue
        correlation_entry = weighted_report.get("central_topics", {}).get(topic_id, {})
        central_topics_output.append(
            {
                "topic_id": topic_id,
                "score": row.get("score"),
                "raw_score": row.get("raw_score", []),
                "percentile_score": row.get("percentile_score", []),
                "keywords": row.get("keywords", []),
                "correlations": correlation_entry.get("correlations", {}),
            }
        )

    return {
        "window_count": window_count,
        "metric_names": metric_names,
        "centrality_metrics_order": list(CENTRALITY_METRICS),
        "central_topics": central_topics_output,
        "params": params,
    }


def build_central_presence_report(
    window_data: Dict[str, object],
    topics_data: Dict[str, object],
    *,
    block_size: int,
    permutations: int,
) -> Dict[str, object]:
    central_ranked = central_topics(topics_data)
    central_topic_ids = {
        row["topic_id"] for row in central_ranked if isinstance(row.get("topic_id"), int)
    }
    if not central_topic_ids:
        return {}

    presence_p = float(DEFAULT_CENTRAL_PRESENCE_P)
    normalize_presence = bool(DEFAULT_CENTRAL_PRESENCE_NORMALIZE)
    if presence_p <= 0:
        raise ValueError("central presence p must be > 0")

    window_entries = window_data.get("syntax", {}).get("windows", [])
    window_table = collect_window_tables(window_data)
    if not window_table:
        return {}

    metric_names = dashboard_metric_names(window_table)
    topic_score_windows = collect_window_topic_scores(
        topics_data,
        window_entries=window_entries,
    )
    if len(topic_score_windows) != len(window_table):
        return {}

    num_central = len(central_topic_ids)
    inv_p = 1.0 / presence_p
    presence_scores: List[float] = []
    for score_map in topic_score_windows:
        powered_sum = 0.0
        for topic_id in central_topic_ids:
            score = score_map.get(topic_id, 0.0)
            if score <= 0.0:
                continue
            powered_sum += score ** presence_p
        if powered_sum > 0.0:
            if normalize_presence:
                powered_sum /= num_central
            presence = powered_sum ** inv_p
        else:
            presence = 0.0
        presence_scores.append(presence)

    presence_rows = []
    for presence_score, window_row in zip(presence_scores, window_table):
        # Soft presence uses a normalized p-norm across central topic scores.
        presence = float(presence_score)
        presence_rows.append((presence, window_row))

    if not presence_rows:
        return {}

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
        correlations[metric] = {
            "pearson_r": corr,
            "p_value": p_value,
            "n_windows": len(values),
        }

    return {
        "window_count": len(window_table),
        "metric_names": metric_names,
        "central_topic_presence": {
            "central_presence_scores": presence_scores,
            "correlations": correlations,
        },
        "params": {
            "presence_aggregation": "p_norm",
            "presence_p": presence_p,
            "presence_normalized": normalize_presence,
            "block_size": block_size,
            "permutations": permutations,
        },
    }


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
    token_counts = []
    token_surprisals = []
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
            token_counts.append(float(num_tokens))
            if variance is None:
                variance = 0.0
            if not isinstance(variance, (int, float)):
                raise ValueError("Non-numeric surprisal_variance")
            if isinstance(variance, float) and math.isnan(variance):
                raise ValueError("NaN surprisal_variance")
            variance_values.append(float(variance))
        log_probs = sent.get("sentence_log_probs", [])
        if isinstance(log_probs, list):
            for lp in log_probs:
                if isinstance(lp, (int, float)) and not (
                    isinstance(lp, float) and math.isnan(lp)
                ):
                    token_surprisals.append(-float(lp))
    mean_surprisals = _require_numbers(mean_surprisals, label="mean_surprisal")
    token_counts = _require_numbers(token_counts, label="num_tokens")
    total_tokens = sum(token_counts)
    avg_token_surprisal = (
        sum(mean * count for mean, count in zip(mean_surprisals, token_counts))
        / total_tokens
        if total_tokens
        else 0.0
    )
    max_token_surprisal = max(token_surprisals) if token_surprisals else 0.0
    pooled_variance = 0.0
    for mean, var, count in zip(mean_surprisals, variance_values, token_counts):
        pooled_variance += count * var
        pooled_variance += count * (mean - avg_token_surprisal) ** 2
    surprisal_variance = pooled_variance / total_tokens if total_tokens else 0.0

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
    lexical_density_per_token = statistics.mean(
        _require_numbers(densities, label="lexical_density_per_token")
    )
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
    max_depths = []
    clauses = []
    subordination_ratios = []
    coordination_ratios = []
    avg_mean_dependency_distances = []
    avg_median_depths = []
    depth_skews = []
    punctuation_per_tokens = []
    for window in syntax_windows:
        if not isinstance(window, dict):
            continue
        max_depths.append(window.get("max_depth"))
        clause_counts_per_token = window.get("clause_counts_per_token", {})
        if not isinstance(clause_counts_per_token, dict):
            raise ValueError("Expected clause_counts_per_token dict in syntax windows")
        clause_total = sum(_require_numbers(list(clause_counts_per_token.values()), label="clause_counts_per_token"))
        clauses.append(clause_total)
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
        punctuation_per_token = window.get("punctuation_per_token")
        if punctuation_per_token is not None:
            punctuation_per_tokens.append(punctuation_per_token)
    max_dependency_depth = max(_require_numbers(max_depths, label="max_depth"))
    clause_density = statistics.mean(_require_numbers(clauses, label="clause_counts_per_token"))
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
    punctuation_per_token = statistics.mean(
        _require_numbers(punctuation_per_tokens, label="punctuation_per_token")
    )

    explicit_connectives_per_token = []
    connective_counts_per_token: Dict[str, List[float]] = {}
    tense_shifts = []
    modality_per_tokens = []
    for window in discourse_windows:
        if not isinstance(window, dict):
            continue
        explicit_connectives = window.get("explicit_connectives_per_token")
        if explicit_connectives is not None:
            explicit_connectives_per_token.append(explicit_connectives)
        modality_per_token = window.get("modality_per_token")
        if modality_per_token is not None:
            modality_per_tokens.append(modality_per_token)
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
    modality_per_token = statistics.mean(_require_numbers(modality_per_tokens, label="modality_per_token"))

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
                {"lexical_density_per_token": lexical_density_per_token},
                {"lexical_diversity_mattr": lexical_diversity_mattr},
                {"avg_word_freq": avg_word_freq},
                {"normalized_freq": normalized_freq},
                {"information_content": information_content},
            ]
        },
        {
            "structure": [
                {"max_dependency_depth": max_dependency_depth},
                {"clause_density": clause_density},
                {"clause_ratios": clause_ratios},
                {"avg_mean_dependency_distance": avg_mean_dependency_distance},
                {"avg_median_depth": avg_median_depth},
                {"depth_skew": depth_skew},
                {"punctuation_per_token": punctuation_per_token},
            ]
        },
        {
            "discourse": [
                {"explicit_connectives_per_token": explicit_connectives_per_token},
                {"modality_per_token": modality_per_token},
                {"connective_counts_per_token": connective_counts_per_token},
                {"tense_shift": tense_shift},
                {"entity_overlap_rate": entity_overlap_rate},
            ]
        },
    ]
    return row
