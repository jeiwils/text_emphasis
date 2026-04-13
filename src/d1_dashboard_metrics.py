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
          "p_value": 0.04
        }
      }
    }
  },
  "params": {
    "near_top_alpha": "<config>",
    "top_score_fraction": "<config>",
    "coherence_floor": "<config>",
    "exclusivity_floor": "<config>",
    "block_size": "<config>",
    "permutations": "<config>"
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
    CENTRALITY_METRICS,
    DASHBOARD_WINDOW_CONFIG,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_CENTRAL_PRESENCE_NORMALIZE,
    DEFAULT_CENTRAL_PRESENCE_P,
    DEFAULT_PRESENCE_K_REFERENCE,
    DEFAULT_CENTRAL_TOPIC_SELECTION_CONFIG,
    DEFAULT_DASHBOARD_PERMUTATIONS,
    DEFAULT_RNG_SEED,
    WINDOW_METRIC_DOMAINS,
)

from .c4_topic_modeling import (
    build_topic_window_metrics,
    collect_soft_topic_mentions,
)


def _window_metrics_base_name(path: Path) -> Optional[str]:
    name = path.name
    for domain in WINDOW_METRIC_DOMAINS:
        if name == f"window_metrics.{domain}.json":
            return ""
    return None


def load_window_metrics(path: Path) -> Dict[str, object]:
    base_name = _window_metrics_base_name(path)
    if base_name is None:
        raise ValueError(
            "Expected window metrics file named `window_metrics.<domain>.json`; "
            f"got {path}"
        )

    window_data: Dict[str, object] = {}
    base_meta: Optional[Dict[str, object]] = None
    for domain in WINDOW_METRIC_DOMAINS:
        domain_filename = f"window_metrics.{domain}.json"
        domain_path = path.with_name(domain_filename)
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


def collect_window_tables(
    window_data: Dict[str, object],
    *,
    include_rms_variance: bool = True,
) -> List[Dict[str, float]]:
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
    if include_rms_variance:
        _append_rms_z_metrics(table)
    return table


def _variance(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pvariance(values)


def _collect_domain_metric_series(
    window_data: Dict[str, object],
    *,
    domain: str,
) -> Tuple[Dict[str, List[float]], int]:
    domain_payload = window_data.get(domain, {}) if isinstance(window_data, dict) else {}
    windows = domain_payload.get("windows", []) if isinstance(domain_payload, dict) else []
    if not windows:
        return {}, 0
    filtered = _filter_dashboard_windows(windows, domain=domain)
    series: Dict[str, List[float]] = {}
    for window in filtered:
        if not isinstance(window, dict):
            continue
        flat = flatten_numeric_metrics(window, domain)
        for metric, value in flat.items():
            series.setdefault(metric, []).append(float(value))
    return series, len(filtered)


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


def _metric_matrix(
    window_table: List[Dict[str, float]],
    metric_names: List[str],
) -> np.ndarray:
    if not window_table or not metric_names:
        return np.empty((0, 0), dtype=float)
    matrix = np.empty((len(metric_names), len(window_table)), dtype=float)
    for metric_idx, metric in enumerate(metric_names):
        for row_idx, row in enumerate(window_table):
            value = row.get(metric)
            if not isinstance(value, (int, float)):
                raise ValueError(f"Non-numeric metric value for {metric}")
            if isinstance(value, float) and math.isnan(value):
                raise ValueError(f"NaN metric value for {metric}")
            matrix[metric_idx, row_idx] = float(value)
    return matrix


def _rms_z_series(
    window_table: List[Dict[str, float]],
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray], Dict[str, int]]:
    if not window_table:
        return {}, None, {}
    metric_names = dashboard_metric_names(window_table)
    if not metric_names:
        return {}, None, {}
    matrix = _metric_matrix(window_table, metric_names)
    if matrix.size == 0:
        return {}, None, {}

    means = np.mean(matrix, axis=1)
    stds = np.std(matrix, axis=1, ddof=0)
    valid = stds > 0
    window_count = matrix.shape[1]
    z_scores = np.zeros_like(matrix)
    if np.any(valid):
        z_scores[valid, :] = (matrix[valid, :] - means[valid, None]) / stds[valid, None]

    rms_by_domain: Dict[str, np.ndarray] = {}
    metric_counts: Dict[str, int] = {}
    for domain in WINDOW_METRIC_DOMAINS:
        domain_indices = [
            idx
            for idx, name in enumerate(metric_names)
            if name.startswith(f"{domain}.")
        ]
        if not domain_indices:
            continue
        valid_indices = [idx for idx in domain_indices if valid[idx]]
        if valid_indices:
            rms = np.sqrt(np.mean(np.square(z_scores[valid_indices, :]), axis=0))
            metric_counts[domain] = len(valid_indices)
        else:
            rms = np.zeros(window_count, dtype=float)
            metric_counts[domain] = 0
        rms_by_domain[domain] = rms

    overall_indices = [idx for idx, is_valid in enumerate(valid) if is_valid]
    if overall_indices:
        overall_rms = np.sqrt(
            np.mean(np.square(z_scores[overall_indices, :]), axis=0)
        )
    else:
        overall_rms = np.zeros(window_count, dtype=float)

    return rms_by_domain, overall_rms, metric_counts


def _append_rms_z_metrics(window_table: List[Dict[str, float]]) -> None:
    if not window_table:
        return
    if any(key.startswith("variance.") for key in window_table[0].keys()):
        return
    rms_by_domain, overall_rms, _metric_counts = _rms_z_series(window_table)
    if not rms_by_domain and overall_rms is None:
        return
    for idx, row in enumerate(window_table):
        for domain, series in rms_by_domain.items():
            row[f"variance.{domain}_rms_z"] = round(float(series[idx]), 6)
        if overall_rms is not None:
            row["variance.overall_rms_z"] = round(float(overall_rms[idx]), 6)


def _pearson_vector(x: np.ndarray, y_matrix: np.ndarray) -> np.ndarray:
    if y_matrix.size == 0:
        return np.array([], dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if x_arr.size < 2 or y_matrix.shape[1] < 2:
        return np.full(y_matrix.shape[0], np.nan, dtype=float)
    x_centered = x_arr - np.mean(x_arr)
    x_ss = np.sum(x_centered ** 2)
    if x_ss == 0:
        return np.full(y_matrix.shape[0], np.nan, dtype=float)
    y_centered = y_matrix - np.mean(y_matrix, axis=1, keepdims=True)
    y_ss = np.sum(y_centered ** 2, axis=1)
    denom = np.sqrt(x_ss * y_ss)
    with np.errstate(divide="ignore", invalid="ignore"):
        corrs = np.sum(y_centered * x_centered, axis=1) / denom
    corrs[~np.isfinite(corrs)] = np.nan
    return corrs


def _build_blocks(x: np.ndarray, block_size: int) -> List[np.ndarray]:
    n = len(x)
    if n == 0:
        return []
    block_size = min(block_size, n)
    return [x[i : i + block_size] for i in range(0, n, block_size)]


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

    topic_windows.sort(key=lambda item: (item[0], item[1]))

    scores_by_window: List[Dict[int, float]] = []
    topic_idx = 0
    for window in window_entries:
        try:
            metric_start = int(window.get("start_sentence", 0))
            metric_end = int(window.get("end_sentence", metric_start))
        except (TypeError, ValueError):
            scores_by_window.append({})
            continue

        while topic_idx < len(topic_windows) and topic_windows[topic_idx][1] < metric_start:
            topic_idx += 1

        score_sums: Dict[int, float] = {}
        weight_sums: Dict[int, float] = {}
        scan_idx = topic_idx
        while scan_idx < len(topic_windows):
            start_sentence, end_sentence, items = topic_windows[scan_idx]
            if start_sentence > metric_end:
                break
            overlap = min(metric_end, end_sentence) - max(metric_start, start_sentence) + 1
            if overlap > 0:
                for topic_id, score in items:
                    score_sums[topic_id] = score_sums.get(topic_id, 0.0) + score * overlap
                    weight_sums[topic_id] = weight_sums.get(topic_id, 0.0) + overlap
            scan_idx += 1

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
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2 or y_arr.size < 2:
        return None
    x_centered = x_arr - np.mean(x_arr)
    y_centered = y_arr - np.mean(y_arr)
    denom = math.sqrt(float(np.sum(x_centered ** 2) * np.sum(y_centered ** 2)))
    if denom == 0.0:
        return None
    corr = float(np.sum(x_centered * y_centered) / denom)
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
    observed: Optional[float] = None,
    blocks: Optional[List[np.ndarray]] = None,
) -> Optional[float]:
    if len(x) < 2 or len(y) < 2:
        return None
    if block_size <= 0 or permutations <= 0:
        return None
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2 or y_arr.size < 2:
        return None
    if observed is None:
        observed = pearson_correlation(x_arr, y_arr)
    if observed is None:
        return None
    n = x_arr.size
    block_size = min(block_size, n)
    if blocks is None:
        blocks = _build_blocks(x_arr, block_size)
    if not blocks:
        return None
    counts = 0
    for _ in range(permutations):
        perm_blocks = rng.permutation(len(blocks))
        shuffled = np.concatenate([blocks[idx] for idx in perm_blocks])[:n]
        corr = pearson_correlation(shuffled, y_arr)
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


def _collect_topic_score_lists(topics_data: Dict[str, object]) -> Dict[int, List[float]]:
    scores_by_topic: Dict[int, List[float]] = {}
    for window in _window_items(topics_data):
        if window.get("is_noise"):
            continue
        scores = window.get("topic_scores") or []
        for entry in scores:
            if not isinstance(entry, dict):
                continue
            topic_id = entry.get("topic_id")
            score = entry.get("score")
            if not isinstance(topic_id, int) or not isinstance(score, (int, float)):
                continue
            scores_by_topic.setdefault(topic_id, []).append(float(score))
    return scores_by_topic


def _top_fraction_mean(scores: List[float], *, fraction: float) -> float:
    if not scores:
        return 0.0
    if fraction <= 0 or fraction > 1:
        raise ValueError("top score fraction must be in (0, 1]")
    top_count = max(1, math.ceil(len(scores) * fraction))
    top_scores = sorted(scores)[-top_count:]
    return sum(top_scores) / len(top_scores)


def _collect_top_fraction_means(
    topics_data: Dict[str, object], *, fraction: float
) -> Dict[int, float]:
    if fraction <= 0 or fraction > 1:
        raise ValueError("top score fraction must be in (0, 1]")
    scores_by_topic = _collect_topic_score_lists(topics_data)
    return {
        topic_id: _top_fraction_mean(scores, fraction=fraction)
        for topic_id, scores in scores_by_topic.items()
    }


def _collect_centrality_rows(topics_data: Dict[str, object]) -> List[Dict[str, object]]:
    topics = _topic_items(topics_data)
    values: Dict[str, List[float]] = {metric: [] for metric in CENTRALITY_METRICS}
    entries: List[Dict[str, object]] = []
    top_fraction = DEFAULT_CENTRAL_TOPIC_SELECTION_CONFIG.top_score_fraction
    top_fraction_means = _collect_top_fraction_means(topics_data, fraction=top_fraction)
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
            if metric == "top10_mean":
                val = top_fraction_means.get(topic_id, 0.0)
            else:
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
        persistence_rank: Optional[float] = None
        top10_rank: Optional[float] = None
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
            if metric == "persistence":
                persistence_rank = rank
            elif metric == "top10_mean":
                top10_rank = rank
            else:
                available.append(rank)
        if persistence_rank is not None or top10_rank is not None:
            # Treat persistence/top10 as an OR: keep only the stronger signal.
            available.append(
                max(
                    rank
                    for rank in (persistence_rank, top10_rank)
                    if rank is not None
                )
            )
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
    *,
    near_top_alpha: Optional[float] = None,
    coherence_floor: Optional[float] = None,
    exclusivity_floor: Optional[float] = None,
) -> List[Dict[str, object]]:
    config = DEFAULT_CENTRAL_TOPIC_SELECTION_CONFIG
    if near_top_alpha is None:
        near_top_alpha = config.near_top_alpha
    if coherence_floor is None:
        coherence_floor = config.coherence_floor
    if exclusivity_floor is None:
        exclusivity_floor = config.exclusivity_floor

    ranked = _collect_centrality_rows(topics_data)
    if not ranked:
        return []

    if coherence_floor is not None or exclusivity_floor is not None:
        try:
            coherence_idx = CENTRALITY_METRICS.index("coherence")
            exclusivity_idx = CENTRALITY_METRICS.index("exclusivity")
        except ValueError:
            coherence_idx = None
            exclusivity_idx = None
        filtered_ranked = []
        for row in ranked:
            scores = row.get("percentile_score", [])
            if not isinstance(scores, list):
                continue
            if coherence_idx is not None:
                if coherence_idx >= len(scores):
                    continue
                coherence_score = scores[coherence_idx]
                if coherence_score is None:
                    continue
                if coherence_floor is not None and coherence_score < coherence_floor:
                    continue
            if exclusivity_idx is not None:
                if exclusivity_idx >= len(scores):
                    continue
                exclusivity_score = scores[exclusivity_idx]
                if exclusivity_score is None:
                    continue
                if exclusivity_floor is not None and exclusivity_score < exclusivity_floor:
                    continue
            filtered_ranked.append(row)
        ranked = filtered_ranked
        if not ranked:
            return []

    ranked.sort(key=lambda row: row["score"], reverse=True)
    scores = [row["score"] for row in ranked]
    if scores:
        max_score = max(scores)
        near_top = near_top_alpha * max_score
        filtered = [row for row in ranked if row["score"] >= near_top]
        ranked = filtered
    return ranked


def _presence_k_value(
    topics_data: Dict[str, object],
    *,
    default_count: int,
    central_ids: Optional[set[int]] = None,
) -> Tuple[int, str]:
    reference = str(DEFAULT_PRESENCE_K_REFERENCE or "self").lower()
    if reference == "central":
        if central_ids is None:
            central_ranked = central_topics(topics_data)
            central_ids = {
                row["topic_id"]
                for row in central_ranked
                if isinstance(row.get("topic_id"), int)
            }
        if central_ids:
            return len(central_ids), "central"
    return default_count, "self"


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
    if not metric_names:
        return {}
    metric_matrix = _metric_matrix(window_table, metric_names)
    topics = set()
    for entry in topic_metrics:
        topics.update(entry.get("topic_counts", {}).keys())

    central_ranked = central_topics(topics_data)
    central_topic_ids = {entry["topic_id"] for entry in central_ranked}
    if topics_key != "topics":
        topics = central_topic_ids

    keyword_map = topic_keywords(topics_data)
    topic_reports: Dict[int, Dict[str, object]] = {}

    rng = np.random.default_rng(DEFAULT_RNG_SEED)

    topic_score_windows = collect_window_topic_scores(
        topics_data,
        window_entries=window_entries,
    )
    window_count = len(window_table)

    for topic_id in sorted(topics):
        presence_scores = np.zeros(window_count, dtype=float)
        for idx, score_map in enumerate(topic_score_windows[:window_count]):
            presence_scores[idx] = score_map.get(topic_id, 0.0)
        corrs = _pearson_vector(presence_scores, metric_matrix)
        if corrs.size == 0:
            continue
        blocks = _build_blocks(presence_scores, block_size) if block_size > 0 else None
        metric_correlations: Dict[str, object] = {}
        for metric_idx, metric in enumerate(metric_names):
            corr = corrs[metric_idx]
            if not np.isfinite(corr):
                continue
            p_value = block_permutation_p_value(
                presence_scores,
                metric_matrix[metric_idx],
                block_size=block_size,
                permutations=permutations,
                rng=rng,
                observed=float(corr),
                blocks=blocks,
            )
            metric_correlations[metric] = {
                "pearson_r": float(corr),
                "p_value": p_value,
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
            "top_score_fraction": DEFAULT_CENTRAL_TOPIC_SELECTION_CONFIG.top_score_fraction,
            "coherence_floor": DEFAULT_CENTRAL_TOPIC_SELECTION_CONFIG.coherence_floor,
            "exclusivity_floor": DEFAULT_CENTRAL_TOPIC_SELECTION_CONFIG.exclusivity_floor,
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
    if not metric_names:
        return {}
    metric_matrix = _metric_matrix(window_table, metric_names)
    topic_score_windows = collect_window_topic_scores(
        topics_data,
        window_entries=window_entries,
    )
    if len(topic_score_windows) != len(window_table):
        return {}

    num_central = len(central_topic_ids)
    k_value, k_reference = _presence_k_value(
        topics_data,
        default_count=num_central,
        central_ids=central_topic_ids,
    )
    if k_value <= 0:
        return {}
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
                powered_sum /= k_value
            presence = powered_sum ** inv_p
        else:
            presence = 0.0
        presence_scores.append(presence)

    correlations: Dict[str, object] = {}
    rng = np.random.default_rng(DEFAULT_RNG_SEED)
    presence_array = np.asarray(presence_scores, dtype=float)
    corrs = _pearson_vector(presence_array, metric_matrix)
    blocks = _build_blocks(presence_array, block_size) if block_size > 0 else None
    for metric_idx, metric in enumerate(metric_names):
        corr = corrs[metric_idx]
        if not np.isfinite(corr):
            continue
        p_value = block_permutation_p_value(
            presence_array,
            metric_matrix[metric_idx],
            block_size=block_size,
            permutations=permutations,
            rng=rng,
            observed=float(corr),
            blocks=blocks,
        )
        correlations[metric] = {
            "pearson_r": float(corr),
            "p_value": p_value,
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
            "presence_k_reference": k_reference,
            "presence_k": k_value,
            "block_size": block_size,
            "permutations": permutations,
        },
    }


def build_topic_presence_report(
    window_data: Dict[str, object],
    topics_data: Dict[str, object],
    *,
    block_size: int,
    permutations: int,
) -> Dict[str, object]:
    topics = _topic_items(topics_data)
    topic_ids = {
        topic.get("topic_id")
        for topic in topics
        if isinstance(topic, dict) and isinstance(topic.get("topic_id"), int)
    }
    topic_ids = {topic_id for topic_id in topic_ids if isinstance(topic_id, int) and topic_id >= 0}
    if not topic_ids:
        return {}

    presence_p = float(DEFAULT_CENTRAL_PRESENCE_P)
    normalize_presence = bool(DEFAULT_CENTRAL_PRESENCE_NORMALIZE)
    if presence_p <= 0:
        raise ValueError("topic presence p must be > 0")

    window_entries = window_data.get("syntax", {}).get("windows", [])
    window_table = collect_window_tables(window_data)
    if not window_table:
        return {}

    metric_names = dashboard_metric_names(window_table)
    if not metric_names:
        return {}
    metric_matrix = _metric_matrix(window_table, metric_names)
    topic_score_windows = collect_window_topic_scores(
        topics_data,
        window_entries=window_entries,
    )
    if len(topic_score_windows) != len(window_table):
        return {}

    num_topics = len(topic_ids)
    k_value, k_reference = _presence_k_value(
        topics_data,
        default_count=num_topics,
    )
    if k_value <= 0:
        return {}
    inv_p = 1.0 / presence_p
    presence_scores: List[float] = []
    for score_map in topic_score_windows:
        powered_sum = 0.0
        for topic_id in topic_ids:
            score = score_map.get(topic_id, 0.0)
            if score <= 0.0:
                continue
            powered_sum += score ** presence_p
        if powered_sum > 0.0:
            if normalize_presence:
                powered_sum /= k_value
            presence = powered_sum ** inv_p
        else:
            presence = 0.0
        presence_scores.append(presence)

    correlations: Dict[str, object] = {}
    rng = np.random.default_rng(DEFAULT_RNG_SEED)
    presence_array = np.asarray(presence_scores, dtype=float)
    corrs = _pearson_vector(presence_array, metric_matrix)
    blocks = _build_blocks(presence_array, block_size) if block_size > 0 else None
    for metric_idx, metric in enumerate(metric_names):
        corr = corrs[metric_idx]
        if not np.isfinite(corr):
            continue
        p_value = block_permutation_p_value(
            presence_array,
            metric_matrix[metric_idx],
            block_size=block_size,
            permutations=permutations,
            rng=rng,
            observed=float(corr),
            blocks=blocks,
        )
        correlations[metric] = {
            "pearson_r": float(corr),
            "p_value": p_value,
        }

    return {
        "window_count": len(window_table),
        "metric_names": metric_names,
        "topic_presence": {
            "topic_presence_scores": presence_scores,
            "correlations": correlations,
        },
        "params": {
            "presence_aggregation": "p_norm",
            "presence_p": presence_p,
            "presence_normalized": normalize_presence,
            "presence_k_reference": k_reference,
            "presence_k": k_value,
            "block_size": block_size,
            "permutations": permutations,
        },
    }


def build_non_central_presence_report(
    window_data: Dict[str, object],
    topics_data: Dict[str, object],
    *,
    block_size: int,
    permutations: int,
) -> Dict[str, object]:
    topics = _topic_items(topics_data)
    topic_ids = {
        topic.get("topic_id")
        for topic in topics
        if isinstance(topic, dict) and isinstance(topic.get("topic_id"), int)
    }
    topic_ids = {topic_id for topic_id in topic_ids if isinstance(topic_id, int) and topic_id >= 0}
    if not topic_ids:
        return {}

    central_ranked = central_topics(topics_data)
    central_topic_ids = {
        row["topic_id"] for row in central_ranked if isinstance(row.get("topic_id"), int)
    }
    non_central_topic_ids = {
        topic_id for topic_id in topic_ids if topic_id not in central_topic_ids
    }
    if not non_central_topic_ids:
        return {}

    presence_p = float(DEFAULT_CENTRAL_PRESENCE_P)
    normalize_presence = bool(DEFAULT_CENTRAL_PRESENCE_NORMALIZE)
    if presence_p <= 0:
        raise ValueError("non-central topic presence p must be > 0")

    window_entries = window_data.get("syntax", {}).get("windows", [])
    window_table = collect_window_tables(window_data)
    if not window_table:
        return {}

    metric_names = dashboard_metric_names(window_table)
    if not metric_names:
        return {}
    metric_matrix = _metric_matrix(window_table, metric_names)
    topic_score_windows = collect_window_topic_scores(
        topics_data,
        window_entries=window_entries,
    )
    if len(topic_score_windows) != len(window_table):
        return {}

    num_topics = len(non_central_topic_ids)
    k_value, k_reference = _presence_k_value(
        topics_data,
        default_count=num_topics,
        central_ids=central_topic_ids,
    )
    if k_value <= 0:
        return {}
    inv_p = 1.0 / presence_p
    presence_scores: List[float] = []
    for score_map in topic_score_windows:
        powered_sum = 0.0
        for topic_id in non_central_topic_ids:
            score = score_map.get(topic_id, 0.0)
            if score <= 0.0:
                continue
            powered_sum += score ** presence_p
        if powered_sum > 0.0:
            if normalize_presence:
                powered_sum /= k_value
            presence = powered_sum ** inv_p
        else:
            presence = 0.0
        presence_scores.append(presence)

    correlations: Dict[str, object] = {}
    rng = np.random.default_rng(DEFAULT_RNG_SEED)
    presence_array = np.asarray(presence_scores, dtype=float)
    corrs = _pearson_vector(presence_array, metric_matrix)
    blocks = _build_blocks(presence_array, block_size) if block_size > 0 else None
    for metric_idx, metric in enumerate(metric_names):
        corr = corrs[metric_idx]
        if not np.isfinite(corr):
            continue
        p_value = block_permutation_p_value(
            presence_array,
            metric_matrix[metric_idx],
            block_size=block_size,
            permutations=permutations,
            rng=rng,
            observed=float(corr),
            blocks=blocks,
        )
        correlations[metric] = {
            "pearson_r": float(corr),
            "p_value": p_value,
        }

    return {
        "window_count": len(window_table),
        "metric_names": metric_names,
        "non_central_topic_presence": {
            "non_central_presence_scores": presence_scores,
            "correlations": correlations,
        },
        "params": {
            "presence_aggregation": "p_norm",
            "presence_p": presence_p,
            "presence_normalized": normalize_presence,
            "presence_k_reference": k_reference,
            "presence_k": k_value,
            "block_size": block_size,
            "permutations": permutations,
        },
    }


def build_window_variance_report(window_data: Dict[str, object]) -> Dict[str, object]:
    metric_variances: Dict[str, Dict[str, float]] = {}
    window_counts: Dict[str, int] = {}
    metric_counts: Dict[str, int] = {}

    for domain in WINDOW_METRIC_DOMAINS:
        series_by_metric, window_count = _collect_domain_metric_series(
            window_data,
            domain=domain,
        )
        if not series_by_metric:
            continue
        window_counts[domain] = window_count
        per_metric_var: Dict[str, float] = {}
        domain_variance_values: List[float] = []
        for metric in sorted(series_by_metric.keys()):
            variance = _variance(series_by_metric[metric])
            per_metric_var[metric] = round(variance, 6)
            domain_variance_values.append(variance)
        if per_metric_var:
            metric_variances[domain] = per_metric_var
            metric_counts[domain] = len(per_metric_var)

    if not metric_variances:
        return {}

    window_table = collect_window_tables(window_data, include_rms_variance=False)
    rms_by_domain, overall_rms, rms_metric_counts = _rms_z_series(window_table)
    rms_domain_means: Dict[str, float] = {}
    rms_domain_variances: Dict[str, float] = {}
    for domain, series in rms_by_domain.items():
        if len(series) == 0:
            continue
        rms_domain_means[domain] = round(float(np.mean(series)), 6)
        rms_domain_variances[domain] = round(float(np.var(series)), 6)

    overall_mean = None
    overall_variance = None
    if overall_rms is not None and len(overall_rms) > 0:
        overall_mean = round(float(np.mean(overall_rms)), 6)
        overall_variance = round(float(np.var(overall_rms)), 6)
    overall_metric_count = sum(rms_metric_counts.values())

    window_count = len(window_table) if window_table else None

    return {
        "window_count": window_count,
        "window_counts": window_counts,
        "metric_variances": metric_variances,
        "metric_counts": metric_counts,
        "rms_z": {
            "domain_means": rms_domain_means,
            "domain_variances": rms_domain_variances,
            "overall_mean": overall_mean,
            "overall_variance": overall_variance,
            "metric_counts": rms_metric_counts,
            "overall_metric_count": overall_metric_count,
        },
        "params": {
            "metric_variance_method": "pvariance",
            "rms_z_method": "rms_z",
            "rms_z_variance_method": "pvariance",
            "window_metric_filter": "DASHBOARD_WINDOW_CONFIG",
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
    normalized_structural_entropies = []
    parataxis_hypotaxis_ratios = []
    breath_unit_per_1000_totals = []
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
        normalized_structural_entropy = window.get("normalized_structural_entropy")
        if normalized_structural_entropy is not None:
            normalized_structural_entropies.append(normalized_structural_entropy)
        parataxis_payload = window.get("parataxis_hypotaxis", {})
        if isinstance(parataxis_payload, dict):
            parataxis_ratio = parataxis_payload.get("parataxis_to_hypotaxis_ratio")
            if parataxis_ratio is not None:
                parataxis_hypotaxis_ratios.append(parataxis_ratio)
        breath_unit_per_1000 = window.get("breath_unit_per_1000_words", {})
        if isinstance(breath_unit_per_1000, dict):
            breath_total = breath_unit_per_1000.get("total")
            if breath_total is not None:
                breath_unit_per_1000_totals.append(breath_total)
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
    normalized_structural_entropy = statistics.mean(
        _require_numbers(normalized_structural_entropies, label="normalized_structural_entropy")
    )
    parataxis_to_hypotaxis_ratio = statistics.mean(
        _require_numbers(parataxis_hypotaxis_ratios, label="parataxis_to_hypotaxis_ratio")
    )
    breath_unit_per_1000_words = statistics.mean(
        _require_numbers(breath_unit_per_1000_totals, label="breath_unit_per_1000_words")
    )
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
                {"normalized_structural_entropy": normalized_structural_entropy},
                {"parataxis_to_hypotaxis_ratio": parataxis_to_hypotaxis_ratio},
                {"breath_unit_per_1000_words": breath_unit_per_1000_words},
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
