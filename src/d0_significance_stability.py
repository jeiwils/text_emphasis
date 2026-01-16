"""
Significance and stability utilities for dashboard correlation outputs.

Outputs (under output_root):
- <output_root>/<genre>/00_genre_central_topic_presence_correlations.json
  {
    "genre": str,
    "text_count": int,
    "metric_names": [str],
    "metrics": {
      "<metric>": {
        "pearson_r": float,
        "p_value": float,
        "text_count": int,
        "total_windows": int,
        "total_effective_windows": int,
        "fisher_z": float,
        "fisher_z_weight_sum": float
      }
    },
    "texts": [
      {
        "category": str,
        "author": str,
        "text_name": str,
        "filename": str,
        "window_count": int|null,
        "effective_window_count": int|null,
        "block_size": int|null,
        "metric_count": int
      }
    ],
    "params": {
      "r_aggregation": "fisher_z",
      "p_value_aggregation": "stouffer",
      "r_weight": "n_eff-3",
      "p_weight": "sqrt(n_eff-3)",
      "n_eff_method": "ceil(n_windows / block_size), min 1",
      "block_size": int|[int]
    }
  }

- <output_root>/00_central_topic_split_half_stability.json
  {
    "text_count": int,
    "pair_count": int,
    "overall": {
      "sign_agreement_rate": float,
      "mean_abs_delta_r": float,
      "median_abs_delta_r": float
    },
    "metrics": {
      "<metric>": {
        "pair_count": int,
        "sign_agreement_rate": float,
        "mean_abs_delta_r": float,
        "median_abs_delta_r": float
      }
    },
    "texts": [
      {
        "category": str,
        "text_name": str,
        "filename": str,
        "pair_count": int,
        "sign_agreement_rate": float,
        "mean_abs_delta_r": float,
        "median_abs_delta_r": float,
        "soft_score_threshold": float|null,
        "soft_top_k": int|null
      }
    ],
    "params": {
      "split_method": "odd_even",
      "soft_score_threshold": [float],
      "soft_top_k": [int]
    }
  }

- <output_root>/<genre>/00_genre_central_topic_split_half_stability.json
  {
    "genre": str,
    "text_count": int,
    "pair_count": int,
    "overall": {
      "sign_agreement_rate": float,
      "mean_abs_delta_r": float,
      "median_abs_delta_r": float
    },
    "metrics": {
      "<metric>": {
        "pair_count": int,
        "sign_agreement_rate": float,
        "mean_abs_delta_r": float,
        "median_abs_delta_r": float
      }
    },
    "texts": [ ... ],
    "params": {
      "split_method": "odd_even",
      "soft_score_threshold": [float],
      "soft_top_k": [int]
    }
  }

- <output_root>/<genre>/<author>/<author>_central_topic_split_half_stability.json
  {
    "genre": str,
    "author": str,
    "text_count": int,
    "pair_count": int,
    "overall": {
      "sign_agreement_rate": float,
      "mean_abs_delta_r": float,
      "median_abs_delta_r": float
    },
    "metrics": {
      "<metric>": {
        "pair_count": int,
        "sign_agreement_rate": float,
        "mean_abs_delta_r": float,
        "median_abs_delta_r": float
      }
    },
    "texts": [ ... ],
    "params": {
      "split_method": "odd_even",
      "soft_score_threshold": [float],
      "soft_top_k": [int]
    }
  }

- data/results/00_cross_block_consistency.json
  {
    "block_labels": [str],
    "pair_count": int,
    "overall": {
      "sign_agreement_rate": float,
      "mean_abs_delta_r": float,
      "median_abs_delta_r": float
    },
    "metrics": {
      "<metric>": {
        "pair_count": int,
        "sign_agreement_rate": float,
        "mean_abs_delta_r": float,
        "median_abs_delta_r": float
      }
    },
    "params": {
      "block_labels": [str],
      "output_roots": [str]
    }
  }
"""

import json
import math
import statistics
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .d1_dashboard_metrics import (
    collect_window_tables,
    collect_window_topic_scores,
    load_window_metrics,
)
from .z_utils import find_topic_file

GENRE_CENTRAL_PRESENCE_FILENAME = "00_genre_central_topic_presence_correlations.json"
CENTRAL_TOPIC_SPLIT_HALF_FILENAME = "00_central_topic_split_half_stability.json"
GENRE_CENTRAL_TOPIC_SPLIT_HALF_FILENAME = "00_genre_central_topic_split_half_stability.json"
AUTHOR_CENTRAL_TOPIC_SPLIT_HALF_TEMPLATE = "{author}_central_topic_split_half_stability.json"
CROSS_BLOCK_CONSISTENCY_FILENAME = "00_cross_block_consistency.json"


def _as_float(value: object) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def _as_count(value: object) -> Optional[int]:
    numeric = _as_float(value)
    if numeric is None:
        return None
    count = int(numeric)
    if count <= 0:
        return None
    return count


def _fisher_z(value: float) -> Optional[float]:
    if value <= -1.0 or value >= 1.0:
        value = max(min(value, 0.999999), -0.999999)
    return math.atanh(value)


def _stouffer_p_value(rows: List[Tuple[float, float, int]]) -> Optional[float]:
    if not rows:
        return None
    normal = statistics.NormalDist()
    weighted = []
    for r_value, p_value, n_value in rows:
        if n_value <= 3:
            continue
        p_value = max(min(p_value, 1.0 - 1e-15), 1e-15)
        z_value = normal.inv_cdf(1.0 - p_value / 2.0)
        if r_value < 0:
            z_value = -z_value
        weight = math.sqrt(n_value - 3)
        weighted.append((weight, z_value))
    if not weighted:
        return None
    denom = math.sqrt(sum(weight ** 2 for weight, _ in weighted))
    if denom <= 0:
        return None
    z_combined = sum(weight * z_value for weight, z_value in weighted) / denom
    return 2.0 * (1.0 - normal.cdf(abs(z_combined)))


def _effective_n(window_count: Optional[int], block_size: Optional[int]) -> Optional[int]:
    if window_count is None:
        return None
    if not isinstance(block_size, int) or block_size <= 0:
        return window_count
    blocks = (window_count + block_size - 1) // block_size
    return max(1, blocks)


def _group_by_genre(entries: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for entry in entries:
        genre = entry.get("genre")
        if not isinstance(genre, str):
            continue
        grouped.setdefault(genre, []).append(entry)
    return grouped


def _extract_central_topic_payload(
    entry: Dict[str, object],
) -> Optional[Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]]:
    if not isinstance(entry, dict):
        return None
    metadata = entry.get("metadata")
    report = entry.get("report")
    params = entry.get("params", {})
    if not isinstance(metadata, dict) or not isinstance(report, dict):
        return None
    if not isinstance(params, dict):
        params = {}
    return metadata, report, params


def _parse_category(category: object) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(category, str):
        return None, None
    if "/" not in category:
        return None, None
    genre, author = category.split("/", 1)
    if not genre or not author:
        return None, None
    return genre, author


def _aggregate_central_presence_correlations(
    entries: List[Dict[str, object]],
) -> Dict[str, object]:
    metrics: Dict[str, List[Tuple[float, float, int, int]]] = {}
    texts: List[Dict[str, object]] = []
    for entry in entries:
        presence_report = entry.get("presence_report", {})
        if not isinstance(presence_report, dict):
            continue
        report = presence_report.get("report")
        if not isinstance(report, dict):
            continue
        params = presence_report.get("params", {})
        if not isinstance(params, dict):
            params = {}
        block_size = _as_count(params.get("block_size"))
        presence = report.get("central_topic_presence", {})
        if not isinstance(presence, dict):
            continue
        correlations = presence.get("correlations", {})
        if not isinstance(correlations, dict) or not correlations:
            continue

        window_count = _as_count(report.get("window_count"))
        if window_count is None:
            for corr_entry in correlations.values():
                if isinstance(corr_entry, dict):
                    window_count = _as_count(
                        corr_entry.get("n_windows")
                        if corr_entry.get("n_windows") is not None
                        else corr_entry.get("n")
                    )
                    if window_count is not None:
                        break
        effective_window_count = _effective_n(window_count, block_size)
        texts.append(
            {
                "category": entry.get("category"),
                "author": entry.get("author"),
                "text_name": entry.get("text_name"),
                "filename": entry.get("filename"),
                "window_count": window_count,
                "effective_window_count": effective_window_count,
                "block_size": block_size,
                "metric_count": len(correlations),
            }
        )

        for metric, corr_entry in correlations.items():
            if not isinstance(corr_entry, dict):
                continue
            r_value = _as_float(corr_entry.get("pearson_r"))
            p_value = _as_float(corr_entry.get("p_value"))
            n_value = _as_count(
                corr_entry.get("n_windows")
                if corr_entry.get("n_windows") is not None
                else corr_entry.get("n")
            )
            n_eff = _effective_n(n_value, block_size)
            if r_value is None or p_value is None or n_value is None or n_eff is None:
                continue
            metrics.setdefault(metric, []).append((r_value, p_value, n_eff, n_value))

    aggregated: Dict[str, object] = {}
    for metric, rows in metrics.items():
        valid_rows = [(r, p, n_eff, n_raw) for r, p, n_eff, n_raw in rows if n_eff > 3]
        if not valid_rows:
            continue
        weights = [(n_eff - 3) for _, _, n_eff, _ in valid_rows]
        z_values = [_fisher_z(r) for r, _, _, _ in valid_rows]
        if any(z is None for z in z_values):
            continue
        weight_sum = sum(weights)
        if weight_sum <= 0:
            continue
        z_bar = sum(z * w for z, w in zip(z_values, weights)) / weight_sum
        r_bar = math.tanh(z_bar)
        p_combined = _stouffer_p_value([(r, p, n_eff) for r, p, n_eff, _ in valid_rows])
        aggregated[metric] = {
            "pearson_r": r_bar,
            "p_value": p_combined,
            "text_count": len(valid_rows),
            "total_windows": sum(n_raw for _, _, _, n_raw in valid_rows),
            "total_effective_windows": sum(n_eff for _, _, n_eff, _ in valid_rows),
            "fisher_z": z_bar,
            "fisher_z_weight_sum": weight_sum,
        }

    return {
        "metric_names": sorted(aggregated.keys()),
        "metrics": aggregated,
        "texts": sorted(
            texts,
            key=lambda row: (row.get("category", ""), row.get("text_name", "")),
        ),
    }


def _write_genre_central_presence_correlations(
    entries: List[Dict[str, object]],
    *,
    use_existing: bool,
    output_root: Path,
) -> None:
    if not entries:
        return
    for genre, genre_entries in _group_by_genre(entries).items():
        output_path = output_root / genre / GENRE_CENTRAL_PRESENCE_FILENAME
        if use_existing and output_path.exists():
            continue
        summary = _aggregate_central_presence_correlations(genre_entries)
        block_sizes = sorted(
            {
                entry.get("presence_report", {}).get("params", {}).get("block_size")
                for entry in genre_entries
                if isinstance(entry.get("presence_report"), dict)
            }
        )
        block_sizes = [size for size in block_sizes if isinstance(size, int)]
        block_size_param: object = block_sizes[0] if len(block_sizes) == 1 else block_sizes
        payload = {
            "genre": genre,
            "text_count": len(summary.get("texts", [])),
            "metric_names": summary.get("metric_names", []),
            "metrics": summary.get("metrics", {}),
            "texts": summary.get("texts", []),
            "params": {
                "r_aggregation": "fisher_z",
                "p_value_aggregation": "stouffer",
                "r_weight": "n_eff-3",
                "p_weight": "sqrt(n_eff-3)",
                "n_eff_method": "ceil(n_windows / block_size), min 1",
                "block_size": block_size_param,
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def _is_number(value: object) -> bool:
    if not isinstance(value, (int, float)):
        return False
    return not (isinstance(value, float) and math.isnan(value))


def _pearson(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2 or len(y) < 2:
        return None
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    num = 0.0
    denom_x = 0.0
    denom_y = 0.0
    for x_val, y_val in zip(x, y):
        dx = x_val - mean_x
        dy = y_val - mean_y
        num += dx * dy
        denom_x += dx * dx
        denom_y += dy * dy
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return num / math.sqrt(denom_x * denom_y)


def _build_central_topic_split_half_entry(
    window_metrics_path: Path,
    central_entry: Dict[str, object],
    *,
    soft_score_threshold: Optional[float],
    soft_top_k: Optional[int],
) -> Optional[Dict[str, object]]:
    extracted = _extract_central_topic_payload(central_entry)
    if extracted is None:
        return None
    metadata, report, params = extracted
    if not isinstance(metadata, dict) or not isinstance(report, dict):
        return None

    topic_file = metadata.get("topic_file")
    if not isinstance(topic_file, str):
        topic_file = str(find_topic_file(window_metrics_path) or "")
    if not topic_file:
        return None
    topic_path = Path(topic_file)
    if not topic_path.exists():
        topic_path = Path(topic_file.replace("\\", "/"))
    if not topic_path.exists():
        return None

    metric_names = report.get("metric_names")
    if not isinstance(metric_names, list) or not metric_names:
        return None

    central_topics = report.get("central_topics", [])
    central_ids = [
        topic.get("topic_id")
        for topic in central_topics
        if isinstance(topic, dict) and isinstance(topic.get("topic_id"), int)
    ]
    if not central_ids:
        return None

    score_threshold = params.get("soft_score_threshold", soft_score_threshold)
    score_top_k = params.get("soft_top_k", soft_top_k)

    window_data = load_window_metrics(window_metrics_path)
    topics_data = load_window_metrics(topic_path)
    window_entries = window_data.get("syntax", {}).get("windows", []) if isinstance(window_data, dict) else []
    window_table = collect_window_tables(window_data)
    if not window_entries or not window_table:
        return None
    if len(window_entries) != len(window_table):
        return None

    score_windows = collect_window_topic_scores(
        topics_data,
        soft_score_threshold=score_threshold,
        soft_top_k=score_top_k,
        window_entries=window_entries,
    )
    if not score_windows or len(score_windows) != len(window_table):
        return None

    metric_series_map: Dict[str, List[float]] = {}
    for metric in metric_names:
        series = [row.get(metric) for row in window_table]
        if not series or any(not _is_number(value) for value in series):
            continue
        metric_series_map[metric] = [float(value) for value in series]

    if not metric_series_map:
        return None

    odd_idx = [idx for idx in range(len(window_table)) if idx % 2 == 1]
    even_idx = [idx for idx in range(len(window_table)) if idx % 2 == 0]
    if len(odd_idx) < 2 or len(even_idx) < 2:
        return None

    overall_signs: List[int] = []
    overall_deltas: List[float] = []
    metric_pairs: Dict[str, Dict[str, List[float]]] = {
        metric: {"signs": [], "deltas": []} for metric in metric_series_map
    }

    for topic_id in central_ids:
        topic_series = [score_map.get(topic_id, 0.0) for score_map in score_windows]
        if any(not _is_number(value) for value in topic_series):
            continue
        topic_series = [float(value) for value in topic_series]
        topic_odd = [topic_series[idx] for idx in odd_idx]
        topic_even = [topic_series[idx] for idx in even_idx]
        for metric, metric_series in metric_series_map.items():
            metric_odd = [metric_series[idx] for idx in odd_idx]
            metric_even = [metric_series[idx] for idx in even_idx]
            r_odd = _pearson(topic_odd, metric_odd)
            r_even = _pearson(topic_even, metric_even)
            if r_odd is None or r_even is None:
                continue
            sign_agree = 1 if (r_odd == 0 or r_even == 0 or (r_odd > 0) == (r_even > 0)) else 0
            delta = abs(r_odd - r_even)
            overall_signs.append(sign_agree)
            overall_deltas.append(delta)
            metric_pairs[metric]["signs"].append(sign_agree)
            metric_pairs[metric]["deltas"].append(delta)

    if not overall_signs or not overall_deltas:
        return None

    category = metadata.get("category")
    text_name = metadata.get("text_name")
    filename = metadata.get("filename")
    if not isinstance(category, str) or not isinstance(text_name, str) or not isinstance(filename, str):
        return None

    text_summary = {
        "category": category,
        "text_name": text_name,
        "filename": filename,
        "pair_count": len(overall_signs),
        "sign_agreement_rate": statistics.mean(overall_signs),
        "mean_abs_delta_r": statistics.mean(overall_deltas),
        "median_abs_delta_r": statistics.median(overall_deltas),
        "soft_score_threshold": score_threshold,
        "soft_top_k": score_top_k,
    }

    return {
        "text": text_summary,
        "overall_signs": overall_signs,
        "overall_deltas": overall_deltas,
        "metrics": metric_pairs,
    }


def _summarize_split_half_entries(
    entries: List[Dict[str, object]],
) -> Optional[Dict[str, object]]:
    if not entries:
        return None
    overall_signs: List[int] = []
    overall_deltas: List[float] = []
    metric_accum: Dict[str, Dict[str, List[float]]] = {}
    texts: List[Dict[str, object]] = []
    score_thresholds: List[float] = []
    score_top_ks: List[int] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        if isinstance(text, dict):
            texts.append(text)
            if isinstance(text.get("soft_score_threshold"), (int, float)):
                score_thresholds.append(float(text["soft_score_threshold"]))
            if isinstance(text.get("soft_top_k"), int):
                score_top_ks.append(text["soft_top_k"])
        signs = entry.get("overall_signs", [])
        deltas = entry.get("overall_deltas", [])
        if isinstance(signs, list):
            overall_signs.extend([int(val) for val in signs if isinstance(val, (int, float))])
        if isinstance(deltas, list):
            overall_deltas.extend([float(val) for val in deltas if isinstance(val, (int, float))])
        metrics = entry.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for metric, rows in metrics.items():
            if not isinstance(rows, dict):
                continue
            metric_accum.setdefault(metric, {"signs": [], "deltas": []})
            metric_accum[metric]["signs"].extend(
                [int(val) for val in rows.get("signs", []) if isinstance(val, (int, float))]
            )
            metric_accum[metric]["deltas"].extend(
                [float(val) for val in rows.get("deltas", []) if isinstance(val, (int, float))]
            )

    if not overall_signs or not overall_deltas:
        return None

    metrics_summary: Dict[str, object] = {}
    for metric, rows in metric_accum.items():
        signs = rows.get("signs", [])
        deltas = rows.get("deltas", [])
        if not signs or not deltas:
            continue
        metrics_summary[metric] = {
            "pair_count": len(signs),
            "sign_agreement_rate": statistics.mean(signs),
            "mean_abs_delta_r": statistics.mean(deltas),
            "median_abs_delta_r": statistics.median(deltas),
        }

    return {
        "text_count": len(texts),
        "pair_count": len(overall_signs),
        "overall": {
            "sign_agreement_rate": statistics.mean(overall_signs),
            "mean_abs_delta_r": statistics.mean(overall_deltas),
            "median_abs_delta_r": statistics.median(overall_deltas),
        },
        "metrics": metrics_summary,
        "texts": sorted(
            texts,
            key=lambda row: (row.get("category", ""), row.get("text_name", "")),
        ),
        "params": {
            "split_method": "odd_even",
            "soft_score_threshold": sorted(set(score_thresholds)),
            "soft_top_k": sorted(set(score_top_ks)),
        },
    }

def _write_central_topic_split_half_summary(
    entries: List[Dict[str, object]],
    *,
    use_existing: bool,
    output_root: Path,
) -> None:
    if not entries:
        return
    output_path = output_root / CENTRAL_TOPIC_SPLIT_HALF_FILENAME
    if use_existing and output_path.exists():
        return
    payload = _summarize_split_half_entries(entries)
    if not payload:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_genre_central_topic_split_half_summary(
    entries: List[Dict[str, object]],
    *,
    use_existing: bool,
    output_root: Path,
) -> None:
    if not entries:
        return
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for entry in entries:
        text = entry.get("text")
        if not isinstance(text, dict):
            continue
        genre, _author = _parse_category(text.get("category"))
        if not genre:
            continue
        grouped.setdefault(genre, []).append(entry)

    for genre, genre_entries in grouped.items():
        output_path = output_root / genre / GENRE_CENTRAL_TOPIC_SPLIT_HALF_FILENAME
        if use_existing and output_path.exists():
            continue
        payload = _summarize_split_half_entries(genre_entries)
        if not payload:
            continue
        payload["genre"] = genre
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def _write_author_central_topic_split_half_summary(
    entries: List[Dict[str, object]],
    *,
    use_existing: bool,
    output_root: Path,
) -> None:
    if not entries:
        return
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for entry in entries:
        text = entry.get("text")
        if not isinstance(text, dict):
            continue
        genre, author = _parse_category(text.get("category"))
        if not genre or not author:
            continue
        grouped.setdefault((genre, author), []).append(entry)

    for (genre, author), author_entries in grouped.items():
        filename = AUTHOR_CENTRAL_TOPIC_SPLIT_HALF_TEMPLATE.format(author=author)
        output_path = output_root / genre / author / filename
        if use_existing and output_path.exists():
            continue
        payload = _summarize_split_half_entries(author_entries)
        if not payload:
            continue
        payload["genre"] = genre
        payload["author"] = author
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def _collect_central_topic_r_values(
    output_root: Path,
) -> Dict[Tuple[str, str, str, int, str], float]:
    values: Dict[Tuple[str, str, str, int, str], float] = {}
    if not output_root.exists():
        return values
    for path in output_root.rglob("*_central_topic_correlations.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        extracted = _extract_central_topic_payload(payload)
        if extracted is None:
            continue
        metadata, report, _params = extracted
        if not isinstance(report, dict):
            continue
        category = metadata.get("category")
        text_name = metadata.get("text_name")
        filename = metadata.get("filename")
        if not isinstance(category, str) or not isinstance(text_name, str) or not isinstance(filename, str):
            continue
        central_topics = report.get("central_topics", [])
        if not isinstance(central_topics, list):
            continue
        for topic in central_topics:
            if not isinstance(topic, dict):
                continue
            topic_id = topic.get("topic_id")
            if not isinstance(topic_id, int):
                continue
            correlations = topic.get("correlations", {})
            if not isinstance(correlations, dict):
                continue
            for metric, corr_entry in correlations.items():
                if not isinstance(metric, str) or not isinstance(corr_entry, dict):
                    continue
                r_value = _as_float(corr_entry.get("pearson_r"))
                if r_value is None:
                    continue
                key = (category or "", text_name or "", filename or "", topic_id, metric)
                values[key] = r_value
    return values


def _write_cross_block_consistency_summary(
    output_roots: List[Path],
    block_labels: List[str],
    *,
    use_existing: bool,
    output_path: Optional[Path] = None,
) -> None:
    if len(output_roots) < 2:
        return
    if output_path is None:
        output_path = Path("data") / "results" / CROSS_BLOCK_CONSISTENCY_FILENAME
    if use_existing and output_path.exists():
        return

    block_data: List[Tuple[str, Path, Dict[Tuple[str, str, str, int, str], float]]] = []
    for label, root in zip(block_labels, output_roots):
        mapping = _collect_central_topic_r_values(root)
        if mapping:
            block_data.append((label, root, mapping))

    if len(block_data) < 2:
        return

    overall_signs: List[int] = []
    overall_deltas: List[float] = []
    metric_accum: Dict[str, Dict[str, List[float]]] = {}

    for (_label_a, _root_a, map_a), (_label_b, _root_b, map_b) in combinations(block_data, 2):
        common_keys = set(map_a.keys()) & set(map_b.keys())
        for key in common_keys:
            r_a = map_a[key]
            r_b = map_b[key]
            sign_agree = 1 if (r_a == 0 or r_b == 0 or (r_a > 0) == (r_b > 0)) else 0
            delta = abs(r_a - r_b)
            overall_signs.append(sign_agree)
            overall_deltas.append(delta)
            metric = key[4]
            metric_accum.setdefault(metric, {"signs": [], "deltas": []})
            metric_accum[metric]["signs"].append(sign_agree)
            metric_accum[metric]["deltas"].append(delta)

    if not overall_signs or not overall_deltas:
        return

    metrics_summary: Dict[str, object] = {}
    for metric, rows in metric_accum.items():
        signs = rows.get("signs", [])
        deltas = rows.get("deltas", [])
        if not signs or not deltas:
            continue
        metrics_summary[metric] = {
            "pair_count": len(signs),
            "sign_agreement_rate": statistics.mean(signs),
            "mean_abs_delta_r": statistics.mean(deltas),
            "median_abs_delta_r": statistics.median(deltas),
        }

    labels = [label for label, _root, _map in block_data]
    roots = [str(root) for _label, root, _map in block_data]
    payload = {
        "block_labels": labels,
        "pair_count": len(overall_signs),
        "overall": {
            "sign_agreement_rate": statistics.mean(overall_signs),
            "mean_abs_delta_r": statistics.mean(overall_deltas),
            "median_abs_delta_r": statistics.median(overall_deltas),
        },
        "metrics": metrics_summary,
        "params": {
            "block_labels": labels,
            "output_roots": roots,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
