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
  "entity_overlap_rate": 0.12,
  "avg_token_surprisal": 2.31,
  "max_token_surprisal": 3.02,
  "surprisal_variance": 0.08,
  "lexical_density": 0.61,
  "mean_dependency_depth": 1.9,
  "max_dependency_depth": 3.3,
  "clause_density": 0.25,
  "avg_dependents_per_head": 2.1,
  "agent_rate_per_clause": 0.4,
  "patient_rate_per_clause": 0.2,
  "role_coverage": 1.2
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
    "variance_only": false,
    "block_size": 5,
    "permutations": 2000
  }
}
"""

import json
import math
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
    return data if isinstance(data, dict) else {}


def safe_mean(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return None
    return sum(clean) / len(clean)


def safe_max(values: List[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return None
    return max(clean)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not (
        isinstance(value, float) and math.isnan(value)
    )


def flatten_numeric_metrics(entry: Dict[str, object], prefix: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in entry.items():
        if key in {"start_sentence", "end_sentence"}:
            continue
        metric_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if is_number(sub_val):
                    metrics[f"{metric_key}.{sub_key}"] = float(sub_val)
        elif is_number(value):
            metrics[metric_key] = float(value)
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


def collect_window_topic_scores(
    topics_data: Dict[str, object],
    *,
    soft_score_threshold: Optional[float],
    soft_top_k: Optional[int],
    window_count: int,
) -> List[Dict[int, float]]:
    if not topics_data or not isinstance(topics_data, dict):
        return [{} for _ in range(window_count)]

    windows = topics_data.get("windows") or []
    scores_by_window: List[Dict[int, float]] = []
    for window in windows[:window_count]:
        if window.get("is_noise"):
            scores_by_window.append({})
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
        if soft_top_k is not None and soft_top_k > 0:
            items = items[:soft_top_k]
        if soft_score_threshold is not None:
            items = [(topic_id, score) for topic_id, score in items if score >= soft_score_threshold]
        scores_by_window.append({topic_id: score for topic_id, score in items})

    if len(scores_by_window) < window_count:
        scores_by_window.extend([{} for _ in range(window_count - len(scores_by_window))])

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
            if is_number(val):
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
        mean = float(np.mean(scores))
        sd = float(np.std(scores, ddof=0))
        threshold = mean + sd
        ranked = [row for row in ranked if row["score"] >= threshold]
    return ranked[:top_n]


def build_topic_correlation_report(
    window_data: Dict[str, object],
    topics_data: Dict[str, object],
    *,
    soft_score_threshold: Optional[float] = 0.5,
    soft_top_k: Optional[int] = 3,
    central_top_n: int = 5,
    variance_only: bool = False,
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
    if variance_only:
        metric_names = [name for name in metric_names if "variance" in name]
        if not metric_names:
            return {}
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
            window_count=len(window_table),
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
                if is_number(val):
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
            metric_correlations[metric] = {
                "pearson_r": corr,
                "p_value": p_value,
                "n": len(values),
                "mean_with_topic": safe_mean(values_with),
                "mean_without_topic": safe_mean(values_without),
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
            "variance_only": variance_only,
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
        variance_only=False,
        topics_key="central_topics",
        use_soft_scores=True,
        block_size=block_size,
        permutations=permutations,
    )

    report: Dict[str, object] = weighted_report if weighted_report else {}
    if central_topic_ids:
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
                    if is_number(value):
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
                    "mean_with_topic": safe_mean(values_with),
                    "mean_without_topic": safe_mean(values_without),
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
        if isinstance(window, dict) and is_number(window.get("entity_overlap_ratio"))
    ]
    row["entity_overlap_rate"] = safe_mean(overlap_rates)

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
        if is_number(mean_surprisal) and is_number(num_tokens) and num_tokens > 0:
            mean_surprisals.append(float(mean_surprisal))
        if is_number(variance):
            variance_values.append(float(variance))
    row["avg_token_surprisal"] = safe_mean(mean_surprisals)
    row["max_token_surprisal"] = safe_max(mean_surprisals)
    row["surprisal_variance"] = safe_mean(variance_values)

    lexico = window_data.get("lexico_semantics", {}) if isinstance(window_data, dict) else {}
    lexico_windows = lexico.get("windows", []) if isinstance(lexico, dict) else []
    densities = [
        window.get("lexical_density")
        for window in lexico_windows
        if isinstance(window, dict) and is_number(window.get("lexical_density"))
    ]
    row["lexical_density"] = safe_mean(densities)

    syntax = window_data.get("syntax", {}) if isinstance(window_data, dict) else {}
    syntax_windows = syntax.get("windows", []) if isinstance(syntax, dict) else []
    mean_depths = []
    max_depths = []
    clauses = []
    dependents = []
    for window in syntax_windows:
        if not isinstance(window, dict):
            continue
        mean_depths.append(window.get("mean_depth"))
        max_depths.append(window.get("max_depth"))
        clause_counts_per_token = window.get("clause_counts_per_token", {})
        if isinstance(clause_counts_per_token, dict):
            clause_total = sum(
                val for val in clause_counts_per_token.values() if is_number(val)
            )
            clauses.append(clause_total)
        dep_per_head = window.get("avg_dependents_per_head", {})
        if isinstance(dep_per_head, dict):
            dep_vals = [val for val in dep_per_head.values() if is_number(val)]
            dependents.append(safe_mean(dep_vals))
    row["mean_dependency_depth"] = safe_mean(mean_depths)
    row["max_dependency_depth"] = safe_max(max_depths)
    row["clause_density"] = safe_mean(clauses)
    row["avg_dependents_per_head"] = safe_mean(dependents)

    agent_rates = []
    patient_rates = []
    role_coverages = []
    for window in lexico_windows:
        if not isinstance(window, dict):
            continue
        num_agents_per_clause = window.get("num_agents_per_clause")
        num_patients_per_clause = window.get("num_patients_per_clause")
        role_count_per_clause = window.get("role_count_per_clause")
        if is_number(num_agents_per_clause):
            agent_rates.append(float(num_agents_per_clause))
        if is_number(num_patients_per_clause):
            patient_rates.append(float(num_patients_per_clause))
        if is_number(role_count_per_clause):
            role_coverages.append(float(role_count_per_clause))
    row["agent_rate_per_clause"] = safe_mean(agent_rates)
    row["patient_rate_per_clause"] = safe_mean(patient_rates)
    row["role_coverage"] = safe_mean(role_coverages)
    return row
