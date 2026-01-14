"""Visualization utilities for dashboard outputs."""

from collections import defaultdict
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from src.e0_dashboard_metrics import collect_window_topic_scores
from x_configs import GENRES
from z_utils import analytics_path, figures_path, load_json


@dataclass(frozen=True)
class CentralTopicXBarConfig:
    top_n: int = 10
    p_threshold: float = 0.05
    fig_width: float = 9.0
    min_height: float = 4.5
    row_height: float = 0.45
    annotation_max_len: int = 40
    positive_color: str = "#2c7fb8"
    negative_color: str = "#d95f0e"
    alpha_significant: float = 0.9
    alpha_nonsignificant: float = 0.4


@dataclass(frozen=True)
class ExemplarScatterConfig:
    top_per_genre: int = 1
    min_points: int = 3
    fig_width: float = 7.5
    fig_height: float = 5.5
    point_size: float = 18.0
    point_alpha: float = 0.7
    cmap_name: str = "viridis"
    ci_z: float = 1.96


@dataclass(frozen=True)
class PresenceSlopegraphConfig:
    p_threshold: float = 0.01
    fig_width: float = 9.0
    min_height: float = 3.5
    row_height: float = 0.4
    positive_color: str = "#2c7fb8"
    negative_color: str = "#d95f0e"


@dataclass(frozen=True)
class AggregatedHeatmapConfig:
    p_threshold: float = 0.05
    fig_width: float = 9.0
    min_height: float = 6.0
    row_height: float = 0.3
    cmap_name: str = "coolwarm"
    mask_color: str = "#d9d9d9"
    exclude_metrics: Sequence[str] = (
        "syntax.clause_ratios.subordination_ratio",
        "discourse.connective_counts_per_token.Comparison",
        "discourse.explicit_connectives_per_token",
        "discourse.tense_shift",
        "lexico_semantics.content_function_ratio",
        "lexico_semantics.lexical_density_per_token",
        "lexico_semantics.lexical_diversity_mattr.mattr_score",
        "discourse.modality_per_token",
        "syntax.avg_dependents_per_head.main_clause",
        "syntax.avg_dependents_per_head.subordinate_clause",
    )


@dataclass(frozen=True)
class TopicMetricHeatmapConfig:
    value_key: str = "variance_delta"
    min_windows: int = 2
    top_n: Optional[int] = None
    min_width: float = 8.0
    min_height: float = 6.0
    col_width: float = 0.5
    row_height: float = 0.4
    cmap_name: str = "viridis"
    mask_color: str = "lightgrey"


@dataclass(frozen=True)
class DataSelectionConfig:
    genres: Optional[Sequence[str]] = None
    authors: Optional[Sequence[str]] = None
    texts: Optional[Sequence[str]] = None
    categories: Optional[Sequence[str]] = None
    exclude_genres: Optional[Sequence[str]] = None
    exclude_authors: Optional[Sequence[str]] = None
    exclude_texts: Optional[Sequence[str]] = None
    exclude_categories: Optional[Sequence[str]] = None


DEFAULT_CENTRAL_TOPIC_X_CONFIG = CentralTopicXBarConfig()
DEFAULT_EXEMPLAR_SCATTER_CONFIG = ExemplarScatterConfig()
DEFAULT_PRESENCE_SLOPEGRAPH_CONFIG = PresenceSlopegraphConfig()
DEFAULT_AGGREGATED_HEATMAP_CONFIG = AggregatedHeatmapConfig()
DEFAULT_TOPIC_METRIC_HEATMAP_CONFIG = TopicMetricHeatmapConfig()
DEFAULT_DATA_SELECTION_CONFIG = DataSelectionConfig(genres=tuple(GENRES))


def _shorten_keywords(keywords: Sequence[str], max_len: int = 40) -> str:
    if not keywords:
        return ""
    gloss = ", ".join([str(word) for word in keywords[:2]])
    return gloss if len(gloss) <= max_len else gloss[: max_len - 3] + "..."


def _matches_selection(
    genre: str,
    author: str,
    text_name: str,
    selection: Optional[DataSelectionConfig],
) -> bool:
    """Return True if the (genre, author, text_name) tuple passes include/exclude filters."""
    if selection is None:
        return True

    category = f"{genre}/{author}" if author else genre

    if selection.genres and genre not in selection.genres:
        return False
    if selection.authors and author not in selection.authors:
        return False
    if selection.texts and text_name not in selection.texts:
        return False
    if selection.categories and category not in selection.categories:
        return False

    if selection.exclude_genres and genre in selection.exclude_genres:
        return False
    if selection.exclude_authors and author in selection.exclude_authors:
        return False
    if selection.exclude_texts and text_name in selection.exclude_texts:
        return False
    if selection.exclude_categories and category in selection.exclude_categories:
        return False

    return True


def _top_n_by_abs_r(entries: Sequence[Dict[str, object]], n: int) -> List[Dict[str, object]]:
    def sort_key(entry: Dict[str, object]) -> Tuple[float, float]:
        r = entry.get("pearson_r")
        p = entry.get("p_value")
        if not isinstance(r, (int, float)):
            r = 0.0
        if not isinstance(p, (int, float)):
            p = 1.0
        return (-abs(float(r)), float(p))

    return sorted(entries, key=sort_key)[:n]


def _output_dir(
    output_root: Optional[Path],
    subfolder: str,
    category: Optional[Sequence[str]] = None,
) -> Path:
    if output_root is None:
        return figures_path(subfolder, category)
    path = Path(output_root) / subfolder
    if category:
        path = path.joinpath(*category)
    return path


def _iter_central_topic_reports(
    dashboard_root: Optional[Path] = None,
) -> Iterable[Tuple[Path, Dict[str, object]]]:
    root = dashboard_root or analytics_path("dashboard")
    if not root.exists():
        raise FileNotFoundError(f"Dashboard directory not found: {root}")
    for path in root.rglob("*_central_topic_correlations.json"):
        if path.name == "00_genre_central_topic_presence_correlations.json":
            continue
        payload = load_json(path)
        if isinstance(payload, dict):
            yield path, payload


def collect_central_topic_data(
    dashboard_root: Optional[Path] = None,
    selection: Optional[DataSelectionConfig] = None,
) -> Tuple[Dict[str, List[Dict[str, object]]], List[Dict[str, object]]]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    entries_by_genre: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    presence_entries: List[Dict[str, object]] = []

    for path, data in _iter_central_topic_reports(dashboard_root):
        category = data.get("category") or ""
        category_parts = [part for part in str(category).split("/") if part]
        genre = category_parts[0] if category_parts else "unknown"
        author = category_parts[1] if len(category_parts) > 1 else "unknown"
        text_name = data.get("text_name") or path.parent.name
        if not _matches_selection(genre, author, text_name, selection):
            continue
        topic_file = data.get("topic_file")
        report = data.get("central_report") or {}
        params = report.get("params") or {}

        for topic in report.get("central_topics_ordered") or []:
            correlations = topic.get("correlations") or {}
            for metric, corr in correlations.items():
                r = corr.get("pearson_r")
                if not isinstance(r, (int, float)):
                    continue
                entries_by_genre[genre].append(
                    {
                        "genre": genre,
                        "author": author,
                        "text_name": text_name,
                        "metric": metric,
                        "pearson_r": float(r),
                        "p_value": corr.get("p_value"),
                        "topic_id": topic.get("topic_id"),
                        "topic_score": topic.get("score"),
                        "keywords": topic.get("keywords") or [],
                        "stats": topic.get("stats") or {},
                        "topic_file": topic_file,
                        "params": params,
                    }
                )

        presence = report.get("central_topic_presence") or {}
        if presence.get("correlations"):
            presence_entries.append(
                {
                    "genre": genre,
                    "author": author,
                    "text_name": text_name,
                    "presence": presence,
                }
            )

    return entries_by_genre, presence_entries


def _filter_entries_by_genre(
    entries_by_genre: Dict[str, List[Dict[str, object]]],
    selection: Optional[DataSelectionConfig],
) -> Dict[str, List[Dict[str, object]]]:
    if selection is None:
        return entries_by_genre
    filtered: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for entries in entries_by_genre.values():
        for entry in entries:
            genre = str(entry.get("genre", ""))
            author = str(entry.get("author", ""))
            text_name = str(entry.get("text_name", ""))
            if _matches_selection(genre, author, text_name, selection):
                filtered[genre].append(entry)
    return filtered


def _filter_presence_entries(
    presence_entries: List[Dict[str, object]],
    selection: Optional[DataSelectionConfig],
) -> List[Dict[str, object]]:
    if selection is None:
        return presence_entries
    filtered = []
    for entry in presence_entries:
        genre = str(entry.get("genre", ""))
        author = str(entry.get("author", ""))
        text_name = str(entry.get("text_name", ""))
        if _matches_selection(genre, author, text_name, selection):
            filtered.append(entry)
    return filtered


def plot_central_topic_x_bars(
    entries_by_genre: Optional[Dict[str, List[Dict[str, object]]]] = None,
    *,
    config: Optional[CentralTopicXBarConfig] = None,
    top_n: Optional[int] = None,
    p_threshold: Optional[float] = None,
    selection: Optional[DataSelectionConfig] = None,
    output_root: Optional[Path] = None,
) -> List[Path]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    if entries_by_genre is None:
        entries_by_genre, _ = collect_central_topic_data(selection=selection)
    else:
        entries_by_genre = _filter_entries_by_genre(entries_by_genre, selection)

    if config is None:
        config = DEFAULT_CENTRAL_TOPIC_X_CONFIG
    if top_n is None:
        top_n = config.top_n
    if p_threshold is None:
        p_threshold = config.p_threshold

    genre_set = {genre for genre in entries_by_genre if genre}
    ordered_genres = [genre for genre in GENRES if genre in genre_set]
    ordered_genres += sorted(genre_set - set(ordered_genres))

    output_paths: List[Path] = []
    for genre in ordered_genres:
        entries = entries_by_genre.get(genre, [])
        top_entries = _top_n_by_abs_r(entries, top_n)
        if not top_entries:
            continue

        entries_rev = list(reversed(top_entries))
        labels = [entry["metric"] for entry in entries_rev]
        values = [entry["pearson_r"] for entry in entries_rev]
        pvals = [entry.get("p_value") for entry in entries_rev]
        colors = [
            config.positive_color if value >= 0 else config.negative_color for value in values
        ]
        alphas = [
            config.alpha_significant
            if isinstance(p, (int, float)) and p < p_threshold
            else config.alpha_nonsignificant
            for p in pvals
        ]

        fig_height = max(config.min_height, config.row_height * len(labels) + 1.5)
        fig, ax = plt.subplots(figsize=(config.fig_width, fig_height))
        y_pos = np.arange(len(labels))
        for idx, (value, color, alpha) in enumerate(zip(values, colors, alphas)):
            ax.barh(y_pos[idx], value, color=color, alpha=alpha)

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Pearson r")
        ax.set_title(f"A: Central Topic-X to Metrics (Top {top_n}) - {genre}")

        max_abs = max(abs(value) for value in values) if values else 0.0
        span = max_abs if max_abs > 0 else 0.1
        ax.set_xlim(-span * 1.1, span * 1.1)
        offset = span * 0.02
        for idx, entry in enumerate(entries_rev):
            value = entry["pearson_r"]
            gloss = _shorten_keywords(entry.get("keywords") or [], config.annotation_max_len)
            topic_id = entry.get("topic_id")
            topic_label = f"topic_id {topic_id}" if topic_id is not None else "topic_id ?"
            annotation = f"{topic_label}: {gloss}" if gloss else topic_label
            x_pos = value + offset if value >= 0 else offset
            ax.text(
                x_pos,
                y_pos[idx],
                annotation,
                va="center",
                ha="left",
                fontsize=7,
            )

        ax.margins(y=0.02)
        plt.tight_layout()

        output_dir = _output_dir(output_root, "A_central_topic_x", [genre])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{genre}_central_topic_x_top{top_n}.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def plot_presence_slopegraphs(
    presence_entries: Optional[List[Dict[str, object]]] = None,
    *,
    config: Optional[PresenceSlopegraphConfig] = None,
    p_threshold: Optional[float] = None,
    selection: Optional[DataSelectionConfig] = None,
    output_root: Optional[Path] = None,
) -> List[Path]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    if presence_entries is None:
        _, presence_entries = collect_central_topic_data(selection=selection)
    else:
        presence_entries = _filter_presence_entries(presence_entries, selection)

    if config is None:
        config = DEFAULT_PRESENCE_SLOPEGRAPH_CONFIG
    if p_threshold is None:
        p_threshold = config.p_threshold

    output_paths: List[Path] = []
    for entry in presence_entries:
        genre = entry["genre"]
        author = entry["author"]
        text_name = entry["text_name"]
        presence = entry["presence"]
        correlations = presence.get("correlations") or {}

        metrics = []
        for metric, corr in correlations.items():
            p_value = corr.get("p_value")
            if not isinstance(p_value, (int, float)) or p_value >= p_threshold:
                continue
            mean_with = corr.get("mean_with_topic")
            mean_without = corr.get("mean_without_topic")
            if not isinstance(mean_with, (int, float)) or not isinstance(mean_without, (int, float)):
                continue
            metrics.append((metric, float(mean_without), float(mean_with)))

        if not metrics:
            continue

        metrics.sort(key=lambda row: row[0])
        y_pos = np.arange(len(metrics))
        fig_height = max(config.min_height, config.row_height * len(metrics) + 1.5)
        fig, ax = plt.subplots(figsize=(config.fig_width, fig_height))

        for idx, (metric, mean_without, mean_with) in enumerate(metrics):
            color = (
                config.positive_color if mean_with >= mean_without else config.negative_color
            )
            ax.plot(
                [mean_without, mean_with],
                [y_pos[idx], y_pos[idx]],
                color=color,
                linewidth=1.8,
                alpha=0.8,
            )
            ax.scatter(
                [mean_without],
                [y_pos[idx]],
                color="white",
                edgecolor=color,
                s=40,
                zorder=3,
                marker="o",
            )
            ax.scatter(
                [mean_with],
                [y_pos[idx]],
                color=color,
                s=40,
                zorder=3,
                marker="s",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([row[0] for row in metrics], fontsize=8)
        ax.set_xlabel("Mean metric value")
        ax.set_title(f"E: Central Topic Presence - {text_name} ({genre})")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        ax.scatter([], [], color="white", edgecolor="black", label="without", marker="o")
        ax.scatter([], [], color="black", label="with", marker="s")
        ax.legend(loc="best", fontsize=8, frameon=False)

        plt.tight_layout()

        output_dir = _output_dir(output_root, "E_presence_slopegraphs", [genre, author])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{text_name}_slopegraph.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def plot_aggregated_presence_heatmap(
    *,
    config: Optional[AggregatedHeatmapConfig] = None,
    p_threshold: Optional[float] = None,
    selection: Optional[DataSelectionConfig] = None,
    output_root: Optional[Path] = None,
) -> Optional[Path]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    if config is None:
        config = DEFAULT_AGGREGATED_HEATMAP_CONFIG
    if p_threshold is None:
        p_threshold = config.p_threshold

    dashboard_root = analytics_path("dashboard")
    agg_paths = sorted(dashboard_root.glob("*/00_genre_central_topic_presence_correlations.json"))
    if not agg_paths:
        return None

    agg_data: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}
    metrics_set = set()
    for path in agg_paths:
        payload = load_json(path)
        genre = payload.get("genre") or path.parent.name
        if selection and not _matches_selection(genre, "", "", selection):
            continue
        metrics = payload.get("metrics") or {}
        agg_data[genre] = {}
        for metric, corr in metrics.items():
            r = corr.get("pearson_r")
            p = corr.get("p_value")
            agg_data[genre][metric] = (r, p)
            metrics_set.add(metric)

    genres = [genre for genre in GENRES if not selection or _matches_selection(genre, "", "", selection)]
    if config.exclude_metrics:
        metrics_set = {metric for metric in metrics_set if metric not in config.exclude_metrics}
    metrics_list = sorted(metrics_set)
    if not genres or not metrics_list:
        return None

    r_matrix = np.full((len(metrics_list), len(genres)), np.nan)
    p_matrix = np.full((len(metrics_list), len(genres)), np.nan)
    for i, metric in enumerate(metrics_list):
        for j, genre in enumerate(genres):
            r_val, p_val = agg_data.get(genre, {}).get(metric, (None, None))
            if isinstance(r_val, (int, float)):
                r_matrix[i, j] = float(r_val)
            if isinstance(p_val, (int, float)):
                p_matrix[i, j] = float(p_val)

    variances = np.nanvar(r_matrix, axis=1)
    order = np.argsort(-variances)
    r_matrix = r_matrix[order]
    p_matrix = p_matrix[order]
    metrics_list = [metrics_list[idx] for idx in order]

    r_matrix_masked = r_matrix.copy()
    r_matrix_masked[p_matrix >= p_threshold] = np.nan
    if np.isfinite(r_matrix_masked).any():
        vmax = float(np.nanmax(np.abs(r_matrix_masked)))
    else:
        vmax = 0.1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig_height = max(config.min_height, config.row_height * len(metrics_list) + 2)
    fig, ax = plt.subplots(figsize=(config.fig_width, fig_height))
    cmap = plt.get_cmap(config.cmap_name).copy()
    cmap.set_bad(color=config.mask_color)
    im = ax.imshow(r_matrix_masked, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(genres)))
    ax.set_xticklabels(genres, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(metrics_list)))
    ax.set_yticklabels(metrics_list, fontsize=8)
    ax.set_title("F: Aggregated Central Topic Presence - Genre x Metric (r)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r")
    plt.tight_layout()

    output_dir = _output_dir(output_root, "F_aggregated_heatmap")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "aggregated_genre_metric_heatmap.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _metric_parts(metric: str) -> Tuple[str, List[str]]:
    """Split a dotted metric name into (domain, remaining path parts)."""
    parts = metric.split(".")
    if len(parts) < 2:
        return metric, []
    return parts[0], parts[1:]


def _get_metric_value(window: Dict[str, object], parts: Sequence[str]) -> Optional[float]:
    """Follow a dotted metric path inside a window dict and return a numeric value if present."""
    value: object = window
    for part in parts:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value.get(part)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _resolve_topic_path(entry: Dict[str, object]) -> Optional[Path]:
    """Resolve the topic_file path in a dashboard entry, returning None if missing or invalid."""
    raw_path = entry.get("topic_file")
    if not raw_path:
        return None
    candidate = Path(str(raw_path))
    if candidate.exists():
        return candidate
    candidate = Path(str(raw_path).replace("\\", "/"))
    return candidate if candidate.exists() else None


def _resolve_soft_params(
    entry: Dict[str, object],
    topics_data: Dict[str, object],
) -> Tuple[float, Optional[int]]:
    """Return (soft_score_threshold, soft_top_k) using entry params with topic meta fallback."""
    params = entry.get("params") or {}
    threshold = params.get("soft_score_threshold")
    top_k = params.get("soft_top_k")
    meta = topics_data.get("meta") if isinstance(topics_data, dict) else {}
    if threshold is None and isinstance(meta, dict):
        threshold = meta.get("soft_score_threshold")
    if threshold is None:
        threshold = 0.5
    if top_k is None and isinstance(meta, dict):
        top_k = meta.get("soft_top_k") or meta.get("soft_top_k_topics")
    return float(threshold), int(top_k) if top_k is not None else None


def plot_exemplar_scatter(
    entries_by_genre: Optional[Dict[str, List[Dict[str, object]]]] = None,
    *,
    config: Optional[ExemplarScatterConfig] = None,
    top_per_genre: Optional[int] = None,
    min_points: Optional[int] = None,
    selection: Optional[DataSelectionConfig] = None,
    output_root: Optional[Path] = None,
) -> List[Path]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    if entries_by_genre is None:
        entries_by_genre, _ = collect_central_topic_data(selection=selection)
    else:
        entries_by_genre = _filter_entries_by_genre(entries_by_genre, selection)

    if config is None:
        config = DEFAULT_EXEMPLAR_SCATTER_CONFIG
    if top_per_genre is None:
        top_per_genre = config.top_per_genre
    if min_points is None:
        min_points = config.min_points

    genre_set = {genre for genre in entries_by_genre if genre}
    ordered_genres = [genre for genre in GENRES if genre in genre_set]
    ordered_genres += sorted(genre_set - set(ordered_genres))

    output_paths: List[Path] = []
    max_candidates = max(int(top_per_genre or 1), 1)
    for genre in ordered_genres:
        candidates = _top_n_by_abs_r(entries_by_genre.get(genre, []), max_candidates)
        for entry in candidates:
            topic_path = _resolve_topic_path(entry)
            if topic_path is None:
                continue
            topics_data = load_json(topic_path)

            metric = entry.get("metric")
            if not isinstance(metric, str):
                continue
            domain, metric_parts = _metric_parts(metric)
            if not metric_parts:
                continue

            text_name = entry.get("text_name")
            author = entry.get("author")
            if not text_name or not author:
                continue

            window_metrics_path = analytics_path(
                "window",
                [genre, author, text_name],
                f"{text_name}_window_metrics.json",
            )
            if not window_metrics_path.exists():
                continue

            window_data = load_json(window_metrics_path)
            domain_windows = window_data.get(domain, {}).get("windows") or []
            if not domain_windows:
                continue

            threshold, top_k = _resolve_soft_params(entry, topics_data)
            scores_by_window = collect_window_topic_scores(
                topics_data,
                soft_score_threshold=threshold,
                soft_top_k=top_k,
                window_entries=domain_windows,
            )
            topic_id = entry.get("topic_id")
            if not isinstance(topic_id, int):
                continue

            topic_scores = [scores.get(topic_id, 0.0) for scores in scores_by_window]
            metric_values = [_get_metric_value(window, metric_parts) for window in domain_windows]

            count = min(len(topic_scores), len(metric_values))
            x_vals = []
            y_vals = []
            positions = []
            for idx in range(count):
                value = metric_values[idx]
                if value is None:
                    continue
                x_vals.append(float(topic_scores[idx]))
                y_vals.append(float(value))
                positions.append(float(idx))

            if len(x_vals) < min_points:
                continue

            x = np.array(x_vals, dtype=float)
            y = np.array(y_vals, dtype=float)
            pos = np.array(positions, dtype=float)

            fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))
            scatter = ax.scatter(
                x,
                y,
                c=pos,
                cmap=config.cmap_name,
                s=config.point_size,
                alpha=config.point_alpha,
                edgecolors="none",
            )

            x_mean = float(x.mean())
            y_mean = float(y.mean())
            sxx = float(((x - x_mean) ** 2).sum())
            if sxx > 0 and len(x) > 2:
                sxy = float(((x - x_mean) * (y - y_mean)).sum())
                slope = sxy / sxx
                intercept = y_mean - slope * x_mean
                x_line = np.linspace(float(x.min()), float(x.max()), 200)
                y_line = intercept + slope * x_line
                resid = y - (intercept + slope * x)
                s_err = math.sqrt(float((resid**2).sum()) / (len(x) - 2))
                se_mean = s_err * np.sqrt(1 / len(x) + (x_line - x_mean) ** 2 / sxx)
                ci = config.ci_z * se_mean
                ax.plot(x_line, y_line, color="black", linewidth=1.5)
                ax.fill_between(x_line, y_line - ci, y_line + ci, color="black", alpha=0.15)

            ax.set_xlabel("Topic soft score")
            ax.set_ylabel(metric)
            ax.set_title(f"C: {text_name} ({genre}) - topic {topic_id} vs {metric}")
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label("Narrative position (window index)")

            plt.tight_layout()
            output_dir = _output_dir(output_root, "C_scatter_exemplars", [genre])
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_metric = metric.replace(":", "_").replace("/", "_")
            output_path = output_dir / f"{text_name}__topic{topic_id}__{safe_metric}.png"
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
            output_paths.append(output_path)
            break

    return output_paths


def plot_topic_metric_heatmap(
    report: Dict[str, object],
    output_path: Path,
    config: Optional[TopicMetricHeatmapConfig] = None,
    value_key: Optional[str] = None,
    min_windows: Optional[int] = None,
    top_n: Optional[int] = None,
) -> None:
    """Render a heatmap for a selected topic/metric statistic."""
    if config is None:
        config = DEFAULT_TOPIC_METRIC_HEATMAP_CONFIG
    if value_key is None:
        value_key = config.value_key
    if min_windows is None:
        min_windows = config.min_windows
    if top_n is None:
        top_n = config.top_n

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
        figsize=(
            max(config.min_width, len(metric_names) * config.col_width),
            max(config.min_height, len(topic_labels) * config.row_height),
        )
    )
    cmap = plt.get_cmap(config.cmap_name).copy()
    cmap.set_bad(color=config.mask_color)
    im = ax.imshow(data, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.set_yticks(range(len(topic_labels)))
    ax.set_yticklabels(topic_labels)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Topic ID")
    ax.set_title(f"Topic metric heatmap ({value_key})")
    fig.colorbar(im, ax=ax)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_all_visualisations(
    *,
    top_n: Optional[int] = None,
    bar_p_threshold: Optional[float] = None,
    slope_p_threshold: Optional[float] = None,
    heatmap_p_threshold: Optional[float] = None,
    a_config: Optional[CentralTopicXBarConfig] = None,
    c_config: Optional[ExemplarScatterConfig] = None,
    e_config: Optional[PresenceSlopegraphConfig] = None,
    f_config: Optional[AggregatedHeatmapConfig] = None,
    selection: Optional[DataSelectionConfig] = None,
    output_root: Optional[Path] = None,
) -> Dict[str, object]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    entries_by_genre, presence_entries = collect_central_topic_data(selection=selection)

    if a_config is None:
        a_config = DEFAULT_CENTRAL_TOPIC_X_CONFIG
    if top_n is not None:
        a_config = replace(a_config, top_n=top_n)
    if bar_p_threshold is not None:
        a_config = replace(a_config, p_threshold=bar_p_threshold)

    if c_config is None:
        c_config = DEFAULT_EXEMPLAR_SCATTER_CONFIG

    if e_config is None:
        e_config = DEFAULT_PRESENCE_SLOPEGRAPH_CONFIG
    if slope_p_threshold is not None:
        e_config = replace(e_config, p_threshold=slope_p_threshold)

    if f_config is None:
        f_config = DEFAULT_AGGREGATED_HEATMAP_CONFIG
    if heatmap_p_threshold is not None:
        f_config = replace(f_config, p_threshold=heatmap_p_threshold)

    results = {
        "A": plot_central_topic_x_bars(
            entries_by_genre,
            config=a_config,
            selection=selection,
            output_root=output_root,
        ),
        "C": plot_exemplar_scatter(
            entries_by_genre,
            config=c_config,
            selection=selection,
            output_root=output_root,
        ),
        "E": plot_presence_slopegraphs(
            presence_entries,
            config=e_config,
            selection=selection,
            output_root=output_root,
        ),
        "F": plot_aggregated_presence_heatmap(
            config=f_config,
            selection=selection,
            output_root=output_root,
        ),
    }
    return results


if __name__ == "__main__":
    generate_all_visualisations()
