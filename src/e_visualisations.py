"""Visualization utilities for dashboard outputs."""

import csv
import textwrap
from collections import defaultdict
from dataclasses import replace
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from .d1_dashboard_metrics import (
    collect_window_tables,
    collect_window_topic_scores,
    load_window_metrics,
)
from .x_configs import (
    CENTRAL_TOPIC_CORRELATIONS_SUFFIX,
    CENTRAL_TOPIC_PRESENCE_CORRELATIONS_SUFFIX,
    GENRES,
    AggregatedHeatmapConfig,
    CentralTopicWindowHeatmapConfig,
    CentralTopicXBarConfig,
    ConvergenceIndexConfig,
    DashboardCorrelationConfig,
    DataSelectionConfig,
    ExemplarScatterConfig,
    ForestPlotConfig,
    PresenceSlopegraphConfig,
    DEFAULT_BLOCK_SIZE,
    GENRE_CENTRAL_PRESENCE_FILENAME,
    GENRE_CENTRAL_TOPIC_SPLIT_HALF_FILENAME,
    StabilityFilterConfig,
    StabilityStackedBarConfig,
    TABLE_COUNT_COLUMNS,
    TABLE_DEFAULT_FLOAT_DECIMALS,
    TABLE_P_VALUE_COLUMNS,
    TABLE_P_VALUE_MIN_DISPLAY,
    TextMetricHeatmapConfig,
    TopicMetricLineConfig,
    TopicMetricHeatmapConfig,
    CONVERGENCE_METRIC_LABELS,
    DEFAULT_AGGREGATED_HEATMAP_CONFIG,
    DEFAULT_CENTRAL_TOPIC_WINDOW_HEATMAP_CONFIG,
    DEFAULT_CENTRAL_TOPIC_X_CONFIG,
    DEFAULT_CONVERGENCE_INDEX_CONFIG,
    DEFAULT_DASHBOARD_CORRELATION_CONFIG,
    DEFAULT_DATA_SELECTION_CONFIG,
    DEFAULT_EXEMPLAR_SCATTER_CONFIG,
    DEFAULT_FOREST_PLOT_CONFIG,
    DEFAULT_PRESENCE_SLOPEGRAPH_CONFIG,
    DEFAULT_STABILITY_FILTER_CONFIG,
    DEFAULT_STABILITY_STACKED_BAR_CONFIG,
    DEFAULT_TEXT_METRIC_HEATMAP_CONFIG,
    DEFAULT_TOPIC_METRIC_LINE_CONFIG,
    DEFAULT_TOPIC_METRIC_HEATMAP_CONFIG,
)
from .z_utils import analytics_path, find_topic_file, load_json, results_path


def _shorten_keywords(keywords: Sequence[str], max_len: int = 40) -> str:
    if not keywords:
        return ""
    gloss = ", ".join([str(word) for word in keywords[:2]])
    return gloss if len(gloss) <= max_len else gloss[: max_len - 3] + "..."


def _shorten_label(label: str, max_len: int) -> str:
    if max_len <= 0:
        return label
    if len(label) <= max_len:
        return label
    return label[: max_len - 3] + "..."


def _format_keywords_for_table(keywords: object) -> str:
    if keywords is None:
        return ""
    if isinstance(keywords, str):
        items = [keywords]
    elif isinstance(keywords, Sequence):
        items = list(keywords)
    else:
        return ""
    cleaned: List[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).replace("\t", " ").replace("\r", " ").replace("\n", " ")
        text = " ".join(text.split())
        if text:
            cleaned.append(text)
    return " | ".join(cleaned)


def _scores_to_stats(order: Sequence[object], scores: object) -> Dict[str, float]:
    if not isinstance(scores, list):
        return {}
    stats: Dict[str, float] = {}
    for metric, val in zip(order, scores):
        if isinstance(val, float) and math.isnan(val):
            continue
        if isinstance(val, (int, float)):
            stats[str(metric)] = float(val)
    return stats


def _format_table_value(value: object, column: str) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, (int, float)):
        num = float(value)
        if column in TABLE_COUNT_COLUMNS:
            return str(int(num))
        if column in TABLE_P_VALUE_COLUMNS:
            if num < TABLE_P_VALUE_MIN_DISPLAY:
                return f"<{TABLE_P_VALUE_MIN_DISPLAY:.3f}"
            return f"{num:.{TABLE_DEFAULT_FLOAT_DECIMALS}f}"
        return f"{num:.{TABLE_DEFAULT_FLOAT_DECIMALS}f}"
    text = str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return " ".join(text.split())


def _write_tsv(path: Path, rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(list(columns))
        for row in rows:
            writer.writerow([_format_table_value(row.get(col), col) for col in columns])


def _wrap_table_cell(text: str, *, max_width: int) -> str:
    if not text or max_width <= 0:
        return text or ""
    if "\n" in text:
        wrapped_lines: List[str] = []
        for line in text.splitlines():
            if not line:
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(
                textwrap.wrap(
                    line,
                    width=max_width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        return "\n".join(wrapped_lines)
    return textwrap.fill(
        text,
        width=max_width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _render_table_png(
    output_path: Path,
    rows: Sequence[Dict[str, object]],
    columns: Sequence[str],
    *,
    title: Optional[str] = None,
    max_col_width: int = 20,
    max_keyword_width: int = 48,
    font_size: int = 8,
    row_scale: float = 1.8,
) -> Optional[Path]:
    if not rows:
        return None

    cell_text: List[List[str]] = []
    col_max_lengths = [len(str(col)) for col in columns]
    row_line_counts: List[int] = []
    for row in rows:
        row_texts: List[str] = []
        max_lines = 1
        for col_idx, col in enumerate(columns):
            raw_text = _format_table_value(row.get(col), col)
            if col == "metric":
                raw_text = raw_text.replace(".", ".\n")
            wrap_width = max_keyword_width if col == "keywords" else max_col_width
            wrapped = _wrap_table_cell(raw_text, max_width=wrap_width)
            lines = wrapped.splitlines() if wrapped else [""]
            max_lines = max(max_lines, len(lines))
            max_line_len = max(len(line) for line in lines)
            col_max_lengths[col_idx] = max(col_max_lengths[col_idx], max_line_len)
            row_texts.append(wrapped)
        row_line_counts.append(max_lines)
        cell_text.append(row_texts)

    total_chars = sum(col_max_lengths) or 1
    total_lines = sum(row_line_counts) + 1
    fig_width = max(6.0, min(18.0, 0.10 * total_chars))
    fig_height = max(2.5, min(18.0, 0.30 * total_lines + 0.5))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, pad=12)

    table = ax.table(
        cellText=cell_text,
        colLabels=list(columns),
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    if row_scale and row_scale > 0:
        table.scale(1.0, row_scale)

    total = float(sum(col_max_lengths)) or 1.0
    for j, col_len in enumerate(col_max_lengths):
        width = col_len / total
        for i in range(len(rows) + 1):
            cell = table[(i, j)]
            cell.set_width(width)
            cell.PAD = 0.02
            if i == 0:
                cell.set_text_props(weight="bold")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output_path


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


def _matches_genre_selection(genre: str, selection: Optional[DataSelectionConfig]) -> bool:
    if selection is None:
        return True
    if selection.genres and genre not in selection.genres:
        return False
    if selection.exclude_genres and genre in selection.exclude_genres:
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


def _ordered_genres(genres: Iterable[str]) -> List[str]:
    genre_set = {genre for genre in genres if genre}
    ordered = [genre for genre in GENRES if genre in genre_set]
    ordered += sorted(genre_set - set(ordered))
    return ordered


def _stability_passes(
    entry: Optional[Dict[str, object]],
    config: StabilityFilterConfig,
) -> bool:
    if not isinstance(entry, dict):
        return False
    value = entry.get(config.metric_key)
    if not isinstance(value, (int, float)) or (isinstance(value, float) and math.isnan(value)):
        return False
    if config.min_pair_count is not None:
        pair_count = entry.get("pair_count")
        if not isinstance(pair_count, (int, float)) or int(pair_count) < config.min_pair_count:
            return False
    direction = str(config.direction or "").lower()
    if direction in {"lte", "le", "<="}:
        return float(value) <= config.threshold
    return float(value) >= config.threshold


def _fisher_z_ci(value: float, n: int, *, z_value: float) -> Optional[Tuple[float, float]]:
    if n <= 3:
        return None
    clamped = max(min(float(value), 0.999999), -0.999999)
    z_score = math.atanh(clamped)
    se = 1.0 / math.sqrt(n - 3)
    z_low = z_score - z_value * se
    z_high = z_score + z_value * se
    return math.tanh(z_low), math.tanh(z_high)


def _output_dir(
    output_root: Optional[Path],
    subfolder: str,
    category: Optional[Sequence[str]] = None,
) -> Path:
    if output_root is None:
        return results_path(
            "figures",
            subfolder,
            category,
            block_size=DEFAULT_BLOCK_SIZE,
        )
    path = Path(output_root) / subfolder
    if category:
        path = path.joinpath(*category)
    return path


def _iter_central_topic_reports(
    dashboard_root: Optional[Path] = None,
) -> Iterable[Tuple[Path, Dict[str, object]]]:
    root = dashboard_root or results_path("dashboard", block_size=DEFAULT_BLOCK_SIZE)
    if not root.exists():
        raise FileNotFoundError(f"Dashboard directory not found: {root}")
    for path in root.rglob(f"*{CENTRAL_TOPIC_CORRELATIONS_SUFFIX}"):
        if path.name == GENRE_CENTRAL_PRESENCE_FILENAME:
            continue
        payload = load_json(path)
        if isinstance(payload, dict):
            yield path, payload


def _load_aggregated_presence_by_genre(
    dashboard_root: Optional[Path] = None,
    selection: Optional[DataSelectionConfig] = None,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    root = dashboard_root or results_path("dashboard", block_size=DEFAULT_BLOCK_SIZE)
    if not root.exists():
        return {}
    aggregated: Dict[str, Dict[str, Dict[str, object]]] = {}
    for path in root.glob(f"*/{GENRE_CENTRAL_PRESENCE_FILENAME}"):
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        genre = payload.get("genre") or path.parent.name
        if not isinstance(genre, str) or not _matches_genre_selection(genre, selection):
            continue
        metrics = payload.get("metrics") or {}
        if isinstance(metrics, dict):
            aggregated[genre] = metrics
    return aggregated


def _load_split_half_stability_by_genre(
    dashboard_root: Optional[Path] = None,
    selection: Optional[DataSelectionConfig] = None,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    root = dashboard_root or results_path("dashboard", block_size=DEFAULT_BLOCK_SIZE)
    if not root.exists():
        return {}
    stability: Dict[str, Dict[str, Dict[str, object]]] = {}
    for path in root.glob(f"*/{GENRE_CENTRAL_TOPIC_SPLIT_HALF_FILENAME}"):
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        genre = payload.get("genre") or path.parent.name
        if not isinstance(genre, str) or not _matches_genre_selection(genre, selection):
            continue
        metrics = payload.get("metrics") or {}
        if isinstance(metrics, dict):
            stability[genre] = metrics
    return stability


def collect_central_topic_data(
    dashboard_root: Optional[Path] = None,
    selection: Optional[DataSelectionConfig] = None,
) -> Tuple[Dict[str, List[Dict[str, object]]], List[Dict[str, object]]]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    entries_by_genre: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    presence_entries: List[Dict[str, object]] = []

    for path, data in _iter_central_topic_reports(dashboard_root):
        metadata = data.get("metadata")
        report = data.get("report")
        params = data.get("params", {})
        if not isinstance(metadata, dict) or not isinstance(report, dict):
            continue
        if not isinstance(params, dict):
            params = {}
        window_count = report.get("window_count")
        category = metadata.get("category") or ""
        category_parts = [part for part in str(category).split("/") if part]
        genre = category_parts[0] if category_parts else "unknown"
        author = category_parts[1] if len(category_parts) > 1 else "unknown"
        text_name = metadata.get("text_name") or path.parent.name
        if not _matches_selection(genre, author, text_name, selection):
            continue
        topic_file = metadata.get("topic_file")
        centrality_order = report.get("centrality_metrics_order")
        if not isinstance(centrality_order, list):
            continue
        for topic in report.get("central_topics") or []:
            stats = _scores_to_stats(centrality_order, topic.get("raw_score"))
            correlations = topic.get("correlations") or {}
            for metric, corr in correlations.items():
                r = corr.get("pearson_r")
                if not isinstance(r, (int, float)):
                    continue
                n_windows = window_count if isinstance(window_count, (int, float)) else None
                entries_by_genre[genre].append(
                    {
                        "genre": genre,
                        "author": author,
                        "text_name": text_name,
                        "metric": metric,
                        "pearson_r": float(r),
                        "p_value": corr.get("p_value"),
                        "n_windows": n_windows,
                        "topic_id": topic.get("topic_id"),
                        "topic_score": topic.get("score"),
                        "keywords": topic.get("keywords") or [],
                        "stats": stats,
                        "raw_score": topic.get("raw_score"),
                        "percentile_score": topic.get("percentile_score"),
                        "topic_file": topic_file,
                        "params": params,
                    }
                )

        presence_path = path.with_name(
            path.name.replace(
                CENTRAL_TOPIC_CORRELATIONS_SUFFIX,
                CENTRAL_TOPIC_PRESENCE_CORRELATIONS_SUFFIX,
            )
        )
        if presence_path.exists():
            presence_payload = load_json(presence_path)
            if isinstance(presence_payload, dict):
                presence_report = presence_payload.get("report")
                if not isinstance(presence_report, dict):
                    continue
                presence_window_count = presence_report.get("window_count")
                presence = presence_report.get("central_topic_presence") or {}
                if presence.get("correlations"):
                    presence_entries.append(
                        {
                            "genre": genre,
                            "author": author,
                            "text_name": text_name,
                            "window_count": presence_window_count,
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
    dashboard_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> List[Path]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    if entries_by_genre is None:
        entries_by_genre, _ = collect_central_topic_data(
            dashboard_root=dashboard_root,
            selection=selection,
        )
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
        ax.set_title(f"Central Topic to Metric Correlations (Top {top_n}) - {genre}")

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

        output_dir = _output_dir(output_root, "bar_charts", [genre])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir / f"{genre}_central_topic_metric_correlations_top{top_n}.png"
        )
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def plot_central_topic_x_bars_stability_filtered(
    entries_by_genre: Optional[Dict[str, List[Dict[str, object]]]] = None,
    *,
    config: Optional[CentralTopicXBarConfig] = None,
    stability_config: Optional[StabilityFilterConfig] = None,
    top_n: Optional[int] = None,
    p_threshold: Optional[float] = None,
    selection: Optional[DataSelectionConfig] = None,
    dashboard_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> List[Path]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    if entries_by_genre is None:
        entries_by_genre, _ = collect_central_topic_data(
            dashboard_root=dashboard_root,
            selection=selection,
        )
    else:
        entries_by_genre = _filter_entries_by_genre(entries_by_genre, selection)

    if config is None:
        config = DEFAULT_CENTRAL_TOPIC_X_CONFIG
    if stability_config is None:
        stability_config = DEFAULT_STABILITY_FILTER_CONFIG
    if top_n is None:
        top_n = config.top_n
    if p_threshold is None:
        p_threshold = config.p_threshold

    stability_by_genre = _load_split_half_stability_by_genre(
        dashboard_root=dashboard_root,
        selection=selection,
    )

    genre_set = {genre for genre in entries_by_genre if genre}
    ordered_genres = [genre for genre in GENRES if genre in genre_set]
    ordered_genres += sorted(genre_set - set(ordered_genres))

    output_paths: List[Path] = []
    for genre in ordered_genres:
        entries = entries_by_genre.get(genre, [])
        stability_metrics = stability_by_genre.get(genre, {})
        if not entries or not stability_metrics:
            continue
        filtered_entries = []
        for entry in entries:
            metric = entry.get("metric")
            if not isinstance(metric, str):
                continue
            stability_entry = stability_metrics.get(metric)
            if _stability_passes(stability_entry, stability_config):
                filtered_entries.append(entry)

        top_entries = _top_n_by_abs_r(filtered_entries, top_n)
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
        direction = str(stability_config.direction or "").lower()
        comp = "<=" if direction in {"lte", "le", "<="} else ">="
        ax.set_title(
            f"Central Topic to Metric Correlations (Stable {stability_config.metric_key} {comp} "
            f"{stability_config.threshold:g}) - {genre}"
        )

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

        output_dir = _output_dir(output_root, "bar_charts_stability", [genre])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{genre}_central_topic_metric_correlations_stable_top{top_n}.png"
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
    dashboard_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> List[Path]:
    # Slopegraphs depend on binary with/without summaries, which are no longer produced.
    return []


def plot_aggregated_presence_heatmap(
    *,
    config: Optional[AggregatedHeatmapConfig] = None,
    p_threshold: Optional[float] = None,
    selection: Optional[DataSelectionConfig] = None,
    dashboard_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> Optional[Path]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    if config is None:
        config = DEFAULT_AGGREGATED_HEATMAP_CONFIG
    if p_threshold is None:
        p_threshold = config.p_threshold

    root = dashboard_root or results_path("dashboard", block_size=DEFAULT_BLOCK_SIZE)
    agg_paths = sorted(root.glob(f"*/{GENRE_CENTRAL_PRESENCE_FILENAME}"))
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
    ax.set_title("Aggregated All-Central-Topic (p-norm) Correlations - Genre x Metric (r)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r")
    plt.tight_layout()

    output_dir = _output_dir(output_root, "aggregated_heatmap")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "aggregated_genre_metric_heatmap.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_stability_family_counts(
    *,
    config: Optional[StabilityStackedBarConfig] = None,
    stability_config: Optional[StabilityFilterConfig] = None,
    selection: Optional[DataSelectionConfig] = None,
    dashboard_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> Optional[Path]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    if config is None:
        config = DEFAULT_STABILITY_STACKED_BAR_CONFIG
    if stability_config is None:
        stability_config = config.stability if config.stability else DEFAULT_STABILITY_FILTER_CONFIG

    stability_by_genre = _load_split_half_stability_by_genre(
        dashboard_root=dashboard_root,
        selection=selection,
    )
    if not stability_by_genre:
        return None

    genres = _ordered_genres(stability_by_genre.keys())
    family_order = list(config.family_order)
    counts_by_family = {family: [] for family in family_order}

    for genre in genres:
        metrics = stability_by_genre.get(genre, {})
        family_counts = {family: 0 for family in family_order}
        for metric, entry in metrics.items():
            if not isinstance(metric, str):
                continue
            family, _parts = _metric_parts(metric)
            if family not in family_counts:
                continue
            if _stability_passes(entry, stability_config):
                family_counts[family] += 1
        for family in family_order:
            counts_by_family[family].append(family_counts[family])

    if not genres:
        return None

    fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))
    x_pos = np.arange(len(genres))
    bottoms = np.zeros(len(genres))
    colors = list(config.family_colors)
    for idx, family in enumerate(family_order):
        values = counts_by_family[family]
        color = colors[idx % len(colors)] if colors else None
        ax.bar(
            x_pos,
            values,
            bottom=bottoms,
            color=color,
            alpha=config.bar_alpha,
            label=family,
        )
        bottoms = bottoms + np.array(values, dtype=float)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(genres, rotation=30, ha="right")
    ax.set_ylabel("Metrics above stability threshold")
    direction = str(stability_config.direction or "").lower()
    comp = "<=" if direction in {"lte", "le", "<="} else ">="
    ax.set_title(
        f"Stable metrics by family ({stability_config.metric_key} {comp} {stability_config.threshold:g})"
    )
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_dir = _output_dir(output_root, "stability_family_counts")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stability_family_counts.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def write_top_correlation_tables(
    entries_by_genre: Optional[Dict[str, List[Dict[str, object]]]] = None,
    presence_entries: Optional[List[Dict[str, object]]] = None,
    *,
    top_n: int = 10,
    render_png: bool = True,
    selection: Optional[DataSelectionConfig] = None,
    dashboard_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> Dict[str, List[Path]]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    if not isinstance(top_n, int) or top_n <= 0:
        top_n = 10

    if entries_by_genre is None or presence_entries is None:
        entries_by_genre, presence_entries = collect_central_topic_data(
            dashboard_root=dashboard_root,
            selection=selection,
        )
    else:
        entries_by_genre = _filter_entries_by_genre(entries_by_genre, selection)
        presence_entries = _filter_presence_entries(presence_entries, selection)

    aggregated = _load_aggregated_presence_by_genre(
        dashboard_root=dashboard_root,
        selection=selection,
    )

    presence_by_genre: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for entry in presence_entries:
        genre = entry.get("genre")
        if isinstance(genre, str):
            presence_by_genre[genre].append(entry)

    genre_set = set(entries_by_genre.keys()) | set(presence_by_genre.keys()) | set(aggregated.keys())
    ordered_genres = _ordered_genres(genre_set)

    output_paths: Dict[str, List[Path]] = {
        "central_topic_metrics": [],
        "central_topic_presence": [],
        "central_topic_presence_aggregated": [],
        "central_topic_metrics_png": [],
        "central_topic_presence_png": [],
        "central_topic_presence_aggregated_png": [],
    }

    topic_metric_columns = [
        "author",
        "text_name",
        "metric",
        "pearson_r",
        "p_value",
        "topic_id",
        "topic_exclusivity",
        "topic_coherence",
        "topic_prevalence",
        "topic_persistence",
        "n_windows",
    ]
    presence_columns = [
        "author",
        "text_name",
        "metric",
        "pearson_r",
        "p_value",
        "n_windows",
    ]
    aggregated_columns = [
        "metric",
        "pearson_r",
        "p_value",
        "text_count",
        "total_windows",
        "fisher_z",
    ]

    for genre in ordered_genres:
        topic_entries = entries_by_genre.get(genre, [])
        if topic_entries:
            top_entries = _top_n_by_abs_r(topic_entries, top_n)
            rows = []
            for entry in top_entries:
                stats = entry.get("stats") or {}
                rows.append(
                    {
                        "genre": genre,
                        "author": entry.get("author") or "",
                        "text_name": entry.get("text_name") or "",
                        "metric": entry.get("metric") or "",
                        "pearson_r": entry.get("pearson_r"),
                        "p_value": entry.get("p_value"),
                        "topic_id": entry.get("topic_id"),
                        "topic_score": entry.get("topic_score"),
                        "topic_exclusivity": stats.get("exclusivity"),
                        "topic_coherence": stats.get("coherence"),
                        "topic_prevalence": stats.get("prevalence"),
                        "topic_persistence": stats.get("persistence"),
                        "n_windows": entry.get("n_windows"),
                    }
                )
            output_dir = _output_dir(
                output_root, "correlation_tables", ["central_topic_metrics", genre]
            )
            output_path = (
                output_dir / f"{genre}_central_topic_metric_correlations_top{top_n}.tsv"
            )
            _write_tsv(output_path, rows, topic_metric_columns)
            output_paths["central_topic_metrics"].append(output_path)
            if render_png:
                png_path = output_path.with_suffix(".png")
                title = f"Central Topic to Metrics - Top {top_n} correlations by |r| ({genre})"
                rendered = _render_table_png(
                    png_path,
                    rows,
                    topic_metric_columns,
                    title=title,
                )
                if rendered:
                    output_paths["central_topic_metrics_png"].append(rendered)

        presence_entries_genre = presence_by_genre.get(genre, [])
        if presence_entries_genre:
            candidates: List[Dict[str, object]] = []
            for entry in presence_entries_genre:
                presence = entry.get("presence") or {}
                correlations = presence.get("correlations") or {}
                for metric, corr in correlations.items():
                    if not isinstance(corr, dict):
                        continue
                    r_val = corr.get("pearson_r")
                    if not isinstance(r_val, (int, float)):
                        continue
                    candidates.append(
                        {
                            "genre": genre,
                            "author": entry.get("author") or "",
                            "text_name": entry.get("text_name") or "",
                            "metric": metric,
                            "pearson_r": float(r_val),
                            "p_value": corr.get("p_value"),
                            "n_windows": (
                                entry.get("window_count")
                                if isinstance(entry.get("window_count"), (int, float))
                                else None
                            ),
                        }
                    )
            if candidates:
                top_entries = _top_n_by_abs_r(candidates, top_n)
                output_dir = _output_dir(
                    output_root, "correlation_tables", ["central_topic_presence", genre]
                )
                output_path = (
                    output_dir / f"{genre}_central_topic_presence_correlations_top{top_n}.tsv"
                )
                _write_tsv(output_path, top_entries, presence_columns)
                output_paths["central_topic_presence"].append(output_path)
                if render_png:
                    png_path = output_path.with_suffix(".png")
                    title = f"Central Topic Presence - Top {top_n} correlations by |r| ({genre})"
                    rendered = _render_table_png(
                        png_path,
                        top_entries,
                        presence_columns,
                        title=title,
                    )
                    if rendered:
                        output_paths["central_topic_presence_png"].append(rendered)

        aggregated_metrics = aggregated.get(genre, {})
        if isinstance(aggregated_metrics, dict) and aggregated_metrics:
            candidates = []
            for metric, corr in aggregated_metrics.items():
                if not isinstance(corr, dict):
                    continue
                r_val = corr.get("pearson_r")
                if not isinstance(r_val, (int, float)):
                    continue
                candidates.append(
                    {
                        "genre": genre,
                        "metric": metric,
                        "pearson_r": float(r_val),
                        "p_value": corr.get("p_value"),
                        "text_count": corr.get("text_count"),
                        "total_windows": corr.get("total_windows"),
                        "fisher_z": corr.get("fisher_z"),
                        "fisher_z_weight_sum": corr.get("fisher_z_weight_sum"),
                    }
                )
            if candidates:
                top_entries = _top_n_by_abs_r(candidates, top_n)
                output_dir = _output_dir(
                    output_root,
                    "correlation_tables",
                    ["central_topic_presence_aggregated", genre],
                )
                output_path = (
                    output_dir
                    / f"{genre}_central_topic_presence_aggregated_top{top_n}.tsv"
                )
                _write_tsv(output_path, top_entries, aggregated_columns)
                output_paths["central_topic_presence_aggregated"].append(output_path)
                if render_png:
                    png_path = output_path.with_suffix(".png")
                    title = f"Aggregated Central Topic Presence - Top {top_n} correlations by |r| ({genre})"
                    rendered = _render_table_png(
                        png_path,
                        top_entries,
                        aggregated_columns,
                        title=title,
                    )
                    if rendered:
                        output_paths["central_topic_presence_aggregated_png"].append(rendered)

    return output_paths


def plot_convergence_index(
    presence_entries: Optional[List[Dict[str, object]]] = None,
    *,
    config: Optional[ConvergenceIndexConfig] = None,
    metrics: Optional[Sequence[str]] = None,
    p_threshold: Optional[float] = None,
    selection: Optional[DataSelectionConfig] = None,
    dashboard_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> Optional[Path]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    if config is None:
        config = DEFAULT_CONVERGENCE_INDEX_CONFIG
    if metrics is None:
        metrics = config.metrics
    if p_threshold is None:
        p_threshold = config.p_threshold

    metric_keys = [metric for metric in metrics if isinstance(metric, str)]
    if not metric_keys:
        return None

    aggregated = _load_aggregated_presence_by_genre(
        dashboard_root=dashboard_root,
        selection=selection,
    )

    presence_by_genre: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    if "sign_agreement" in metric_keys:
        if presence_entries is None:
            _, presence_entries = collect_central_topic_data(
                dashboard_root=dashboard_root,
                selection=selection,
            )
        else:
            presence_entries = _filter_presence_entries(presence_entries, selection)
        for entry in presence_entries:
            genre = entry.get("genre")
            if isinstance(genre, str):
                presence_by_genre[genre].append(entry)

    genres = _ordered_genres(set(aggregated.keys()) | set(presence_by_genre.keys()))
    if not genres:
        return None

    series_by_metric: Dict[str, List[Optional[float]]] = {}
    for metric_key in metric_keys:
        values: List[Optional[float]] = []
        for genre in genres:
            value: Optional[float] = None
            if metric_key in {"significant_count", "mean_abs_r", "mean_abs_r_zeroed"}:
                corr_map = aggregated.get(genre, {})
                pairs = []
                for corr in corr_map.values():
                    if not isinstance(corr, dict):
                        continue
                    r_val = corr.get("pearson_r")
                    p_val = corr.get("p_value")
                    if isinstance(r_val, (int, float)):
                        p_val = float(p_val) if isinstance(p_val, (int, float)) else None
                        pairs.append((float(r_val), p_val))
                if pairs:
                    if metric_key == "significant_count":
                        value = float(
                            sum(1 for _, p_val in pairs if p_val is not None and p_val < p_threshold)
                        )
                    elif metric_key == "mean_abs_r":
                        vals = []
                        for r_val, p_val in pairs:
                            if config.zero_nonsignificant and (
                                p_val is None or p_val >= p_threshold
                            ):
                                vals.append(0.0)
                            else:
                                vals.append(abs(r_val))
                        if vals:
                            value = statistics.mean(vals)
                    elif metric_key == "mean_abs_r_zeroed":
                        vals = [
                            abs(r_val) if (p_val is not None and p_val < p_threshold) else 0.0
                            for r_val, p_val in pairs
                        ]
                        if vals:
                            value = statistics.mean(vals)
            elif metric_key == "sign_agreement":
                entries = presence_by_genre.get(genre, [])
                metric_signs: Dict[str, List[int]] = defaultdict(list)
                for entry in entries:
                    presence = entry.get("presence") or {}
                    correlations = presence.get("correlations") or {}
                    for metric_name, corr in correlations.items():
                        if not isinstance(corr, dict):
                            continue
                        r_val = corr.get("pearson_r")
                        p_val = corr.get("p_value")
                        if not isinstance(r_val, (int, float)):
                            continue
                        if config.sign_agreement_use_p_threshold:
                            if not isinstance(p_val, (int, float)) or p_val >= p_threshold:
                                continue
                        sign = 1 if r_val > 0 else -1 if r_val < 0 else 0
                        if sign != 0:
                            metric_signs[str(metric_name)].append(sign)
                total = 0
                consistent = 0
                for signs in metric_signs.values():
                    if len(signs) < config.sign_agreement_min_texts:
                        continue
                    total += 1
                    if len(set(signs)) == 1:
                        consistent += 1
                if total > 0:
                    value = consistent / total
            values.append(value)
        if any(val is not None for val in values):
            series_by_metric[metric_key] = values

    if not series_by_metric:
        return None

    fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))
    x_pos = np.arange(len(genres))
    for metric_key, values in series_by_metric.items():
        y_vals = [np.nan if v is None else v for v in values]
        label = CONVERGENCE_METRIC_LABELS.get(metric_key, metric_key)
        ax.plot(
            x_pos,
            y_vals,
            marker="o",
            linewidth=config.line_width,
            markersize=config.marker_size,
            label=label,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(genres, rotation=20, ha="right")
    if len(series_by_metric) == 1:
        label = next(iter(series_by_metric.keys()))
        ax.set_ylabel(CONVERGENCE_METRIC_LABELS.get(label, label))
        if label == "sign_agreement":
            ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylabel("Convergence index")
        ax.legend(loc="best", fontsize=8, frameon=False)
    ax.set_title("Convergence index across movements")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_dir = _output_dir(output_root, "convergence_index")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "convergence_index.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_forest_core_metrics(
    presence_entries: Optional[List[Dict[str, object]]] = None,
    *,
    config: Optional[ForestPlotConfig] = None,
    metrics: Optional[Sequence[str]] = None,
    p_threshold: Optional[float] = None,
    selection: Optional[DataSelectionConfig] = None,
    dashboard_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> List[Path]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    if presence_entries is None:
        _, presence_entries = collect_central_topic_data(
            dashboard_root=dashboard_root,
            selection=selection,
        )
    else:
        presence_entries = _filter_presence_entries(presence_entries, selection)

    if config is None:
        config = DEFAULT_FOREST_PLOT_CONFIG
    if metrics is None:
        metrics = config.metrics
    if p_threshold is None:
        p_threshold = config.p_threshold

    aggregated = _load_aggregated_presence_by_genre(
        dashboard_root=dashboard_root,
        selection=selection,
    )
    presence_by_genre: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for entry in presence_entries:
        genre = entry.get("genre")
        if isinstance(genre, str):
            presence_by_genre[genre].append(entry)

    output_paths: List[Path] = []
    for genre in _ordered_genres(presence_by_genre.keys()):
        entries = presence_by_genre.get(genre, [])
        if not entries:
            continue
        for metric in metrics:
            rows = []
            for entry in entries:
                presence = entry.get("presence") or {}
                correlations = presence.get("correlations") or {}
                corr = correlations.get(metric)
                if not isinstance(corr, dict):
                    continue
                r_val = corr.get("pearson_r")
                if not isinstance(r_val, (int, float)):
                    continue
                p_val = corr.get("p_value")
                n_val = entry.get("window_count")
                n_val = int(n_val) if isinstance(n_val, (int, float)) else None
                rows.append(
                    {
                        "author": entry.get("author") or "",
                        "text_name": entry.get("text_name") or "",
                        "r": float(r_val),
                        "p": float(p_val) if isinstance(p_val, (int, float)) else None,
                        "n_windows": n_val,
                    }
                )

            if not rows:
                continue

            rows.sort(key=lambda row: (row["author"], row["text_name"]))
            labels = []
            for row in rows:
                label = (
                    f"{row['author']}/{row['text_name']}"
                    if row["author"]
                    else row["text_name"]
                )
                labels.append(_shorten_label(label, config.label_max_len))

            agg_entry = aggregated.get(genre, {}).get(metric, {})
            agg_r = agg_entry.get("pearson_r") if isinstance(agg_entry, dict) else None
            agg_p = agg_entry.get("p_value") if isinstance(agg_entry, dict) else None
            agg_n = agg_entry.get("total_windows") if isinstance(agg_entry, dict) else None
            if not isinstance(agg_n, (int, float)):
                agg_n = sum(row["n_windows"] for row in rows if row["n_windows"])

            show_agg = isinstance(agg_r, (int, float))
            total_rows = len(rows) + (1 if show_agg else 0)
            fig_height = max(config.min_height, config.row_height * total_rows + 1.5)
            fig, ax = plt.subplots(figsize=(config.fig_width, fig_height))

            ax.axvline(0, color="black", linewidth=0.8)

            y_pos = np.arange(len(rows))
            for idx, row in enumerate(rows):
                r_val = row["r"]
                p_val = row["p"]
                n_val = row["n_windows"] or 0
                color = config.positive_color if r_val >= 0 else config.negative_color
                alpha = (
                    config.alpha_significant
                    if isinstance(p_val, (int, float)) and p_val < p_threshold
                    else config.alpha_nonsignificant
                )
                ci = _fisher_z_ci(r_val, n_val, z_value=config.ci_z) if n_val else None
                if ci:
                    ax.plot(
                        [ci[0], ci[1]],
                        [y_pos[idx], y_pos[idx]],
                        color=color,
                        alpha=alpha,
                        linewidth=config.line_width,
                    )
                ax.scatter(
                    [r_val],
                    [y_pos[idx]],
                    color=color,
                    alpha=alpha,
                    s=config.point_size,
                    zorder=3,
                )

            labels_out = list(labels)
            if show_agg:
                agg_r_val = float(agg_r)
                agg_n_val = int(agg_n) if isinstance(agg_n, (int, float)) else 0
                agg_p_val = float(agg_p) if isinstance(agg_p, (int, float)) else None
                agg_color = config.positive_color if agg_r_val >= 0 else config.negative_color
                agg_alpha = (
                    config.alpha_significant
                    if isinstance(agg_p_val, (int, float)) and agg_p_val < p_threshold
                    else config.alpha_nonsignificant
                )
                agg_ci = (
                    _fisher_z_ci(agg_r_val, agg_n_val, z_value=config.ci_z)
                    if agg_n_val
                    else None
                )
                agg_y = len(rows)
                if agg_ci:
                    ax.plot([agg_ci[0], agg_ci[1]], [agg_y, agg_y], color=agg_color, alpha=agg_alpha, linewidth=2.0)
                ax.scatter(
                    [agg_r_val],
                    [agg_y],
                    color=agg_color,
                    alpha=agg_alpha,
                    s=config.aggregate_size,
                    marker="D",
                    zorder=4,
                )
                labels_out.append("aggregate")
                y_ticks = np.arange(len(rows) + 1)
            else:
                y_ticks = y_pos

            ax.set_yticks(y_ticks)
            ax.set_yticklabels(labels_out, fontsize=8)
            ax.set_xlabel("Pearson r")
            ax.set_title(f"Central Topic Presence vs {metric} ({genre})")
            ax.grid(axis="x", linestyle="--", alpha=0.3)
            if config.xlim:
                ax.set_xlim(config.xlim[0], config.xlim[1])

            plt.tight_layout()
            output_dir = _output_dir(output_root, "forest_core_metrics", [genre])
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_metric = metric.replace(":", "_").replace("/", "_")
            output_path = output_dir / f"{genre}_forest_{safe_metric}.png"
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
            output_paths.append(output_path)

    return output_paths


def plot_text_metric_heatmaps(
    presence_entries: Optional[List[Dict[str, object]]] = None,
    *,
    config: Optional[TextMetricHeatmapConfig] = None,
    p_threshold: Optional[float] = None,
    selection: Optional[DataSelectionConfig] = None,
    dashboard_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> List[Path]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    if presence_entries is None:
        _, presence_entries = collect_central_topic_data(
            dashboard_root=dashboard_root,
            selection=selection,
        )
    else:
        presence_entries = _filter_presence_entries(presence_entries, selection)

    if config is None:
        config = DEFAULT_TEXT_METRIC_HEATMAP_CONFIG
    if p_threshold is None:
        p_threshold = config.p_threshold

    presence_by_genre: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for entry in presence_entries:
        genre = entry.get("genre")
        if isinstance(genre, str):
            presence_by_genre[genre].append(entry)

    output_paths: List[Path] = []
    for genre in _ordered_genres(presence_by_genre.keys()):
        entries = presence_by_genre.get(genre, [])
        if not entries:
            continue

        text_rows = []
        metrics_set = set()
        for entry in entries:
            presence = entry.get("presence") or {}
            correlations = presence.get("correlations") or {}
            if not correlations:
                continue
            author = str(entry.get("author") or "")
            text_name = str(entry.get("text_name") or "")
            label = f"{author}/{text_name}" if author else text_name
            text_rows.append(
                {
                    "author": author,
                    "text_name": text_name,
                    "label": _shorten_label(label, config.label_max_len),
                    "correlations": correlations,
                }
            )
            metrics_set.update(correlations.keys())

        if config.exclude_metrics:
            metrics_set = {metric for metric in metrics_set if metric not in config.exclude_metrics}

        if config.metrics:
            metrics_list = [metric for metric in config.metrics if metric in metrics_set]
        else:
            metrics_list = sorted(metrics_set)

        if not text_rows or not metrics_list:
            continue

        text_rows.sort(key=lambda row: (row["author"], row["text_name"]))

        r_matrix = np.full((len(text_rows), len(metrics_list)), np.nan)
        p_matrix = np.full((len(text_rows), len(metrics_list)), np.nan)
        for i, row in enumerate(text_rows):
            correlations = row["correlations"]
            for j, metric in enumerate(metrics_list):
                corr = correlations.get(metric)
                if not isinstance(corr, dict):
                    continue
                r_val = corr.get("pearson_r")
                p_val = corr.get("p_value")
                if isinstance(r_val, (int, float)):
                    r_matrix[i, j] = float(r_val)
                if isinstance(p_val, (int, float)):
                    p_matrix[i, j] = float(p_val)

        if config.metrics is None:
            variances = np.nanvar(r_matrix, axis=0)
            order = np.argsort(-np.nan_to_num(variances, nan=-np.inf))
            r_matrix = r_matrix[:, order]
            p_matrix = p_matrix[:, order]
            metrics_list = [metrics_list[idx] for idx in order]

        if config.top_n:
            metrics_list = metrics_list[: config.top_n]
            r_matrix = r_matrix[:, : config.top_n]
            p_matrix = p_matrix[:, : config.top_n]

        r_matrix_masked = r_matrix.copy()
        if p_threshold is not None:
            mask = ~(p_matrix < p_threshold)
            r_matrix_masked[mask] = np.nan

        if np.isfinite(r_matrix_masked).any():
            vmax = float(np.nanmax(np.abs(r_matrix_masked)))
        else:
            vmax = 0.1
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        fig, ax = plt.subplots(
            figsize=(
                max(config.min_width, len(metrics_list) * config.col_width),
                max(config.min_height, len(text_rows) * config.row_height),
            )
        )
        cmap = plt.get_cmap(config.cmap_name).copy()
        cmap.set_bad(color=config.mask_color)
        im = ax.imshow(r_matrix_masked, aspect="auto", cmap=cmap, norm=norm)

        ax.set_xticks(np.arange(len(metrics_list)))
        ax.set_xticklabels(metrics_list, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(text_rows)))
        ax.set_yticklabels([row["label"] for row in text_rows], fontsize=8)
        ax.set_title(f"Text x Metric (Central Topic Presence) - {genre}")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Pearson r")
        plt.tight_layout()

        output_dir = _output_dir(output_root, "text_metric_heatmaps", [genre])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{genre}_text_metric_heatmap.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def _select_window_entries(window_data: Dict[str, object]) -> List[Dict[str, object]]:
    for domain in ("syntax", "lexico_semantics", "discourse", "log_prob"):
        domain_block = window_data.get(domain, {}) if isinstance(window_data, dict) else {}
        windows = domain_block.get("windows") if isinstance(domain_block, dict) else None
        if isinstance(windows, list) and windows:
            return windows
    return []


def plot_central_topic_window_heatmaps(
    *,
    dashboard_root: Optional[Path] = None,
    config: Optional[CentralTopicWindowHeatmapConfig] = None,
    selection: Optional[DataSelectionConfig] = None,
    output_root: Optional[Path] = None,
) -> List[Path]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    if config is None:
        config = DEFAULT_CENTRAL_TOPIC_WINDOW_HEATMAP_CONFIG

    output_paths: List[Path] = []

    for path, data in _iter_central_topic_reports(dashboard_root):
        metadata = data.get("metadata")
        report = data.get("report")
        params = data.get("params", {})
        if not isinstance(metadata, dict) or not isinstance(report, dict):
            continue
        if not isinstance(params, dict):
            params = {}
        category = metadata.get("category") or ""
        category_parts = [part for part in str(category).split("/") if part]
        genre = category_parts[0] if category_parts else "unknown"
        author = category_parts[1] if len(category_parts) > 1 else "unknown"
        text_name = metadata.get("text_name") or path.parent.name
        if not _matches_selection(genre, author, text_name, selection):
            continue

        central_topics = report.get("central_topics") or []
        if not central_topics:
            continue

        topic_ids: List[int] = []
        labels: List[str] = []
        for topic in central_topics:
            topic_id = topic.get("topic_id")
            if not isinstance(topic_id, int):
                continue
            topic_ids.append(topic_id)
            label = str(topic_id)
            if config.show_keywords:
                gloss = _shorten_keywords(topic.get("keywords") or [], config.label_max_len)
                if gloss:
                    label = f"{topic_id}: {gloss}"
            labels.append(label)
        if not topic_ids:
            continue

        topic_path = _resolve_topic_path(data)
        if topic_path is None:
            window_metrics_path = _window_metrics_path(genre, author, text_name)
            if window_metrics_path.exists():
                topic_path = find_topic_file(window_metrics_path)
        if topic_path is None or not topic_path.exists():
            continue
        topics_data = load_json(topic_path)
        if not isinstance(topics_data, dict):
            continue

        window_data = _load_window_data(genre, author, text_name)
        if not window_data:
            continue
        window_entries = _select_window_entries(window_data)
        if not window_entries:
            continue

        scores_by_window = collect_window_topic_scores(
            topics_data,
            window_entries=window_entries,
        )
        if not scores_by_window:
            continue

        matrix = np.array(
            [
                [scores.get(topic_id, 0.0) for scores in scores_by_window]
                for topic_id in topic_ids
            ],
            dtype=float,
        )
        if matrix.size == 0:
            continue

        fig, ax = plt.subplots(
            figsize=(
                max(config.min_width, len(scores_by_window) * config.col_width),
                max(config.min_height, len(topic_ids) * config.row_height),
            )
        )
        cmap = plt.get_cmap(config.cmap_name).copy()
        if config.mask_color:
            cmap.set_bad(color=config.mask_color)
        vmax = config.vmax
        if vmax is None:
            vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else config.vmin
        if vmax <= config.vmin:
            vmax = config.vmin + 1e-6
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=config.vmin, vmax=vmax)

        window_count = len(scores_by_window)
        step = max(1, math.ceil(window_count / max(1, config.max_xticks)))
        x_ticks = np.arange(0, window_count, step)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(int(x_tick)) for x_tick in x_ticks])
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Window index")
        ax.set_ylabel("Central topic")
        ax.set_title(f"Central Topic Presence (Soft Scores) - {text_name} ({genre})")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Soft score")
        plt.tight_layout()

        output_dir = _output_dir(output_root, "central_topic_window_heatmaps", [genre, author])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{text_name}_central_topic_window_heatmap.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def plot_topic_metric_family_lines(
    *,
    config: Optional[TopicMetricLineConfig] = None,
    selection: Optional[DataSelectionConfig] = None,
    dashboard_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> List[Path]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    if config is None:
        config = DEFAULT_TOPIC_METRIC_LINE_CONFIG

    output_paths: List[Path] = []
    for path, data in _iter_central_topic_reports(dashboard_root):
        metadata = data.get("metadata")
        report = data.get("report")
        if not isinstance(metadata, dict) or not isinstance(report, dict):
            continue
        category = metadata.get("category") or ""
        category_parts = [part for part in str(category).split("/") if part]
        genre = category_parts[0] if category_parts else "unknown"
        author = category_parts[1] if len(category_parts) > 1 else "unknown"
        text_name = metadata.get("text_name") or path.parent.name
        if not _matches_selection(genre, author, text_name, selection):
            continue

        central_topics = report.get("central_topics") or []
        if not central_topics:
            continue
        valid_topics = []
        for topic in central_topics:
            if not isinstance(topic, dict):
                continue
            topic_id = topic.get("topic_id")
            if not isinstance(topic_id, int):
                continue
            correlations = topic.get("correlations")
            if not isinstance(correlations, dict) or not correlations:
                continue
            valid_topics.append(topic)
        if not valid_topics:
            continue

        topic_path = _resolve_topic_path(data)
        if topic_path is None:
            window_metrics_path = _window_metrics_path(genre, author, text_name)
            if window_metrics_path.exists():
                topic_path = find_topic_file(window_metrics_path)
        if topic_path is None or not topic_path.exists():
            continue

        topics_data = load_json(topic_path)
        if not isinstance(topics_data, dict):
            continue

        window_data = _load_window_data(genre, author, text_name)
        if not window_data:
            continue
        window_entries = _select_window_entries(window_data)
        if not window_entries:
            continue
        window_table = collect_window_tables(window_data)
        if not window_table:
            continue

        scores_by_window = collect_window_topic_scores(
            topics_data,
            window_entries=window_entries,
        )
        if not scores_by_window:
            continue

        if len(scores_by_window) != len(window_table):
            continue
        count = len(window_table)
        if count < 2:
            continue

        for family in config.families:
            prefix = f"{family}."
            best_topic = None
            best_r = None
            for topic in valid_topics:
                correlations = topic.get("correlations") or {}
                best_for_topic = None
                for metric, corr in correlations.items():
                    if not isinstance(metric, str) or not metric.startswith(prefix):
                        continue
                    r_val = corr.get("pearson_r")
                    if not isinstance(r_val, (int, float)):
                        continue
                    if config.p_threshold is not None:
                        p_val = corr.get("p_value")
                        if not isinstance(p_val, (int, float)) or p_val >= config.p_threshold:
                            continue
                    abs_r = abs(float(r_val))
                    if best_for_topic is None or abs_r > best_for_topic:
                        best_for_topic = abs_r
                if best_for_topic is None:
                    continue
                if best_r is None or best_for_topic > best_r:
                    best_r = best_for_topic
                    best_topic = topic

            if best_topic is None:
                continue

            correlations = best_topic.get("correlations") or {}
            if not isinstance(correlations, dict) or not correlations:
                continue

            topic_id = best_topic.get("topic_id")
            if not isinstance(topic_id, int):
                continue
            topic_scores = [scores_by_window[idx].get(topic_id, 0.0) for idx in range(count)]
            if config.normalize:
                topic_scores = _normalize_series(topic_scores, config.normalization)

            candidates = []
            for metric, corr in correlations.items():
                if not isinstance(metric, str) or not metric.startswith(prefix):
                    continue
                r_val = corr.get("pearson_r")
                if not isinstance(r_val, (int, float)):
                    continue
                if config.p_threshold is not None:
                    p_val = corr.get("p_value")
                    if not isinstance(p_val, (int, float)) or p_val >= config.p_threshold:
                        continue
                candidates.append((metric, abs(float(r_val))))

            if not candidates:
                continue
            candidates.sort(key=lambda row: row[1], reverse=True)
            if config.top_n_metrics and config.top_n_metrics > 0:
                candidates = candidates[: config.top_n_metrics]

            series_map: Dict[str, List[float]] = {}
            for metric, _abs_r in candidates:
                series = []
                valid = True
                for idx in range(count):
                    value = window_table[idx].get(metric)
                    if not isinstance(value, (int, float)) or (
                        isinstance(value, float) and math.isnan(value)
                    ):
                        valid = False
                        break
                    series.append(float(value))
                if not valid:
                    continue
                if config.normalize:
                    series = _normalize_series(series, config.normalization)
                series_map[metric] = series

            if not series_map:
                continue

            fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))
            x_vals = np.arange(count)
            topic_label = f"topic {topic_id} soft score"
            if config.normalize:
                topic_label = f"topic {topic_id} (normalized)"
            ax.plot(
                x_vals,
                topic_scores,
                color=config.topic_color,
                linewidth=config.topic_line_width,
                alpha=config.line_alpha,
                label=topic_label,
            )

            colors = list(config.metric_colors)
            for idx, (metric, series) in enumerate(series_map.items()):
                color = colors[idx % len(colors)] if colors else None
                label = _shorten_label(metric, config.label_max_len)
                ax.plot(
                    x_vals,
                    series,
                    color=color,
                    linewidth=config.metric_line_width,
                    alpha=config.line_alpha,
                    label=label,
                )

            step = max(1, math.ceil(count / max(1, config.max_xticks)))
            x_ticks = np.arange(0, count, step)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(int(x_tick)) for x_tick in x_ticks])
            ax.set_xlabel("Window index")
            if config.normalize:
                norm_label = (
                    "Z-score"
                    if str(config.normalization or "").lower() == "zscore"
                    else "Normalized value"
                )
                ax.set_ylabel(norm_label)
            else:
                ax.set_ylabel("Metric value")
            gloss = _shorten_keywords(best_topic.get("keywords") or [], config.label_max_len)
            topic_title = f"topic {topic_id}" + (f": {gloss}" if gloss else "")
            ax.set_title(f"{text_name} ({genre}) - {topic_title} vs {family}")
            ax.legend(loc="best", fontsize=8, frameon=False)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            plt.tight_layout()
            output_dir = _output_dir(output_root, "topic_metric_lines", [genre, author])
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{text_name}__topic{topic_id}__{family}_lines.png"
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
            output_paths.append(output_path)

    return output_paths


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


def _normalize_series(values: Sequence[float], method: str) -> List[float]:
    if not values:
        return []
    arr = np.array(values, dtype=float)
    method_key = (method or "zscore").lower()
    if method_key == "minmax":
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if vmax <= vmin:
            return [0.0 for _ in values]
        return ((arr - vmin) / (vmax - vmin)).tolist()
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std <= 0.0:
        return [0.0 for _ in values]
    return ((arr - mean) / std).tolist()


def _resolve_topic_path(entry: Dict[str, object]) -> Optional[Path]:
    """Resolve the topic_file path in a dashboard entry, returning None if missing or invalid."""
    raw_path = entry.get("topic_file")
    if not raw_path:
        metadata = entry.get("metadata")
        if isinstance(metadata, dict):
            raw_path = metadata.get("topic_file")
    if not raw_path:
        return None
    candidate = Path(str(raw_path))
    if candidate.exists():
        return candidate
    candidate = Path(str(raw_path).replace("\\", "/"))
    return candidate if candidate.exists() else None


def _window_metrics_path(genre: str, author: str, text_name: str) -> Path:
    return analytics_path(
        "window",
        [genre, author, text_name],
        f"{text_name}_window_metrics.syntax.json",
    )


def _load_window_data(
    genre: str,
    author: str,
    text_name: str,
) -> Optional[Dict[str, object]]:
    window_metrics_path = _window_metrics_path(genre, author, text_name)
    if not window_metrics_path.exists():
        return None
    return load_window_metrics(window_metrics_path)


def plot_exemplar_scatter(
    entries_by_genre: Optional[Dict[str, List[Dict[str, object]]]] = None,
    *,
    config: Optional[ExemplarScatterConfig] = None,
    top_per_genre: Optional[int] = None,
    min_points: Optional[int] = None,
    selection: Optional[DataSelectionConfig] = None,
    dashboard_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> List[Path]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    if entries_by_genre is None:
        entries_by_genre, _ = collect_central_topic_data(
            dashboard_root=dashboard_root,
            selection=selection,
        )
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

            window_data = _load_window_data(genre, author, text_name)
            if not window_data:
                continue
            domain_windows = window_data.get(domain, {}).get("windows") or []
            if not domain_windows:
                continue

            scores_by_window = collect_window_topic_scores(
                topics_data,
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
            ax.set_title(f"{text_name} ({genre}) - topic {topic_id} vs {metric}")
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label("Narrative position (window index)")

            plt.tight_layout()
            output_dir = _output_dir(output_root, "scatter_exemplars", [genre])
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
    convergence_p_threshold: Optional[float] = None,
    forest_p_threshold: Optional[float] = None,
    text_heatmap_p_threshold: Optional[float] = None,
    a_config: Optional[CentralTopicXBarConfig] = None,
    b_config: Optional[CentralTopicWindowHeatmapConfig] = None,
    c_config: Optional[ExemplarScatterConfig] = None,
    e_config: Optional[PresenceSlopegraphConfig] = None,
    f_config: Optional[AggregatedHeatmapConfig] = None,
    g_config: Optional[ConvergenceIndexConfig] = None,
    h_config: Optional[ForestPlotConfig] = None,
    i_config: Optional[TextMetricHeatmapConfig] = None,
    stability_config: Optional[StabilityFilterConfig] = None,
    stability_bar_config: Optional[CentralTopicXBarConfig] = None,
    stability_stack_config: Optional[StabilityStackedBarConfig] = None,
    topic_line_config: Optional[TopicMetricLineConfig] = None,
    selection: Optional[DataSelectionConfig] = None,
    dashboard_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
) -> Dict[str, object]:
    if selection is None:
        selection = DEFAULT_DATA_SELECTION_CONFIG
    entries_by_genre, presence_entries = collect_central_topic_data(
        dashboard_root=dashboard_root,
        selection=selection,
    )

    if a_config is None:
        a_config = DEFAULT_CENTRAL_TOPIC_X_CONFIG
    if top_n is not None:
        a_config = replace(a_config, top_n=top_n)
    if bar_p_threshold is not None:
        a_config = replace(a_config, p_threshold=bar_p_threshold)

    if b_config is None:
        b_config = DEFAULT_CENTRAL_TOPIC_WINDOW_HEATMAP_CONFIG

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

    if g_config is None:
        g_config = DEFAULT_CONVERGENCE_INDEX_CONFIG
    if convergence_p_threshold is not None:
        g_config = replace(g_config, p_threshold=convergence_p_threshold)

    if h_config is None:
        h_config = DEFAULT_FOREST_PLOT_CONFIG
    if forest_p_threshold is not None:
        h_config = replace(h_config, p_threshold=forest_p_threshold)

    if i_config is None:
        i_config = DEFAULT_TEXT_METRIC_HEATMAP_CONFIG
    if text_heatmap_p_threshold is not None:
        i_config = replace(i_config, p_threshold=text_heatmap_p_threshold)

    if stability_config is None:
        stability_config = DEFAULT_STABILITY_FILTER_CONFIG

    if stability_bar_config is None:
        stability_bar_config = DEFAULT_CENTRAL_TOPIC_X_CONFIG
    if top_n is not None:
        stability_bar_config = replace(stability_bar_config, top_n=top_n)
    if bar_p_threshold is not None:
        stability_bar_config = replace(stability_bar_config, p_threshold=bar_p_threshold)

    if stability_stack_config is None:
        stability_stack_config = DEFAULT_STABILITY_STACKED_BAR_CONFIG
    stability_stack_config = replace(stability_stack_config, stability=stability_config)

    if topic_line_config is None:
        topic_line_config = DEFAULT_TOPIC_METRIC_LINE_CONFIG

    results = {
        "A": plot_central_topic_x_bars(
            entries_by_genre,
            config=a_config,
            selection=selection,
            dashboard_root=dashboard_root,
            output_root=output_root,
        ),
        "B": plot_central_topic_window_heatmaps(
            config=b_config,
            selection=selection,
            dashboard_root=dashboard_root,
            output_root=output_root,
        ),
        "C": plot_exemplar_scatter(
            entries_by_genre,
            config=c_config,
            selection=selection,
            dashboard_root=dashboard_root,
            output_root=output_root,
        ),
        "E": plot_presence_slopegraphs(
            presence_entries,
            config=e_config,
            selection=selection,
            dashboard_root=dashboard_root,
            output_root=output_root,
        ),
        "F": plot_aggregated_presence_heatmap(
            config=f_config,
            selection=selection,
            dashboard_root=dashboard_root,
            output_root=output_root,
        ),
        "G": plot_convergence_index(
            presence_entries,
            config=g_config,
            selection=selection,
            dashboard_root=dashboard_root,
            output_root=output_root,
        ),
        "H": plot_forest_core_metrics(
            presence_entries,
            config=h_config,
            selection=selection,
            dashboard_root=dashboard_root,
            output_root=output_root,
        ),
        "I": plot_text_metric_heatmaps(
            presence_entries,
            config=i_config,
            selection=selection,
            dashboard_root=dashboard_root,
            output_root=output_root,
        ),
        "J": write_top_correlation_tables(
            entries_by_genre,
            presence_entries,
            top_n=a_config.top_n,
            selection=selection,
            dashboard_root=dashboard_root,
            output_root=output_root,
        ),
        "K": plot_central_topic_x_bars_stability_filtered(
            entries_by_genre,
            config=stability_bar_config,
            stability_config=stability_config,
            selection=selection,
            dashboard_root=dashboard_root,
            output_root=output_root,
        ),
        "L": plot_stability_family_counts(
            config=stability_stack_config,
            stability_config=stability_config,
            selection=selection,
            dashboard_root=dashboard_root,
            output_root=output_root,
        ),
        "M": plot_topic_metric_family_lines(
            config=topic_line_config,
            selection=selection,
            dashboard_root=dashboard_root,
            output_root=output_root,
        ),
    }
    return results


def generate_all_visualisations_with_config(
    *,
    config: DashboardCorrelationConfig = DEFAULT_DASHBOARD_CORRELATION_CONFIG,
    output_root_template: str = "figures_L{block_size}",
    top_n: Optional[int] = None,
    bar_p_threshold: Optional[float] = None,
    slope_p_threshold: Optional[float] = None,
    heatmap_p_threshold: Optional[float] = None,
    convergence_p_threshold: Optional[float] = None,
    forest_p_threshold: Optional[float] = None,
    text_heatmap_p_threshold: Optional[float] = None,
    a_config: Optional[CentralTopicXBarConfig] = None,
    b_config: Optional[CentralTopicWindowHeatmapConfig] = None,
    c_config: Optional[ExemplarScatterConfig] = None,
    e_config: Optional[PresenceSlopegraphConfig] = None,
    f_config: Optional[AggregatedHeatmapConfig] = None,
    g_config: Optional[ConvergenceIndexConfig] = None,
    h_config: Optional[ForestPlotConfig] = None,
    i_config: Optional[TextMetricHeatmapConfig] = None,
    stability_config: Optional[StabilityFilterConfig] = None,
    stability_bar_config: Optional[CentralTopicXBarConfig] = None,
    stability_stack_config: Optional[StabilityStackedBarConfig] = None,
    topic_line_config: Optional[TopicMetricLineConfig] = None,
    selection: Optional[DataSelectionConfig] = None,
) -> Dict[int, Dict[str, object]]:
    if not config.loop_enabled:
        results = generate_all_visualisations(
            top_n=top_n,
            bar_p_threshold=bar_p_threshold,
            slope_p_threshold=slope_p_threshold,
            heatmap_p_threshold=heatmap_p_threshold,
            convergence_p_threshold=convergence_p_threshold,
            forest_p_threshold=forest_p_threshold,
            text_heatmap_p_threshold=text_heatmap_p_threshold,
            a_config=a_config,
            b_config=b_config,
            c_config=c_config,
            e_config=e_config,
            f_config=f_config,
            g_config=g_config,
            h_config=h_config,
            i_config=i_config,
            stability_config=stability_config,
            stability_bar_config=stability_bar_config,
            stability_stack_config=stability_stack_config,
            topic_line_config=topic_line_config,
            selection=selection,
            dashboard_root=results_path("dashboard", block_size=config.block_size),
            output_root=results_path("figures", block_size=config.block_size),
        )
        return {config.block_size: results}

    block_sizes = list(config.loop_block_sizes or ())
    if not block_sizes:
        block_sizes = [config.block_size]

    results_by_block: Dict[int, Dict[str, object]] = {}
    for block_size in block_sizes:
        dashboard_root = Path("data") / "results" / config.loop_output_template.format(
            block_size=block_size
        )
        if not dashboard_root.exists():
            print(
                f"Skipping figures for block_size={block_size}; dashboard root missing: "
                f"{dashboard_root}"
            )
            continue
        output_root = Path("data") / "results" / output_root_template.format(
            block_size=block_size
        )
        results_by_block[block_size] = generate_all_visualisations(
            top_n=top_n,
            bar_p_threshold=bar_p_threshold,
            slope_p_threshold=slope_p_threshold,
            heatmap_p_threshold=heatmap_p_threshold,
            convergence_p_threshold=convergence_p_threshold,
            forest_p_threshold=forest_p_threshold,
            text_heatmap_p_threshold=text_heatmap_p_threshold,
            a_config=a_config,
            b_config=b_config,
            c_config=c_config,
            e_config=e_config,
            f_config=f_config,
            g_config=g_config,
            h_config=h_config,
            i_config=i_config,
            stability_config=stability_config,
            stability_bar_config=stability_bar_config,
            stability_stack_config=stability_stack_config,
            topic_line_config=topic_line_config,
            selection=selection,
            dashboard_root=dashboard_root,
            output_root=output_root,
        )
    return results_by_block


if __name__ == "__main__":
    generate_all_visualisations_with_config()
