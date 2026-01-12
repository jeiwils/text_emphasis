"""Heatmap plotting utilities for topic metric reports."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
def plot_topic_metric_heatmap(
    report: Dict[str, object],
    output_path,
    value_key: str = "variance_delta",
    min_windows: int = 2,
    top_n: Optional[int] = None,
):
    """Render a heatmap for a selected topic/metric statistic."""
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
    ax.set_title(f"Topic metric heatmap ({value_key})")
    fig.colorbar(im, ax=ax)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
