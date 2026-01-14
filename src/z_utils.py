

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import json
import numpy as np
from statistics import mean
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN






def _category_parts(category: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if category is None:
        return []
    if isinstance(category, (list, tuple)):
        return [str(part) for part in category if part]
    category_str = str(category)
    return [part for part in category_str.replace("\\", "/").split("/") if part]


def text_path(
    kind: str,
    subfolder: Optional[str] = None,
    category: Optional[Union[str, Sequence[str]]] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Unified helper for text storage under data/texts.
    kind: "raw" or "processed".
    """
    base = Path("data") / "texts"
    if kind == "raw":
        path = base / "raw"
    elif kind == "processed":
        path = base / "processed"
        if subfolder:
            path = path / subfolder
    else:
        raise ValueError('kind must be "raw" or "processed"')

    category_parts = _category_parts(category)
    if category_parts:
        path = path.joinpath(*category_parts)
    if filename:
        path = path / filename
    return path



def analytics_path(
    kind: str,
    category: Optional[Union[str, Sequence[str]]] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Unified helper for analytics outputs under data/analytics.
    kind: "corpus", "window", "topic", or "dashboard".
    """
    base = Path("data") / "analytics"
    folder_map = {
        "corpus": base / "corpus_analytics",
        "window": base / "window_metrics",
        "topic": base / "topic_modelling",
        "dashboard": base / "dashboard",
    }
    if kind not in folder_map:
        raise ValueError(f"kind must be one of {list(folder_map.keys())}")
    path = folder_map[kind]
    category_parts = _category_parts(category)
    if category_parts:
        path = path.joinpath(*category_parts)
    if filename:
        path = path / filename
    return path


def figures_path(
    subfolder: Optional[str] = None,
    category: Optional[Union[str, Sequence[str]]] = None,
    filename: Optional[str] = None,
) -> Path:
    """
    Unified helper for visualization outputs under analytics/figures.
    """
    path = Path("analytics") / "figures"
    if subfolder:
        path = path / subfolder
    category_parts = _category_parts(category)
    if category_parts:
        path = path.joinpath(*category_parts)
    if filename:
        path = path / filename
    return path


def iter_category_dirs(root: Path) -> Iterable[Tuple[str, Path]]:
    """
    Yield (category_key, dir_path) pairs for leaf category directories.
    Supports either <root>/<category>/... or <root>/<group>/<author>/... layouts.
    """
    if not root.exists():
        return
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if any(child.is_file() for child in entry.iterdir()):
            yield entry.name, entry
            continue
        for child in entry.iterdir():
            if child.is_dir():
                if any(grandchild.is_file() for grandchild in child.iterdir()):
                    category_key = f"{entry.name}/{child.name}"
                    yield category_key, child


def iter_genre_author_dirs(
    root: Path,
    genres: Sequence[str],
    authors: Optional[Sequence[str]] = None,
) -> Iterable[Tuple[str, str, Path]]:
    """
    Yield (genre, author, dir_path) for hardcoded genre roots.
    If authors is provided, only those author folders are used.
    """
    for genre in genres:
        genre_dir = root / genre
        if not genre_dir.exists():
            continue
        if authors:
            author_dirs = [genre_dir / author for author in authors]
        else:
            author_dirs = [path for path in genre_dir.iterdir() if path.is_dir()]
        for author_dir in author_dirs:
            if author_dir.is_dir():
                yield genre, author_dir.name, author_dir



def embeddings_path(
    embedding_type: str,
    filename: Optional[str] = None,
) -> Path:
    """

    
    """
    base_dir = "data/embeddings"

    folder_map = {
        "concept": "concept_embeddings",
        "passage": "passage_embeddings",
    }

    if embedding_type not in folder_map:
        raise ValueError(f"embedding_type must be one of {list(folder_map.keys())}")

    path = Path(base_dir) / folder_map[embedding_type]

    if filename:
        path = path / filename

    return path



def graph_path(
    graph_type: str,
    subfolder: Optional[str] = None,
    filename: Optional[str] = None,
) -> Path:
    """

    """
    base_dir = "data/graphs"

    folder_map = {
        "network": "network_analysis",
        "syntactic": "syntactic_graphs",
    }

    if graph_type not in folder_map:
        raise ValueError(f"graph_type must be one of {list(folder_map.keys())}")

    path = Path(base_dir) / folder_map[graph_type]
    if subfolder:
        path = path / subfolder
    if filename:
        path = path / filename
    return path


















def sliding_windows(seq, n, step: int = 1):
    """
    Sliding windows of width `n` with stride `step` (default 1).
    """
    seq = list(seq)
    if n <= 0:
        raise ValueError("window size must be positive")
    if step <= 0:
        raise ValueError("step must be a positive integer")
    if len(seq) < n:
        yield seq
        return
    for i in range(0, len(seq) - n + 1, step):
        yield seq[i : i + n]


def aggregate_windows(sent_metrics, window_size):
    """
    Aggregate sentence-level metrics over sliding windows of sentences.
    Returns a flat list of dicts with averaged numeric values per window.
    Each window includes 'start_sentence' and 'end_sentence'. Raw text is not preserved.
    """
    windows = []
    if not sent_metrics:
        return windows
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    for i, window_sents in enumerate(sliding_windows(sent_metrics, window_size)):
        agg = {}

        all_keys = set()
        for sent in window_sents:
            all_keys.update(sent.keys())

        for key in all_keys:
            if key in {"sentence_text", "sentences"}:
                # skip raw text emission
                continue
            dict_values = [d[key] for d in window_sents if isinstance(d.get(key), dict)]
            if dict_values:
                # Average numeric values in nested dicts.
                agg[key] = {}
                all_inner_keys = set(k for d in dict_values for k in d.keys())
                for k in all_inner_keys:
                    nums = [
                        d[k]
                        for d in dict_values
                        if k in d and isinstance(d[k], (int, float))
                    ]
                    agg[key][k] = round(mean(nums), 2) if nums else 0
                continue

            nums = [
                d.get(key)
                for d in window_sents
                if isinstance(d.get(key), (int, float))
            ]
            if nums:
                # Average numeric values, ignoring None.
                agg[key] = round(mean(nums), 2)
            else:
                # Keep first non-None non-numeric value.
                first_value = next((d.get(key) for d in window_sents if d.get(key) is not None), None)
                agg[key] = first_value

        # Add window metadata
        agg["start_sentence"] = i
        agg["end_sentence"] = i + len(window_sents) - 1  # correct end index for partial windows

        windows.append(agg)

    return windows








def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)





def encode_texts(
    encoder: SentenceTransformer,
    texts: Sequence[str],
    normalize: bool = True,
) -> np.ndarray:
    """Encode texts into embeddings using a shared encoder."""
    if not texts:
        dim = encoder.get_sentence_embedding_dimension()
        return np.empty((0, dim))
    embeddings = encoder.encode(list(texts))
    if normalize:
        return l2_normalize_embeddings(embeddings)
    return embeddings


def l2_normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings row-wise, keeping zero vectors unchanged."""
    if embeddings.size == 0:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def hdbscan_cluster_labels(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
) -> np.ndarray:
    """Cluster embeddings with HDBSCAN and return labels."""
    if embeddings is None or len(embeddings) == 0:
        return np.array([], dtype=int)
    if len(embeddings) < min_cluster_size:
        return np.full(len(embeddings), -1, dtype=int)
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        copy=False,  # keep embeddings in-place to silence sklearn future warning
    )
    return clusterer.fit_predict(embeddings)


def labels_to_clusters(labels: Sequence[int]) -> Dict[int, List[int]]:
    """Convert cluster labels to index lists, skipping noise (-1)."""
    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(idx)
    return clusters
