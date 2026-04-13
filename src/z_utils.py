

from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import json
import numpy as np
import spacy
from statistics import mean
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN

from .x_configs import (
    DEFAULT_METRIC_WINDOW_STRIDE,
    DEFAULT_SPACY_DISABLE,
    DEFAULT_SPACY_MAX_LENGTH,
    DEFAULT_SPACY_MODEL,
)


@lru_cache(maxsize=None)
def load_spacy_model(
    model_name: str = DEFAULT_SPACY_MODEL,
    disable: Optional[Sequence[str]] = None,
):
    """Load and cache a spaCy pipeline using the shared project defaults."""
    disable_components = tuple(disable) if disable else DEFAULT_SPACY_DISABLE
    nlp = spacy.load(model_name, disable=list(disable_components))
    nlp.max_length = max(getattr(nlp, "max_length", 1_000_000), DEFAULT_SPACY_MAX_LENGTH)
    return nlp


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
    kind: "corpus", "window", "topic", or "embeddings".
    """
    base = Path("data") / "analytics"
    folder_map = {
        "corpus": base / "corpus_analytics",
        "window": base / "window_metrics",
        "topic": base / "topic_modelling",
        "embeddings": base / "embeddings",
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


def results_path(
    kind: str,
    subfolder: Optional[str] = None,
    category: Optional[Union[str, Sequence[str]]] = None,
    filename: Optional[str] = None,
    block_size: Optional[int] = None,
) -> Path:
    """
    Unified helper for results outputs under data/results.
    kind: "figures" or "dashboard".
    block_size: optional block size to target <kind>_L{block_size}.
    """
    base = Path("data") / "results"
    folder_map = {
        "figures": base / "figures",
        "dashboard": base / "dashboard",
    }
    if kind not in folder_map:
        raise ValueError(f"kind must be one of {list(folder_map.keys())}")
    if block_size is not None:
        if not isinstance(block_size, int):
            raise ValueError("block_size must be an int")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        path = base / f"{kind}_L{block_size}"
    else:
        path = folder_map[kind]
    if subfolder:
        path = path / subfolder
    category_parts = _category_parts(category)
    if category_parts:
        path = path.joinpath(*category_parts)
    if filename:
        path = path / filename
    return path


def results_root_path(filename: Optional[str] = None) -> Path:
    """Return the root `data/results` path, optionally with a filename appended."""
    path = Path("data") / "results"
    if filename:
        path = path / filename
    return path


def block_results_path(
    kind: str,
    block_size: int,
    *,
    template: Optional[str] = None,
    subfolder: Optional[str] = None,
    filename: Optional[str] = None,
) -> Path:
    """Return a block-specific results path while preserving any custom template override."""
    default_template = f"{kind}_L{{block_size}}"
    if template and template != default_template:
        path = results_root_path(template.format(block_size=block_size))
        if subfolder:
            path = path / subfolder
        if filename:
            path = path / filename
        return path
    return results_path(kind, subfolder=subfolder, filename=filename, block_size=block_size)


def window_metrics_filename(domain: str) -> str:
    """Return the standardized filename for a window-metrics JSON file."""
    domain_name = str(domain).strip()
    if not domain_name:
        raise ValueError("domain must be a non-empty string")
    return f"window_metrics.{domain_name}.json"


def window_metrics_path(
    *,
    domain: str,
    text_dir: Optional[Path] = None,
    genre: Optional[str] = None,
    author: Optional[str] = None,
    text_name: Optional[str] = None,
) -> Path:
    """Return the standardized window-metrics path for a text."""
    if text_dir is not None:
        return Path(text_dir) / window_metrics_filename(domain)
    if not all((genre, author, text_name)):
        raise ValueError("Provide `text_dir` or all of `genre`, `author`, and `text_name`.")
    return analytics_path("window", [genre, author, text_name], window_metrics_filename(domain))


def dashboard_report_candidates(base_dir: Path, text_name: str, suffix: str) -> List[Path]:
    """Return the standardized dashboard report path for a text."""
    resolved_base_dir = Path(base_dir)
    return [resolved_base_dir / f"text{suffix}"]


def dashboard_report_path(base_dir: Path, text_name: str, suffix: str) -> Path:
    """Return the standardized dashboard report path for a text."""
    return dashboard_report_candidates(base_dir, text_name, suffix)[0]


def find_topic_file(window_metrics_path: Path) -> Optional[Path]:
    """Return the topic JSON for a window metrics file, preferring clustered topics."""
    text_dir = window_metrics_path.parent
    author_dir = window_metrics_path.parent.parent
    genre_dir = window_metrics_path.parent.parent.parent
    topic_root = analytics_path("topic") / genre_dir.name / author_dir.name / text_dir.name
    candidates = [
        topic_root / f"{text_dir.name}_clustered_topics.json",
        topic_root / f"{text_dir.name}_topics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_window_metrics_files() -> List[Path]:
    """Return sorted syntax window metrics JSON paths under data/analytics/window_metrics."""
    root = analytics_path("window")
    if not root.exists():
        return []
    standard_name = window_metrics_filename("syntax")
    return sorted(root.glob(f"*/*/*/{standard_name}"))


def iter_dirs(
    root: Path,
    *,
    genres: Optional[Sequence[str]] = None,
    authors: Optional[Sequence[str]] = None,
    depth: Optional[int] = None,
) -> Iterable[Tuple[str, Path]]:
    """
    Yield (category_key, dir_path) pairs for category directories.
    depth=1 yields <root>/<category>, depth=2 yields <root>/<genre>/<author>.
    depth=None auto-detects leaf categories by file presence.
    Optional genres/authors filter first/second level names.
    """
    if not root.exists():
        return
    if depth is not None and depth not in (1, 2):
        raise ValueError("depth must be 1, 2, or None")

    if depth is None:
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            if genres and entry.name not in genres:
                continue
            if any(child.is_file() for child in entry.iterdir()):
                yield entry.name, entry
                continue
            for child in entry.iterdir():
                if not child.is_dir():
                    continue
                if authors and child.name not in authors:
                    continue
                if any(grandchild.is_file() for grandchild in child.iterdir()):
                    category_key = f"{entry.name}/{child.name}"
                    yield category_key, child
        return

    if depth == 1:
        entries = (
            [root / genre for genre in genres]
            if genres
            else [path for path in root.iterdir() if path.is_dir()]
        )
        for entry in entries:
            if entry.is_dir():
                yield entry.name, entry
        return

    genre_dirs = (
        [root / genre for genre in genres]
        if genres
        else [path for path in root.iterdir() if path.is_dir()]
    )
    for genre_dir in genre_dirs:
        if not genre_dir.is_dir():
            continue
        if authors:
            author_dirs = [genre_dir / author for author in authors]
        else:
            author_dirs = [path for path in genre_dir.iterdir() if path.is_dir()]
        for author_dir in author_dirs:
            if author_dir.is_dir():
                category_key = f"{genre_dir.name}/{author_dir.name}"
                yield category_key, author_dir

def sliding_windows(seq, n, step: int = DEFAULT_METRIC_WINDOW_STRIDE):
    """
    Sliding windows of width `n` with stride `step` (default DEFAULT_METRIC_WINDOW_STRIDE).
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
