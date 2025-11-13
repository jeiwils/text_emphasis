

from typing import Optional
from pathlib import Path
import json


def sliding_windows(seq, n):
    """
    there may be issues with this - what happens with truncation? 
    """
    seq = list(seq)
    for i in range(len(seq) - n + 1):
        yield seq[i:i+n]




def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def raw_text_path(
    category: Optional[str] = None,
    base_dir: str = "data/raw_texts"
) -> Path:
    """

    """
    path = Path(base_dir)

    if category:
        path = path / category

    return path


def processed_text_path(
    folder_type: str,
    subfolder: Optional[str] = None,
    filename: Optional[str] = None,
) -> Path:
    """

    """
    base_dir = "data/texts"

    folder_map = {
        "cleaned": "cleaned_texts",
        "corpus": "corpus_analytics",
        "window": "window_metrics",
    }

    if folder_type not in folder_map:
        raise ValueError(f"folder_type must be one of {list(folder_map.keys())}")

    path = Path(base_dir) / folder_map[folder_type]
    if subfolder:
        path = path / subfolder
    if filename:
        path = path / filename
    return path



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