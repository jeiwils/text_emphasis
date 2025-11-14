

from typing import Optional
from pathlib import Path
import json
import numpy as np
from statistics import mean


def sliding_windows(seq, n):
    """
    there may be issues with this - what happens with truncation? 
    """
    seq = list(seq)
    for i in range(len(seq) - n + 1):
        yield seq[i:i+n]


def aggregate_windows(sent_metrics, window_size):
    """
    Aggregate sentence-level metrics over sliding windows.
    Returns a flat list of dicts with averaged numeric values per window.
    Each window includes 'start_sentence' and 'end_sentence'.
    """
    windows = []
    n = len(sent_metrics)
    if n == 0:
        return windows

    for i in range(0, n, window_size):
        window_sents = sent_metrics[i:i + window_size]  # handles last partial window
        agg = {}

        for key in window_sents[0]:
            if isinstance(window_sents[0][key], dict):
                # Average numeric values in nested dict
                agg[key] = {}
                all_inner_keys = set(k for d in window_sents for k in d[key].keys())
                for k in all_inner_keys:
                    nums = [d[key][k] for d in window_sents
                            if k in d[key] and isinstance(d[key][k], (int, float))]
                    agg[key][k] = round(mean(nums), 2) if nums else 0
            elif isinstance(window_sents[0][key], (int, float)):
                # Average numeric values
                nums = [d[key] for d in window_sents if isinstance(d[key], (int, float))]
                agg[key] = round(mean(nums), 2) if nums else 0
            else:
                # Keep non-numeric fields (e.g., strings)
                agg[key] = window_sents[0][key]

        # Add window metadata
        agg["start_sentence"] = i
        agg["end_sentence"] = i + len(window_sents) - 1  # correct end index for partial windows

        windows.append(agg)

    return windows



# def aggregate_windows(sent_metrics, window_size):
#     windows = []
#     for i in range(0, len(sent_metrics) - window_size + 1):
#         window_sents = sent_metrics[i:i + window_size]

#         agg = {}
#         for key in window_sents[0]:
#             if isinstance(window_sents[0][key], dict):
#                 # Collect all keys that appear in any dict in the window
#                 all_inner_keys = set(k for d in window_sents for k in d[key].keys())
#                 agg[key] = {}
#                 for k in all_inner_keys:
#                     nums = [d[key][k] for d in window_sents if k in d[key] and isinstance(d[key][k], (int, float))]
#                     agg[key][k] = round(mean(nums), 2) if nums else 0
#             elif isinstance(window_sents[0][key], (int, float)):
#                 nums = [d[key] for d in window_sents if isinstance(d[key], (int, float))]
#                 agg[key] = round(mean(nums), 2) if nums else 0
#             else:
#                 # Keep strings or other non-numeric fields as-is
#                 agg[key] = window_sents[0][key]

#         # Add metadata
#         agg["start_sentence"] = i
#         agg["end_sentence"] = i + window_size - 1
#         windows.append(agg)

#     return windows





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