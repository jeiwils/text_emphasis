

from typing import Optional
from pathlib import Path



def sliding_windows(seq, n):
    """

    """
    seq = list(seq)
    for i in range(len(seq) - n + 1):
        yield seq[i:i+n]






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
    filename: Optional[str] = None, 
    base_dir: str = "data/texts"
) -> Path:
    """

    
    
    """
    folder_map = {
        "cleaned": "cleaned_texts",
        "corpus": "corpus_analytics",
        "window": "window_metrics",
        "concept": "concept_embeddings",
        "network": "network_analysis"
    }
    
    if folder_type not in folder_map:
        raise ValueError(f"folder_type must be one of {list(folder_map.keys())}")
    
    path = Path(base_dir) / folder_map[folder_type]
    
    if filename:
        path = path / filename
    
    return path