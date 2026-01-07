from functools import lru_cache
from typing import Optional, Sequence

import spacy

USE_EXISTING = True

MODEL_CONFIGS = {
    "causal_lm": "gpt2",
    "sentence_embedding": "all-MiniLM-L6-v2",
    "asr": "openai/whisper-small",
}

# Default spaCy pipeline configuration
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_SPACY_DISABLE: Sequence[str] = ()
# Shared window size (in sentences) for sliding window metrics
DEFAULT_WINDOW_SIZE: int = 3
# Category-specific window multiples for topic modelling
TOPIC_WINDOW_MULTIPLES = {
    "short_stories": 3,
    "novellas": 3,
    "novels": 3,
}

# Category-specific clustering settings for topic modelling (HDBSCAN)
# Entries fall back to extract_topics defaults when unspecified.
TOPIC_CLUSTERING = {
    "short_stories": {"min_cluster_size": 3, "min_samples": 1},
    "novellas": {"min_cluster_size": 5, "min_samples": 1},
    "novels": {"min_cluster_size": 8, "min_samples": 3},
}

# Book-level overrides for topic modelling when defaults over/under-cluster.
# Keys: <category> -> <book_base_name> (stem without _normalised_segmented).
TOPIC_BOOK_OVERRIDES = {
    "short_stories": {
        # Very short texts; use smaller clusters and stride 1 to avoid losing detail.
        "the_telltale_heart": {
            "window_multiple": 1,
            "window_stride": 1,
            "min_cluster_size": 2,
            "min_samples": 1,
        },
        "the_black_cat": {
            "window_multiple": 1,
            "window_stride": 1,
            "min_cluster_size": 2,
            "min_samples": 1,
        },
    },
    "novellas": {
        # Slightly reduce cluster size for tighter narratives.
        "animal_farm": {
            "window_multiple": 3,
            "window_stride": 3,
            "min_cluster_size": 6,
            "min_samples": 2,
        },
        "a_clockwork_orange": {
            "window_multiple": 4,
            "window_stride": 3,
            "min_cluster_size": 5,
            "min_samples": 1,
        },
        "coraline": {
            "window_multiple": 2,
            "window_stride": 2,
            "min_cluster_size": 3,
            "min_samples": 1,
        },
        "the_case_of_charles_dexter_ward": {
            "window_multiple": 3,
            "window_stride": 3,
            "min_cluster_size": 6,
            "min_samples": 2,
        },
        "the_metamorphosis": {
            "window_multiple": 2,
            "window_stride": 2,
            "min_cluster_size": 4,
            "min_samples": 2,
        },
    },
}


@lru_cache(maxsize=None)
def load_spacy_model(
    model_name: str = DEFAULT_SPACY_MODEL,
    disable: Optional[Sequence[str]] = None,
):
    """
    Shared spaCy loader with simple caching driven by config defaults.
    Pass a different model_name/disable list to override per call.
    """
    disable_components = tuple(disable) if disable else DEFAULT_SPACY_DISABLE
    return spacy.load(model_name, disable=list(disable_components))
