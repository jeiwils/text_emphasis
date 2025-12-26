from functools import lru_cache
from typing import Optional, Sequence

import spacy

model = 'PUT MODEL NAME HERE'

# Default spaCy pipeline configuration
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_SPACY_DISABLE: Sequence[str] = ()


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
