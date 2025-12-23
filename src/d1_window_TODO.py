import re
import statistics
from typing import Iterable, List


def _tokenize_words(text: str, lowercase: bool = True) -> List[str]:
    """
    Simple word tokenizer for window-level metrics.

    Uses a basic regex to keep alphanumeric tokens (and apostrophes) so that
    punctuation does not inflate token counts. Optionally lowercases tokens to
    keep type counts consistent.
    """
    tokens = re.findall(r"[\\w']+", text)
    if lowercase:
        tokens = [t.lower() for t in tokens]
    return tokens


def moving_average_type_token_ratio(tokens: Iterable[str], window_size: int = 50) -> float:
    """
    Compute Moving Average Type-Token Ratio (MATTR) for a sequence of tokens.

    Args:
        tokens: Iterable of word tokens.
        window_size: Sliding window size for MATTR calculation.

    Returns:
        Rounded MATTR score (0.0–1.0). Empty input yields 0.0.
    """
    tokens = [t for t in tokens if t]  # remove empties/None
    total_tokens = len(tokens)

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    if total_tokens == 0:
        return 0.0

    # If fewer tokens than the window, fall back to classic TTR.
    if total_tokens < window_size:
        return round(len(set(tokens)) / total_tokens, 3)

    ttr_values = []
    for i in range(total_tokens - window_size + 1):
        window = tokens[i : i + window_size]
        ttr_values.append(len(set(window)) / window_size)

    return round(statistics.mean(ttr_values), 3)


def compute_window_metrics(text: str, window_size: int = 50, lowercase: bool = True) -> dict:
    """
    Convenience wrapper that tokenizes raw text then computes window metrics.

    Currently returns MATTR, but can be extended with additional window-level
    lexical diversity or dispersion scores.
    """
    words = _tokenize_words(text, lowercase=lowercase)
    mattr = moving_average_type_token_ratio(words, window_size=window_size)

    return {
        "mattr_score": mattr,
        "window_size": min(window_size, len(words)),
        "total_tokens": len(words),
    }