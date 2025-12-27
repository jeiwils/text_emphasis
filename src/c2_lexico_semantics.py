import re
import statistics

import numpy as np

from x_configs import DEFAULT_WINDOW_SIZE
from .z_utils import sliding_windows, aggregate_windows

"""
Lexical and semantic content metrics (computation only).

Typical outputs:
- Windowed MATTR over sentence windows:
  {"mattr_score": 0.68, "token_count": 180, "window_token_span": 50, "start_sentence": 0, "end_sentence": 2}
- Lexical density window:
  {"lexical_density": 0.62, "token_count": 90, "content_count": 56, "start_sentence": 0, "end_sentence": 2}
- Average word frequency window:
  {"avg_word_freq": 14.2, "normalized_freq": 0.88, "content_function_ratio": 0.63}
- Semantic structures window:
  {"total_clauses": 5, "total_agents": 3, "total_patients": 2, "sentences": ["S1 text", "S2 text", "S3 text"]}

Use with the d-layer orchestrator to persist results.
"""



def _tokenize_words(text: str, lowercase: bool = True):
    """Lightweight tokenizer for lexical diversity (MATTR)."""
    tokens = re.findall(r"[A-Za-z0-9']+", text)
    if lowercase:
        tokens = [t.lower() for t in tokens]
    return tokens


def _moving_average_type_token_ratio(tokens, window_size: int = 50) -> float:
    """Compute Moving Average Type-Token Ratio (MATTR) over a sliding window."""
    tokens = [t for t in tokens if t]
    total_tokens = len(tokens)

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if total_tokens == 0:
        return 0.0
    if total_tokens < window_size:
        return round(len(set(tokens)) / total_tokens, 3)

    ttr_values = []
    for i in range(total_tokens - window_size + 1):
        window = tokens[i : i + window_size]
        ttr_values.append(len(set(window)) / window_size)

    return round(statistics.mean(ttr_values), 3)


def compute_mattr_metrics(text: str, window_size: int = 50, lowercase: bool = True):
    """Compute MATTR over the whole text for inclusion in window metrics."""
    words = _tokenize_words(text, lowercase=lowercase)
    mattr = _moving_average_type_token_ratio(words, window_size=window_size)
    return {
        "mattr_score": mattr,
        "window_size": min(window_size, len(words)),
        "total_tokens": len(words),
    }


class LexicoSemanticsAnalyzer:
    def __init__(self, nlp, corpus_freqs=None):
        self.nlp = nlp
        self.corpus_freqs = corpus_freqs or {}

    # ---------------------
    # Lexical Density
    # ---------------------
    def analyze_lexical_density(self, doc, window_size=None):
        sent_metrics = []
        for sent in doc.sents:
            tokens = [t for t in sent if not t.is_punct]
            content_words = [t for t in tokens if t.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]]

            sent_metrics.append({
                "sentence_text": sent.text,
                "token_count": len(tokens),
                "content_count": len(content_words),
                "lexical_density": len(content_words) / len(tokens) if tokens else None,
            })

        if window_size and window_size > 1:
            return aggregate_windows(sent_metrics, window_size)

        return sent_metrics

    # ---------------------
    # Windowed MATTR over sentence windows
    # ---------------------
    def compute_windowed_mattr(
        self,
        doc,
        window_size: int = DEFAULT_WINDOW_SIZE,
        mattr_window_size: int = 50,
        lowercase: bool = True,
    ):
        """
        Compute MATTR within each sentence window (same windowing as other metrics).

        Returns:
            list of dicts like {"mattr_score": 0.68, "token_count": 180, "window_token_span": 50,
                               "start_sentence": 0, "end_sentence": 2}
        """
        sentences = list(doc.sents)
        if not sentences:
            return []
        metrics = []
        for i, window in enumerate(sliding_windows(sentences, window_size)):
            window_text = " ".join(sent.text for sent in window)
            tokens = _tokenize_words(window_text, lowercase=lowercase)
            mattr = _moving_average_type_token_ratio(tokens, window_size=mattr_window_size) if tokens else 0.0
            metrics.append({
                "mattr_score": mattr,
                "token_count": len(tokens),
                "window_token_span": min(mattr_window_size, len(tokens)),
                "start_sentence": i,
                "end_sentence": i + len(window) - 1,
            })
        return metrics


    # ---------------------
    # Information Content
    # ---------------------
    def analyze_information_content(self, doc, word_frequencies, window_size=None):
        sent_metrics = []
        total_count = sum(word_frequencies.values()) if word_frequencies else 0

        for sent in doc.sents:
            ics = []
            for token in sent:
                if token.is_alpha:
                    freq = word_frequencies.get(token.text.lower(), 0)
                    if freq and total_count > 0:
                        prob = freq / total_count
                        ics.append(-np.log(prob))

            sent_metrics.append({
                "sentence_text": sent.text,
                "information_content": float(np.mean(ics)) if ics else None,
                "ic_values": ics,
            })

        if window_size and window_size > 1:
            return aggregate_windows(sent_metrics, window_size)

        return sent_metrics



    # ---------------------
    # Semantic Roles / Arguments
    # ---------------------
    def analyze_semantic_roles(self, doc, window_size=None):
        sent_metrics = []

        for sent in doc.sents:
            roles = []
            for token in sent:
                if token.dep_ in ["nsubj", "dobj", "iobj", "pobj"]:
                    roles.append({
                        "role": token.dep_,
                        "text": token.text,
                        "head": token.head.text,
                    })

            sent_metrics.append({
                "sentence_text": sent.text,
                "semantic_roles": roles,
                "role_count": len(roles),
            })

        if window_size and window_size > 1:
            return aggregate_windows(sent_metrics, window_size)

        return sent_metrics

# ----------------------------
# Average word frequency per sentence + sliding window
# ----------------------------
    def compute_avg_word_frequency(self, doc, global_avg_freq=None, window_size=None):
        """
        Compute average word frequency and content/function ratio per sentence or window,
        normalized by global frequency statistics if provided.
        """
        sent_metrics = []

        for sent in doc.sents:
            words = [token.text.lower() for token in sent if token.is_alpha]
            total_tokens = len(words)
            if self.corpus_freqs and words:
                freqs = [self.corpus_freqs.get(w, 1) for w in words]
                avg_word_freq = statistics.mean(freqs)
            else:
                avg_word_freq = 0

            # Normalization relative to global mean
            if global_avg_freq and global_avg_freq > 0:
                norm_freq = round(avg_word_freq / global_avg_freq, 3)
            else:
                norm_freq = round(avg_word_freq, 3)

            content_words = [t for t in sent if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]
            content_function_ratio = round(len(content_words)/total_tokens, 3) if total_tokens else 0

            sent_metrics.append({
                "sentence_text": sent.text,
                "avg_word_freq": round(avg_word_freq, 3),
                "normalized_freq": norm_freq,
                "content_function_ratio": content_function_ratio
            })

        # Apply sliding window if requested
        if window_size and window_size > 1:
            windowed_metrics = []
            for window in sliding_windows(sent_metrics, window_size):
                avg_freq = statistics.mean(d["avg_word_freq"] for d in window)
                avg_norm = statistics.mean(d["normalized_freq"] for d in window)
                avg_cfr = statistics.mean(d["content_function_ratio"] for d in window)
                windowed_metrics.append({
                    "avg_word_freq": round(avg_freq, 3),
                    "normalized_freq": round(avg_norm, 3),
                    "content_function_ratio": round(avg_cfr, 3)
                })
            return windowed_metrics

        return sent_metrics



    # ----------------------------
    # Extract semantic structures per clause + sliding window aggregation
    # ----------------------------
    def extract_semantic_structures(self, doc, window_size=None):
        clause_metrics_per_sentence = []

        for sent in doc.sents:
            clauses = []
            for token in sent:
                if token.pos_ != "VERB":
                    continue

                # Determine clause type
                if token.dep_ == "ROOT":
                    clause_type = "main"
                elif "advcl" in token.dep_ or "ccomp" in token.dep_ or "xcomp" in token.dep_:
                    clause_type = "subordinate"
                elif token.dep_ == "conj":
                    clause_type = "coordinate"
                else:
                    continue  # skip verbs that are not part of a clause

                # Extract agent (subject) - full subtree
                subjects = [child for child in token.children if "subj" in child.dep_]
                agent_phrases = [" ".join([t.text for t in subj.subtree]) for subj in subjects]
                agent = "; ".join(agent_phrases) if agent_phrases else None

                # Extract patient (object) - full subtree
                objects = [child for child in token.children if "obj" in child.dep_]
                patient_phrases = [" ".join([t.text for t in obj.subtree]) for obj in objects]
                patient = "; ".join(patient_phrases) if patient_phrases else None

                clause_tokens = [t.text for t in token.subtree]

                clauses.append({
                    "clause_level": clause_type,
                    "predicate": token.lemma_,
                    "agent": agent,
                    "patient": patient,
                    "clause_tokens": clause_tokens
                })

            clause_metrics_per_sentence.append({
                "sentence": sent.text,
                "sentence_text": sent.text,
                "clauses": clauses,
                "num_clauses": len(clauses),
                "num_agents": sum(1 for c in clauses if c["agent"]),
                "num_patients": sum(1 for c in clauses if c["patient"])
            })

        # Sliding window aggregation
        if window_size and window_size > 1:
            windowed_metrics = []
            for window in sliding_windows(clause_metrics_per_sentence, window_size):
                total_clauses = sum(d["num_clauses"] for d in window)
                total_agents = sum(d["num_agents"] for d in window)
                total_patients = sum(d["num_patients"] for d in window)

                windowed_metrics.append({
                    "sentences": [d["sentence"] for d in window],
                    "total_clauses": total_clauses,
                    "total_agents": total_agents,
                    "total_patients": total_patients
                })
            return windowed_metrics

        return clause_metrics_per_sentence

