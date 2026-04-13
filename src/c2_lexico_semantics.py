"""
Lexical and semantic content metrics (computation only).

Input (LexicoSemanticsAnalyzer.analyze_document):
{
  "doc": "spaCy Doc with sentence boundaries",
  "window_size": 3,
  "mattr_window_size": 50,
  "global_avg_freq": 12.4
}

Output:
{
  "meta": {"window_size": 3, "num_sentences": 120},
  "sentences": [
    {
      "sentence_id": 0,
      "token_count": 12,
      "content_count": 8,
      "lexical_density": 0.62,
      "information_content": 1.2,
      "information_content_token_count": 8,
      "information_content_values": [1.1, 1.3, ...],
      "role_count": 2,
      "role_counts": {"nsubj": 1, "dobj": 1, "iobj": 0, "pobj": 0},
      "role_count_per_token": 0.166667,
      "role_counts_per_token": {"nsubj": 0.083333, "dobj": 0.083333, "iobj": 0.0, "pobj": 0.0},
      "avg_word_freq": 14.2,
      "normalized_freq": 0.8,
      "content_function_ratio": 0.63,
      "avg_word_freq_token_count": 10,
      "num_clauses": 2,
      "num_agents": 1,
      "num_patients": 1,
      "num_clauses_per_token": 0.166667,
      "num_agents_per_token": 0.083333,
      "num_patients_per_token": 0.083333,
      "semantic_structures": {
        "clause_level_counts": {"main": 1, "subordinate": 1, "coordinate": 0},
        "num_clauses": 2,
        "num_agents": 1,
        "num_patients": 1
      }
    }
  ],
  "windows": [
    {
      "start_sentence": 0,
      "end_sentence": 2,
      "token_count": 36,
      "content_count": 22,
      "lexical_density": 0.611111,
      "information_content": 1.1,
      "avg_word_freq": 12.0,
      "normalized_freq": 0.76,
      "content_function_ratio": 0.611111,
      "num_clauses": 5,
      "num_agents": 2,
      "num_patients": 1,
      "role_count": 6,
      "role_count_per_token": 0.166667,
      "num_clauses_per_token": 0.138889,
      "num_agents_per_token": 0.055556,
      "num_patients_per_token": 0.027778,
      "num_agents_per_clause": 0.4,
      "num_patients_per_clause": 0.2,
      "role_count_per_clause": 1.2,
      "role_counts_per_token": {"nsubj": 0.083333, "dobj": 0.083333, "iobj": 0.0, "pobj": 0.0},
      "lexical_density_window": {"lexical_density": 0.611111, "token_count": 36, "content_count": 22},
      "information_content_window": {"information_content": 1.1, "token_count": 18},
      "semantic_roles_window": {"role_count": 6, "role_count_per_token": 0.166667, "token_count": 36},
      "avg_word_freq_window": {"avg_word_freq": 12.0, "normalized_freq": 0.76, "token_count": 30},
      "semantic_structures_window": {
        "clause_level_counts": {"main": 3, "subordinate": 2, "coordinate": 0},
        "num_clauses": 5,
        "num_agents": 2,
        "num_patients": 1
      },
      "lexical_diversity_mattr": {"mattr_score": 0.68, "token_count": 180, "window_token_span": 50}
    }
  ]
}
"""

import math
import statistics
from collections import Counter

import numpy as np

from .x_configs import DEFAULT_MATTR_WINDOW_SIZE, DEFAULT_WINDOW_SIZE
from .z_utils import aggregate_windows, load_spacy_model, sliding_windows

def _tokenize_words_from_tokens(tokens, lowercase: bool = True):
    """Tokenize using spaCy tokens for lexical diversity (MATTR)."""
    words = []
    for token in tokens:
        if token.is_space or token.is_punct:
            continue
        text = token.text
        if lowercase:
            text = text.lower()
        if text:
            words.append(text)
    return words


def _tokenize_words(text: str, lowercase: bool = True, nlp=None):
    """Tokenize with spaCy so MATTR matches pipeline tokenization."""
    nlp = nlp or load_spacy_model()
    doc = nlp(text)
    return _tokenize_words_from_tokens(doc, lowercase=lowercase)


def _moving_average_type_token_ratio(
    tokens, window_size: int = DEFAULT_MATTR_WINDOW_SIZE
) -> float:
    """Compute Moving Average Type-Token Ratio (MATTR) over a sliding window."""
    tokens = [t for t in tokens if t]
    total_tokens = len(tokens)

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if total_tokens == 0:
        return 0.0
    if total_tokens < window_size:
        return round(len(set(tokens)) / total_tokens, 3)

    counts = Counter(tokens[:window_size])
    ttr_values = [len(counts) / window_size]
    for i in range(window_size, total_tokens):
        outgoing = tokens[i - window_size]
        incoming = tokens[i]
        counts[outgoing] -= 1
        if counts[outgoing] <= 0:
            del counts[outgoing]
        counts[incoming] += 1
        ttr_values.append(len(counts) / window_size)

    return round(statistics.mean(ttr_values), 3)


def _shannon_entropy_from_tokens(tokens) -> dict:
    """Compute Shannon lexical entropy over a token sequence."""
    cleaned = [token for token in tokens if token]
    token_count = len(cleaned)
    if token_count == 0:
        return {
            "lexical_entropy": 0.0,
            "normalized_lexical_entropy": 0.0,
            "unique_token_count": 0,
            "token_count": 0,
        }

    counts = Counter(cleaned)
    probabilities = [count / token_count for count in counts.values() if count > 0]
    entropy = -sum(prob * math.log2(prob) for prob in probabilities)
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 0.0
    normalized_entropy = (entropy / max_entropy) if max_entropy > 0 else 0.0
    return {
        "lexical_entropy": round(entropy, 6),
        "normalized_lexical_entropy": round(normalized_entropy, 6),
        "unique_token_count": len(counts),
        "token_count": token_count,
    }


def compute_mattr_metrics(
    text: str,
    window_size: int = DEFAULT_MATTR_WINDOW_SIZE,
    lowercase: bool = True,
    nlp=None,
):
    """Compute MATTR over the whole text for inclusion in window metrics."""
    words = _tokenize_words(text, lowercase=lowercase, nlp=nlp)
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
                "token_count": len(tokens),
                "content_count": len(content_words),
                "lexical_density": len(content_words) / len(tokens) if tokens else 0.0,
            })

        windowed = aggregate_windows(sent_metrics, window_size) if window_size and window_size > 1 else []
        return sent_metrics, windowed

    # ---------------------
    # Windowed MATTR over sentence windows
    # ---------------------
    def compute_windowed_mattr(
        self,
        doc,
        window_size: int = DEFAULT_WINDOW_SIZE,
        mattr_window_size: int = DEFAULT_MATTR_WINDOW_SIZE,
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
        sentence_tokens = [
            _tokenize_words_from_tokens(sent, lowercase=lowercase) for sent in sentences
        ]
        metrics = []
        for i, window in enumerate(sliding_windows(sentence_tokens, window_size)):
            tokens = [token for sent_tokens in window for token in sent_tokens]
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
    # Lexical entropy (Shannon)
    # ---------------------
    def compute_lexical_entropy(
        self,
        doc,
        window_size: int = DEFAULT_WINDOW_SIZE,
        lowercase: bool = True,
    ):
        """Compute Shannon lexical entropy over tokens for each sentence and sentence window."""
        sentences = list(doc.sents)
        sentence_tokens = [
            _tokenize_words_from_tokens(sent, lowercase=lowercase) for sent in sentences
        ]
        sent_metrics = [_shannon_entropy_from_tokens(tokens) for tokens in sentence_tokens]

        windowed_metrics = []
        for idx, window in enumerate(sliding_windows(sentence_tokens, window_size)):
            tokens = [token for sent_tokens in window for token in sent_tokens]
            payload = _shannon_entropy_from_tokens(tokens)
            payload["start_sentence"] = idx
            payload["end_sentence"] = idx + len(window) - 1
            windowed_metrics.append(payload)

        return sent_metrics, windowed_metrics

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
                "information_content": float(np.mean(ics)) if ics else 0.0,
                "ic_values": ics,
                "token_count": len(ics),

            })

        windowed = aggregate_windows(sent_metrics, window_size) if window_size and window_size > 1 else []
        return sent_metrics, windowed



    # ---------------------
    # Semantic Roles / Arguments
    # ---------------------
    def analyze_semantic_roles(self, doc, window_size=None):
        sent_metrics = []

        for sent in doc.sents:
            tokens = [t for t in sent if not t.is_punct]
            token_count = len(tokens)
            role_counts = {"nsubj": 0, "dobj": 0, "iobj": 0, "pobj": 0}
            for token in sent:
                if token.dep_ in ["nsubj", "dobj", "iobj", "pobj"]:
                    role_counts[token.dep_] += 1

            role_count = sum(role_counts.values())
            role_counts_per_token = {
                key: (count / token_count if token_count else 0.0)
                for key, count in role_counts.items()
            }
            role_count_per_token = role_count / token_count if token_count else 0.0

            sent_metrics.append({
                "role_counts": role_counts,
                "role_count": role_count,
                "role_count_per_token": round(role_count_per_token, 6),
                "role_counts_per_token": {k: round(v, 6) for k, v in role_counts_per_token.items()},
                "token_count": token_count,
            })

        windowed = aggregate_windows(sent_metrics, window_size) if window_size and window_size > 1 else []
        return sent_metrics, windowed

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
                "avg_word_freq": round(avg_word_freq, 3),
                "normalized_freq": norm_freq,
                "content_function_ratio": content_function_ratio,
                "token_count": total_tokens,
            })

        windowed_metrics = aggregate_windows(sent_metrics, window_size) if window_size and window_size > 1 else []
        return sent_metrics, windowed_metrics



    # ----------------------------
    # Extract semantic structures per clause + sliding window aggregation
    # ----------------------------
    def extract_semantic_structures(self, doc, window_size=None):
        clause_metrics_per_sentence = []

        for sent in doc.sents:
            clauses = []
            clause_level_counts = {"main": 0, "subordinate": 0, "coordinate": 0}
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
                clause_level_counts[clause_type] += 1

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
                "clause_level_counts": clause_level_counts,
                "num_clauses": len(clauses),
                "num_agents": sum(1 for c in clauses if c["agent"]),
                "num_patients": sum(1 for c in clauses if c["patient"])
            })

        windowed_metrics = []
        if window_size and window_size > 1:
            for window in sliding_windows(clause_metrics_per_sentence, window_size):
                total_clauses = sum(d["num_clauses"] for d in window)
                total_agents = sum(d["num_agents"] for d in window)
                total_patients = sum(d["num_patients"] for d in window)
                clause_counts_window = {"main": 0, "subordinate": 0, "coordinate": 0}
                for d in window:
                    for level, count in d.get("clause_level_counts", {}).items():
                        clause_counts_window[level] = clause_counts_window.get(level, 0) + count

                windowed_metrics.append({
                    "clause_level_counts": clause_counts_window,
                    "num_clauses": total_clauses,
                    "num_agents": total_agents,
                    "num_patients": total_patients
                })

        return clause_metrics_per_sentence, windowed_metrics

    def analyze_document(
        self,
        doc,
        window_size=DEFAULT_WINDOW_SIZE,
        mattr_window_size=DEFAULT_MATTR_WINDOW_SIZE,
        lowercase=True,
        global_avg_freq=None,
    ):
        """
        Build aligned per-sentence and windowed lexico-semantic metrics.
        Window rows use aggregate_windows to average numeric fields over contiguous spans of size
        `window_size`, tagging each window with inclusive start/end indices; no raw text is emitted.
        """
        sentences = list(doc.sents)
        lexical_density_sent, lexical_density_win = self.analyze_lexical_density(doc, window_size=window_size)
        info_content_sent, info_content_win = self.analyze_information_content(
            doc, word_frequencies=self.corpus_freqs, window_size=window_size
        )
        semantic_roles_sent, semantic_roles_win = self.analyze_semantic_roles(doc, window_size=window_size)
        avg_word_freq_sent, avg_word_freq_win = self.compute_avg_word_frequency(
            doc, global_avg_freq=global_avg_freq, window_size=window_size
        )
        semantic_structures_sent, semantic_structures_win = self.extract_semantic_structures(
            doc, window_size=window_size
        )
        lexical_entropy_sent, lexical_entropy_win = self.compute_lexical_entropy(
            doc,
            window_size=window_size,
            lowercase=lowercase,
        )
        mattr_windows = self.compute_windowed_mattr(
            doc,
            window_size=window_size,
            mattr_window_size=mattr_window_size,
            lowercase=lowercase,
        )

        combined_sentences = []
        for idx, sent in enumerate(sentences):
            combined_sentences.append(
                {
                    "sentence_id": idx,
                    "lexical_density": lexical_density_sent[idx].get("lexical_density") if idx < len(lexical_density_sent) else None,
                    "token_count": lexical_density_sent[idx].get("token_count") if idx < len(lexical_density_sent) else 0,
                    "content_count": lexical_density_sent[idx].get("content_count") if idx < len(lexical_density_sent) else 0,
                    "information_content": info_content_sent[idx].get("information_content") if idx < len(info_content_sent) else None,
                    "information_content_token_count": info_content_sent[idx].get("token_count") if idx < len(info_content_sent) else 0,
                    "lexical_entropy": lexical_entropy_sent[idx].get("lexical_entropy") if idx < len(lexical_entropy_sent) else 0.0,
                    "normalized_lexical_entropy": lexical_entropy_sent[idx].get("normalized_lexical_entropy") if idx < len(lexical_entropy_sent) else 0.0,
                    "unique_token_count": lexical_entropy_sent[idx].get("unique_token_count") if idx < len(lexical_entropy_sent) else 0,
                    "role_count": semantic_roles_sent[idx].get("role_count") if idx < len(semantic_roles_sent) else 0,
                    "role_counts": semantic_roles_sent[idx].get("role_counts") if idx < len(semantic_roles_sent) else {},
                    "role_count_per_token": 0.0,
                    "role_counts_per_token": {},
                    "avg_word_freq": avg_word_freq_sent[idx].get("avg_word_freq") if idx < len(avg_word_freq_sent) else 0,
                    "normalized_freq": avg_word_freq_sent[idx].get("normalized_freq") if idx < len(avg_word_freq_sent) else 0,
                    "content_function_ratio": avg_word_freq_sent[idx].get("content_function_ratio") if idx < len(avg_word_freq_sent) else 0,
                    "avg_word_freq_token_count": avg_word_freq_sent[idx].get("token_count") if idx < len(avg_word_freq_sent) else 0,
                    "num_clauses": semantic_structures_sent[idx].get("num_clauses") if idx < len(semantic_structures_sent) else 0,
                    "num_agents": semantic_structures_sent[idx].get("num_agents") if idx < len(semantic_structures_sent) else 0,
                    "num_patients": semantic_structures_sent[idx].get("num_patients") if idx < len(semantic_structures_sent) else 0,
                    "num_clauses_per_token": 0.0,
                    "num_agents_per_token": 0.0,
                    "num_patients_per_token": 0.0,
                    "information_content_values": info_content_sent[idx].get("ic_values") if idx < len(info_content_sent) else [],
                    "semantic_structures": semantic_structures_sent[idx] if idx < len(semantic_structures_sent) else {},
                }
            )
            token_count = combined_sentences[-1]["token_count"]
            role_count = combined_sentences[-1]["role_count"]
            role_counts = combined_sentences[-1]["role_counts"]
            num_clauses = combined_sentences[-1]["num_clauses"]
            num_agents = combined_sentences[-1]["num_agents"]
            num_patients = combined_sentences[-1]["num_patients"]
            if token_count:
                combined_sentences[-1]["role_count_per_token"] = round(role_count / token_count, 6)
                combined_sentences[-1]["role_counts_per_token"] = {
                    k: round(v / token_count, 6) for k, v in role_counts.items()
                }
                combined_sentences[-1]["num_clauses_per_token"] = round(num_clauses / token_count, 6)
                combined_sentences[-1]["num_agents_per_token"] = round(num_agents / token_count, 6)
                combined_sentences[-1]["num_patients_per_token"] = round(num_patients / token_count, 6)

        # Build window inputs with numeric fields only (avoid nested lists/dicts during aggregation)
        window_inputs = [
            {
                "lexical_density": sent.get("lexical_density"),
                "lexical_entropy": sent.get("lexical_entropy"),
                "normalized_lexical_entropy": sent.get("normalized_lexical_entropy"),
                "information_content": sent.get("information_content"),
                "avg_word_freq": sent.get("avg_word_freq"),
                "normalized_freq": sent.get("normalized_freq"),
                "content_function_ratio": sent.get("content_function_ratio"),
                "num_clauses": sent.get("num_clauses"),
                "num_agents": sent.get("num_agents"),
                "num_patients": sent.get("num_patients"),
                "role_count": sent.get("role_count"),
            }
            for sent in combined_sentences
        ]

        windows = aggregate_windows(window_inputs, window_size) if window_inputs else []

        if windows:
            def _prefix(values):
                acc = [0.0]
                running = 0.0
                for value in values:
                    running += float(value)
                    acc.append(running)
                return acc

            def _range_sum(prefix, start, end):
                if start < 0 or end < start:
                    return 0.0
                return prefix[end + 1] - prefix[start]

            token_counts = [sent.get("token_count", 0) for sent in combined_sentences]
            content_counts = [sent.get("content_count", 0) for sent in combined_sentences]
            info_token_counts = [
                sent.get("information_content_token_count", 0) for sent in combined_sentences
            ]
            avg_word_token_counts = [
                sent.get("avg_word_freq_token_count", 0) for sent in combined_sentences
            ]
            info_contents = [
                (sent.get("information_content") or 0.0) for sent in combined_sentences
            ]
            avg_word_freqs = [
                sent.get("avg_word_freq", 0.0) for sent in combined_sentences
            ]
            normalized_freqs = [
                sent.get("normalized_freq", 0.0) for sent in combined_sentences
            ]
            role_counts = [sent.get("role_count", 0) for sent in combined_sentences]
            num_clauses = [sent.get("num_clauses", 0) for sent in combined_sentences]
            num_agents = [sent.get("num_agents", 0) for sent in combined_sentences]
            num_patients = [sent.get("num_patients", 0) for sent in combined_sentences]
            role_keys = set()
            for sent in combined_sentences:
                role_keys.update((sent.get("role_counts") or {}).keys())
            role_keys = sorted(role_keys)
            role_counts_by_key = {
                key: [
                    (sent.get("role_counts") or {}).get(key, 0) for sent in combined_sentences
                ]
                for key in role_keys
            }

            token_prefix = _prefix(token_counts)
            content_prefix = _prefix(content_counts)
            info_token_prefix = _prefix(info_token_counts)
            avg_word_token_prefix = _prefix(avg_word_token_counts)
            info_weighted_prefix = _prefix(
                [
                    info_contents[idx] * info_token_counts[idx]
                    for idx in range(len(info_contents))
                ]
            )
            avg_word_weighted_prefix = _prefix(
                [
                    avg_word_freqs[idx] * avg_word_token_counts[idx]
                    for idx in range(len(avg_word_freqs))
                ]
            )
            normalized_weighted_prefix = _prefix(
                [
                    normalized_freqs[idx] * avg_word_token_counts[idx]
                    for idx in range(len(normalized_freqs))
                ]
            )
            role_count_prefix = _prefix(role_counts)
            clause_prefix = _prefix(num_clauses)
            agent_prefix = _prefix(num_agents)
            patient_prefix = _prefix(num_patients)
            role_prefixes = {key: _prefix(values) for key, values in role_counts_by_key.items()}

            for window_idx, window in enumerate(windows):
                start = int(window.get("start_sentence", 0))
                end = int(window.get("end_sentence", start))
                total_tokens = _range_sum(token_prefix, start, end)
                total_content = _range_sum(content_prefix, start, end)
                info_tokens = _range_sum(info_token_prefix, start, end)
                avg_word_tokens = _range_sum(avg_word_token_prefix, start, end)

                if total_tokens > 0:
                    token_weighted_lexical_density = total_content / total_tokens
                else:
                    token_weighted_lexical_density = 0.0

                if info_tokens > 0:
                    info_weighted_sum = _range_sum(info_weighted_prefix, start, end)
                    token_weighted_information_content = info_weighted_sum / info_tokens
                else:
                    token_weighted_information_content = 0.0

                if avg_word_tokens > 0:
                    avg_word_weighted_sum = _range_sum(avg_word_weighted_prefix, start, end)
                    normalized_weighted_sum = _range_sum(normalized_weighted_prefix, start, end)
                    token_weighted_avg_word_freq = avg_word_weighted_sum / avg_word_tokens
                    token_weighted_normalized_freq = normalized_weighted_sum / avg_word_tokens
                else:
                    token_weighted_avg_word_freq = 0.0
                    token_weighted_normalized_freq = 0.0

                total_role_count = _range_sum(role_count_prefix, start, end)
                total_num_clauses = _range_sum(clause_prefix, start, end)
                total_num_agents = _range_sum(agent_prefix, start, end)
                total_num_patients = _range_sum(patient_prefix, start, end)
                role_counts_total = {
                    key: _range_sum(prefix, start, end)
                    for key, prefix in role_prefixes.items()
                }

                if total_tokens > 0:
                    role_count_per_token = total_role_count / total_tokens
                    num_clauses_per_token = total_num_clauses / total_tokens
                    num_agents_per_token = total_num_agents / total_tokens
                    num_patients_per_token = total_num_patients / total_tokens
                    role_counts_per_token = {
                        k: v / total_tokens for k, v in role_counts_total.items()
                    }
                else:
                    role_count_per_token = 0.0
                    num_clauses_per_token = 0.0
                    num_agents_per_token = 0.0
                    num_patients_per_token = 0.0
                    role_counts_per_token = {}

                windows[window_idx]["token_count"] = int(total_tokens)
                windows[window_idx]["content_count"] = int(total_content)
                windows[window_idx]["lexical_density"] = round(
                    token_weighted_lexical_density, 6
                )
                windows[window_idx]["lexical_density_per_token"] = windows[window_idx]["lexical_density"]
                windows[window_idx]["information_content_token_count"] = int(info_tokens)
                windows[window_idx]["avg_word_freq_token_count"] = int(avg_word_tokens)
                windows[window_idx]["information_content"] = round(
                    token_weighted_information_content, 6
                )
                windows[window_idx]["avg_word_freq"] = round(
                    token_weighted_avg_word_freq, 6
                )
                windows[window_idx]["normalized_freq"] = round(
                    token_weighted_normalized_freq, 6
                )
                windows[window_idx]["content_function_ratio"] = round(
                    total_content / total_tokens, 6
                ) if total_tokens else 0.0
                windows[window_idx]["role_count"] = int(total_role_count)
                windows[window_idx]["num_clauses"] = int(total_num_clauses)
                windows[window_idx]["num_agents"] = int(total_num_agents)
                windows[window_idx]["num_patients"] = int(total_num_patients)
                windows[window_idx]["role_count_per_token"] = round(role_count_per_token, 6)
                windows[window_idx]["num_clauses_per_token"] = round(num_clauses_per_token, 6)
                windows[window_idx]["num_agents_per_token"] = round(num_agents_per_token, 6)
                windows[window_idx]["num_patients_per_token"] = round(num_patients_per_token, 6)
                windows[window_idx]["role_counts_per_token"] = {
                    k: round(v, 6) for k, v in role_counts_per_token.items()
                }
                if total_num_clauses > 0:
                    windows[window_idx]["num_agents_per_clause"] = round(
                        total_num_agents / total_num_clauses, 6
                    )
                    windows[window_idx]["num_patients_per_clause"] = round(
                        total_num_patients / total_num_clauses, 6
                    )
                    windows[window_idx]["role_count_per_clause"] = round(
                        total_role_count / total_num_clauses, 6
                    )
                else:
                    windows[window_idx]["num_agents_per_clause"] = 0.0
                    windows[window_idx]["num_patients_per_clause"] = 0.0
                    windows[window_idx]["role_count_per_clause"] = 0.0
        # Attach window-level metrics from individual analyzers for richer payloads
        for idx, win in enumerate(windows):
            if idx < len(lexical_density_win):
                lex_win = lexical_density_win[idx]
                if isinstance(lex_win, dict):
                    lex_win["lexical_density"] = win.get("lexical_density")
                    lex_win["token_count"] = win.get("token_count", lex_win.get("token_count"))
                    lex_win["content_count"] = win.get("content_count", lex_win.get("content_count"))
                win["lexical_density_window"] = lex_win
            if idx < len(info_content_win):
                ic_win = info_content_win[idx]
                if isinstance(ic_win, dict):
                    ic_win["information_content"] = win.get("information_content")
                    ic_win["token_count"] = win.get(
                        "information_content_token_count", ic_win.get("token_count")
                    )
                win["information_content_window"] = ic_win
            if idx < len(semantic_roles_win):
                role_win = semantic_roles_win[idx]
                if isinstance(role_win, dict):
                    role_win["role_count"] = win.get("role_count", role_win.get("role_count"))
                    role_win["role_count_per_token"] = win.get(
                        "role_count_per_token", role_win.get("role_count_per_token")
                    )
                    role_win["role_counts_per_token"] = win.get(
                        "role_counts_per_token", role_win.get("role_counts_per_token")
                    )
                    role_win["token_count"] = win.get("token_count", role_win.get("token_count"))
                win["semantic_roles_window"] = role_win
            if idx < len(avg_word_freq_win):
                freq_win = avg_word_freq_win[idx]
                if isinstance(freq_win, dict):
                    freq_win["avg_word_freq"] = win.get("avg_word_freq")
                    freq_win["normalized_freq"] = win.get("normalized_freq")
                    freq_win["content_function_ratio"] = win.get("content_function_ratio")
                    freq_win["token_count"] = win.get(
                        "avg_word_freq_token_count", freq_win.get("token_count")
                    )
                win["avg_word_freq_window"] = freq_win
            if idx < len(semantic_structures_win):
                win["semantic_structures_window"] = semantic_structures_win[idx]
            if idx < len(lexical_entropy_win):
                entropy_win = lexical_entropy_win[idx]
                win["lexical_entropy"] = entropy_win.get("lexical_entropy", win.get("lexical_entropy", 0.0))
                win["normalized_lexical_entropy"] = entropy_win.get(
                    "normalized_lexical_entropy",
                    win.get("normalized_lexical_entropy", 0.0),
                )
                win["unique_token_count"] = entropy_win.get("unique_token_count", win.get("unique_token_count", 0))
                win["lexical_entropy_window"] = entropy_win
            if idx < len(mattr_windows):
                win["lexical_diversity_mattr"] = mattr_windows[idx]

        return {
            "meta": {
                "window_size": window_size,
                "num_sentences": len(sentences),
            },
            "sentences": combined_sentences,
            "windows": windows,
        }
