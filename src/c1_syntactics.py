"""
Sentence-level grammar metrics (clauses, depth, dependency complexity).

Input (SyntaxAnalyzer.analyze_document):
{
  "doc": "spaCy Doc with sentence boundaries",
  "window_size": 3
}

Output:
{
  "meta": {"window_size": 3, "num_sentences": 120},
  "sentences": [
    {
      "sentence_id": 0,
      "clause_counts": {"main": 1, "subordinate": 0, "coordinate": 0},
      "clause_counts_per_token": {"main": 0.083333, "subordinate": 0.0, "coordinate": 0.0},
      "clause_ratios": {"subordination_ratio": 0.0, "coordination_ratio": 0.0},
      "max_depth": 4,
      "mean_depth": 2.1,
      "median_depth": 2.0,
      "depth_skew": 0.1,
      "avg_dependents_per_head": {
        "main_clause": 2.1,
        "subordinate_clause": 0.0,
        "coordinate_clause": 0.0
      },
      "avg_max_dependents_per_head": 5,
      "avg_mean_dependency_distance": 1.4,
      "token_count": 12
    }
  ],
  "windows": [
    {
      "start_sentence": 0,
      "end_sentence": 2,
      "token_count": 36,
      "avg_tokens_per_sentence": 12.0,
      "clause_counts_per_token": {"main": 0.083333, "subordinate": 0.0, "coordinate": 0.0},
      "clause_ratios": {"subordination_ratio": 0.0, "coordination_ratio": 0.0},
      "avg_counts_per_token": {"main": 0.083333, "subordinate": 0.0, "coordinate": 0.0},
      "avg_ratios": {"subordination_ratio": 0.0, "coordination_ratio": 0.0},
      "max_depth": 3.3,
      "mean_depth": 1.9,
      "median_depth": 1.8,
      "depth_skew": 0.2,
      "avg_dependents_per_head": {
        "main_clause": 2.2,
        "subordinate_clause": 0.0,
        "coordinate_clause": 0.0
      },
      "avg_max_dependents_per_head": 4.3,
      "avg_mean_dependency_distance": 1.5
    }
  ]
}
"""

import math
import statistics
from collections import Counter

from .x_configs import DEFAULT_WINDOW_SIZE
from .z_utils import aggregate_windows, sliding_windows


class SyntaxAnalyzer:
    HYPOTAXIS_DEPS = {"advcl", "ccomp", "xcomp", "acl", "relcl"}
    PARATAXIS_DEPS = {"conj", "parataxis"}

    def __init__(self, nlp):
        self.nlp = nlp

    @staticmethod
    def _non_punct_tokens(sent):
        return [token for token in sent if not token.is_space and not token.is_punct]

    @classmethod
    def _classify_clause_role(cls, token):
        if token.dep_ == "ROOT":
            return "main"
        dep = token.dep_.lower()
        if dep in cls.HYPOTAXIS_DEPS:
            return "subordinate"
        if dep in cls.PARATAXIS_DEPS:
            return "coordinate"
        return None

    @staticmethod
    def _count_breath_units(sent):
        counts = {"comma": 0, "semicolon": 0, "dash": 0}
        for token in sent:
            token_text = token.text.strip()
            if token_text == ",":
                counts["comma"] += 1
            elif token_text == ";":
                counts["semicolon"] += 1
            elif token_text in {"—", "–"} or token_text.startswith("--"):
                counts["dash"] += 1
        return counts

    @staticmethod
    def _entropy_from_counts(counts):
        total = sum(counts.values())
        if total <= 0:
            return 0.0, 0.0
        probabilities = [count / total for count in counts.values() if count > 0]
        entropy = -sum(prob * math.log2(prob) for prob in probabilities)
        max_entropy = math.log2(len(probabilities)) if len(probabilities) > 1 else 0.0
        normalized_entropy = (entropy / max_entropy) if max_entropy > 0 else 0.0
        return round(entropy, 6), round(normalized_entropy, 6)

    @staticmethod
    def _approximate_entropy(values, m=2, r=None):
        cleaned = [float(value) for value in values if isinstance(value, (int, float))]
        n = len(cleaned)
        if n <= m + 1:
            return 0.0

        std_dev = statistics.pstdev(cleaned) if n > 1 else 0.0
        tolerance = float(r) if r is not None else (0.2 * std_dev)
        if tolerance <= 0:
            return 0.0

        def _phi(order):
            patterns = [cleaned[idx : idx + order] for idx in range(n - order + 1)]
            if not patterns:
                return 0.0
            match_rates = []
            for pattern in patterns:
                matches = 0
                for candidate in patterns:
                    if max(abs(a - b) for a, b in zip(pattern, candidate)) <= tolerance:
                        matches += 1
                match_rate = matches / len(patterns)
                if match_rate > 0:
                    match_rates.append(math.log(match_rate))
            return sum(match_rates) / len(patterns) if match_rates else 0.0

        return round(max(_phi(m) - _phi(m + 1), 0.0), 6)

    @classmethod
    def _build_parataxis_payload(cls, sent, token_count):
        hypotaxis_count = 0
        parataxis_count = 0
        for token in cls._non_punct_tokens(sent):
            dep = token.dep_.lower()
            if dep in cls.HYPOTAXIS_DEPS:
                hypotaxis_count += 1
            elif dep in cls.PARATAXIS_DEPS:
                parataxis_count += 1

        punctuation_linked_main_count = sum(1 for token in sent if token.text == ";")
        total_parataxis = parataxis_count + punctuation_linked_main_count
        return {
            "parataxis_count": int(total_parataxis),
            "hypotaxis_count": int(hypotaxis_count),
            "punctuation_linked_main_count": int(punctuation_linked_main_count),
            "parataxis_to_hypotaxis_ratio": round(total_parataxis / hypotaxis_count, 6)
            if hypotaxis_count
            else 0.0,
            "parataxis_per_token": round(total_parataxis / token_count, 6) if token_count else 0.0,
            "hypotaxis_per_token": round(hypotaxis_count / token_count, 6) if token_count else 0.0,
        }

    def compute_clause_metrics(self, doc, window_size=DEFAULT_WINDOW_SIZE):
        sentence_metrics = []

        for sent in doc.sents:
            tokens = self._non_punct_tokens(sent)
            token_count = len(tokens)
            main_counts = sub_counts = coord_counts = 0
            for token in tokens:
                clause_role = self._classify_clause_role(token)
                if clause_role == "main":
                    main_counts += 1
                elif clause_role == "subordinate":
                    sub_counts += 1
                elif clause_role == "coordinate":
                    coord_counts += 1

            parataxis_payload = self._build_parataxis_payload(sent, token_count)
            sub_to_main_ratio = sub_counts / main_counts if main_counts else 0.0
            coord_to_main_ratio = coord_counts / main_counts if main_counts else 0.0

            sentence_metrics.append(
                {
                    "avg_counts": {
                        "main": main_counts,
                        "subordinate": sub_counts,
                        "coordinate": coord_counts,
                    },
                    "avg_counts_per_token": {
                        "main": round(main_counts / token_count, 6) if token_count else 0.0,
                        "subordinate": round(sub_counts / token_count, 6) if token_count else 0.0,
                        "coordinate": round(coord_counts / token_count, 6) if token_count else 0.0,
                    },
                    "avg_ratios": {
                        "subordination_ratio": round(sub_to_main_ratio, 2),
                        "coordination_ratio": round(coord_to_main_ratio, 2),
                        "parataxis_to_hypotaxis_ratio": parataxis_payload[
                            "parataxis_to_hypotaxis_ratio"
                        ],
                    },
                    "parataxis_hypotaxis": parataxis_payload,
                    "token_count": token_count,
                }
            )

        windowed = aggregate_windows(sentence_metrics, window_size)
        return sentence_metrics, windowed

    def compute_clause_embedding_depth(self, doc, window_size=DEFAULT_WINDOW_SIZE):
        def token_depth(token):
            depth = 0
            while token.head != token:
                depth += 1
                token = token.head
            return depth

        sentence_depths = []
        for sent in doc.sents:
            sent_depths = [token_depth(token) for token in sent]
            if sent_depths:
                sentence_depths.append(
                    {
                        "max_depth": max(sent_depths),
                        "mean_depth": round(statistics.mean(sent_depths), 6),
                        "median_depth": round(statistics.median(sent_depths), 6),
                        "depth_skew": round(statistics.mean(sent_depths) - statistics.median(sent_depths), 6),
                    }
                )

        aggregated = aggregate_windows(sentence_depths, window_size)
        for window in aggregated:
            window["avg_max_depth"] = window.pop("max_depth", 0)
            window["avg_mean_depth"] = window.pop("mean_depth", 0)
            window["avg_median_depth"] = window.pop("median_depth", 0)
            window["avg_depth_skew"] = window.pop("depth_skew", 0)

        return sentence_depths, aggregated

    def compute_dependency_complexity(self, doc, window_size=DEFAULT_WINDOW_SIZE):
        sentence_metrics = []
        aux_metrics = []

        for sent in doc.sents:
            filtered_tokens = self._non_punct_tokens(sent)
            token_positions = {token.i: idx for idx, token in enumerate(filtered_tokens)}
            dependents_per_head = {"main_clause": [], "subordinate_clause": [], "coordinate_clause": []}
            dependency_distances = []

            for token in filtered_tokens:
                children = [child for child in token.children if not child.is_space and not child.is_punct]
                num_dependents = len(children)
                for child in children:
                    if token.i in token_positions and child.i in token_positions:
                        dependency_distances.append(
                            abs(token_positions[token.i] - token_positions[child.i])
                        )

                clause_role = self._classify_clause_role(token)
                if clause_role == "main":
                    dependents_per_head["main_clause"].append(num_dependents)
                elif clause_role == "subordinate":
                    dependents_per_head["subordinate_clause"].append(num_dependents)
                elif clause_role == "coordinate":
                    dependents_per_head["coordinate_clause"].append(num_dependents)

            all_dependents = (
                dependents_per_head["main_clause"]
                + dependents_per_head["subordinate_clause"]
                + dependents_per_head["coordinate_clause"]
            )
            dependents_sums = {
                "main_clause": sum(dependents_per_head["main_clause"]),
                "subordinate_clause": sum(dependents_per_head["subordinate_clause"]),
                "coordinate_clause": sum(dependents_per_head["coordinate_clause"]),
            }
            dependents_counts = {
                "main_clause": len(dependents_per_head["main_clause"]),
                "subordinate_clause": len(dependents_per_head["subordinate_clause"]),
                "coordinate_clause": len(dependents_per_head["coordinate_clause"]),
            }

            sentence_metrics.append(
                {
                    "avg_dependents_per_head": {
                        "main_clause": round(statistics.mean(dependents_per_head["main_clause"]), 6)
                        if dependents_per_head["main_clause"]
                        else 0,
                        "subordinate_clause": round(statistics.mean(dependents_per_head["subordinate_clause"]), 6)
                        if dependents_per_head["subordinate_clause"]
                        else 0,
                        "coordinate_clause": round(statistics.mean(dependents_per_head["coordinate_clause"]), 6)
                        if dependents_per_head["coordinate_clause"]
                        else 0,
                    },
                    "avg_max_dependents_per_head": max(all_dependents, default=0),
                    "avg_mean_dependency_distance": round(statistics.mean(dependency_distances), 6)
                    if dependency_distances
                    else 0,
                }
            )
            aux_metrics.append(
                {
                    "dependents_sums": dependents_sums,
                    "dependents_counts": dependents_counts,
                    "dependency_distance_sum": sum(dependency_distances),
                    "dependency_distance_count": len(dependency_distances),
                    "max_dependents": max(all_dependents, default=0),
                }
            )

        windowed = aggregate_windows(sentence_metrics, window_size)
        if windowed:
            for idx, window_aux in enumerate(sliding_windows(aux_metrics, window_size)):
                total_sums = {"main_clause": 0, "subordinate_clause": 0, "coordinate_clause": 0}
                total_counts = {"main_clause": 0, "subordinate_clause": 0, "coordinate_clause": 0}
                distance_sum = 0
                distance_count = 0
                max_dependents = 0
                for entry in window_aux:
                    for clause_key, value in entry["dependents_sums"].items():
                        total_sums[clause_key] += value
                    for clause_key, value in entry["dependents_counts"].items():
                        total_counts[clause_key] += value
                    distance_sum += entry["dependency_distance_sum"]
                    distance_count += entry["dependency_distance_count"]
                    max_dependents = max(max_dependents, entry["max_dependents"])

                windowed[idx]["avg_dependents_per_head"] = {
                    clause_key: round(total_sums[clause_key] / total_counts[clause_key], 6)
                    if total_counts[clause_key]
                    else 0
                    for clause_key in total_sums
                }
                windowed[idx]["avg_mean_dependency_distance"] = (
                    round(distance_sum / distance_count, 6) if distance_count else 0
                )
                windowed[idx]["avg_max_dependents_per_head"] = max_dependents

        return sentence_metrics, windowed

    def compute_structural_entropy(self, doc, window_size=DEFAULT_WINDOW_SIZE):
        sentence_metrics = []

        for sent in doc.sents:
            tokens = self._non_punct_tokens(sent)
            branching_factors = [
                sum(1 for child in token.children if not child.is_space and not child.is_punct)
                for token in tokens
            ]
            if not branching_factors:
                sentence_metrics.append(
                    {
                        "structural_entropy": 0.0,
                        "normalized_structural_entropy": 0.0,
                    }
                )
                continue

            distribution = {}
            for factor in branching_factors:
                distribution[factor] = distribution.get(factor, 0) + 1

            total = sum(distribution.values())
            probabilities = [count / total for count in distribution.values() if count > 0]
            entropy = -sum(prob * math.log(prob) for prob in probabilities)
            max_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
            normalized_entropy = (entropy / max_entropy) if max_entropy > 0 else 0.0

            sentence_metrics.append(
                {
                    "structural_entropy": round(entropy, 6),
                    "normalized_structural_entropy": round(normalized_entropy, 6),
                }
            )

        windowed = aggregate_windows(sentence_metrics, window_size)
        return sentence_metrics, windowed

    def compute_pos_ngram_entropy(self, doc, window_size=DEFAULT_WINDOW_SIZE, n=3):
        """Compute POS n-gram entropy to approximate syntactic unpredictability."""
        sentence_metrics = []
        sentence_tags = []

        def _payload(tags):
            cleaned_tags = [tag for tag in tags if tag]
            if not cleaned_tags:
                return {
                    "pos_ngram_entropy": 0.0,
                    "normalized_pos_ngram_entropy": 0.0,
                    "pos_ngram_count": 0,
                }
            effective_n = min(max(1, int(n)), len(cleaned_tags))
            ngrams = [
                tuple(cleaned_tags[idx : idx + effective_n])
                for idx in range(len(cleaned_tags) - effective_n + 1)
            ]
            if not ngrams:
                ngrams = [tuple(cleaned_tags)]
            counts = Counter(ngrams)
            entropy, normalized_entropy = self._entropy_from_counts(counts)
            return {
                "pos_ngram_entropy": entropy,
                "normalized_pos_ngram_entropy": normalized_entropy,
                "pos_ngram_count": len(ngrams),
            }

        for sent in doc.sents:
            tags = [token.pos_ or token.tag_ for token in self._non_punct_tokens(sent)]
            sentence_tags.append(tags)
            sentence_metrics.append(_payload(tags))

        windowed = []
        for idx, window in enumerate(sliding_windows(sentence_tags, window_size)):
            combined_tags = [tag for sent_tags in window for tag in sent_tags]
            payload = _payload(combined_tags)
            payload["start_sentence"] = idx
            payload["end_sentence"] = idx + len(window) - 1
            windowed.append(payload)

        return sentence_metrics, windowed

    def compute_structural_rhythm(self, doc, window_size=DEFAULT_WINDOW_SIZE, apen_window_size=None):
        """Compute approximate entropy over the sentence-length sequence as a rhythm signal."""
        sentence_lengths = [len(self._non_punct_tokens(sent)) for sent in doc.sents]
        if not sentence_lengths:
            return [], []

        local_window = max(5, int(apen_window_size or (window_size * 3)))
        radius = max(1, local_window // 2)

        sentence_metrics = []
        for idx, sent_length in enumerate(sentence_lengths):
            start = max(0, idx - radius)
            end = min(len(sentence_lengths), idx + radius + 1)
            local_sequence = sentence_lengths[start:end]
            sentence_metrics.append(
                {
                    "sentence_length": int(sent_length),
                    "sentence_length_approx_entropy": self._approximate_entropy(local_sequence),
                }
            )

        windowed = aggregate_windows(sentence_metrics, window_size)
        for idx, window_lengths in enumerate(sliding_windows(sentence_lengths, window_size)):
            if idx >= len(windowed):
                continue
            contextual_lengths = list(window_lengths)
            if len(contextual_lengths) <= 3:
                start = max(0, idx - radius)
                end = min(len(sentence_lengths), idx + window_size + radius)
                contextual_lengths = sentence_lengths[start:end]
            windowed[idx]["avg_sentence_length"] = round(statistics.mean(window_lengths), 6) if window_lengths else 0.0
            windowed[idx]["sentence_length_std"] = round(statistics.pstdev(window_lengths), 6) if len(window_lengths) > 1 else 0.0
            windowed[idx]["sentence_length_approx_entropy"] = self._approximate_entropy(contextual_lengths)

        return sentence_metrics, windowed

    def analyze_document(self, doc, window_size=DEFAULT_WINDOW_SIZE):
        """
        Convenience wrapper to compute all syntax metrics for a spaCy Doc.

        Returns:
            dict with meta, sentences (per-sentence combined syntax), windows (aggregated).
            Window rows are built with aggregate_windows: contiguous spans of size `window_size`
            are averaged across numeric fields and tagged with inclusive start/end sentence indices.

        Example:
            >>> from .z_utils import load_spacy_model
            >>> nlp = load_spacy_model()
            >>> doc = nlp("One. Two. Three.")
            >>> SyntaxAnalyzer(nlp).analyze_document(doc)["sentences"][0]["clause_counts"]["main"]
            1.0
        """
        sentences = list(doc.sents)
        punctuation_counts = [sum(1 for t in sent if t.is_punct) for sent in sentences]
        clause_sent, clause_windows = self.compute_clause_metrics(doc, window_size=window_size)
        depth_sent, depth_windows = self.compute_clause_embedding_depth(doc, window_size=window_size)
        dep_sent, dep_windows = self.compute_dependency_complexity(doc, window_size=window_size)
        entropy_sent, entropy_windows = self.compute_structural_entropy(doc, window_size=window_size)
        pos_entropy_sent, pos_entropy_windows = self.compute_pos_ngram_entropy(doc, window_size=window_size)
        rhythm_sent, rhythm_windows = self.compute_structural_rhythm(doc, window_size=window_size)

        combined_sentences = []
        for idx, sent in enumerate(sentences):
            clause_payload = clause_sent[idx] if idx < len(clause_sent) else {}
            depth_payload = depth_sent[idx] if idx < len(depth_sent) else {}
            dep_payload = dep_sent[idx] if idx < len(dep_sent) else {}
            entropy_payload = entropy_sent[idx] if idx < len(entropy_sent) else {}
            pos_entropy_payload = pos_entropy_sent[idx] if idx < len(pos_entropy_sent) else {}
            rhythm_payload = rhythm_sent[idx] if idx < len(rhythm_sent) else {}
            token_count = clause_payload.get("token_count", 0)
            punctuation_count = punctuation_counts[idx] if idx < len(punctuation_counts) else 0
            total_non_space = token_count + punctuation_count
            punctuation_per_token = round(punctuation_count / total_non_space, 6) if total_non_space else 0.0
            breath_unit_counts = self._count_breath_units(sent)
            breath_unit_per_1000_words = {
                key: round(value / token_count * 1000, 6) if token_count else 0.0
                for key, value in breath_unit_counts.items()
            }
            breath_unit_per_1000_words["total"] = round(
                sum(breath_unit_counts.values()) / token_count * 1000, 6
            ) if token_count else 0.0

            combined_sentences.append(
                {
                    "sentence_id": idx,
                    "clause_counts": clause_payload.get("avg_counts", {}),
                    "clause_counts_per_token": clause_payload.get("avg_counts_per_token", {}),
                    "clause_ratios": clause_payload.get("avg_ratios", {}),
                    "parataxis_hypotaxis": clause_payload.get("parataxis_hypotaxis", {}),
                    "max_depth": depth_payload.get("max_depth", 0),
                    "mean_depth": depth_payload.get("mean_depth", 0),
                    "median_depth": depth_payload.get("median_depth", 0),
                    "depth_skew": depth_payload.get("depth_skew", 0),
                    "structural_entropy": entropy_payload.get("structural_entropy", 0.0),
                    "normalized_structural_entropy": entropy_payload.get(
                        "normalized_structural_entropy", 0.0
                    ),
                    "pos_ngram_entropy": pos_entropy_payload.get("pos_ngram_entropy", 0.0),
                    "normalized_pos_ngram_entropy": pos_entropy_payload.get(
                        "normalized_pos_ngram_entropy", 0.0
                    ),
                    "sentence_length": rhythm_payload.get("sentence_length", token_count),
                    "sentence_length_approx_entropy": rhythm_payload.get(
                        "sentence_length_approx_entropy", 0.0
                    ),
                    "avg_dependents_per_head": dep_payload.get("avg_dependents_per_head", {}),
                    "avg_max_dependents_per_head": dep_payload.get("avg_max_dependents_per_head", 0),
                    "avg_mean_dependency_distance": dep_payload.get("avg_mean_dependency_distance", 0),
                    "token_count": clause_payload.get("token_count", 0),
                    "punctuation_count": punctuation_count,
                    "punctuation_per_token": punctuation_per_token,
                    "breath_unit_counts": breath_unit_counts,
                    "breath_unit_per_1000_words": breath_unit_per_1000_words,
                }
            )

        windows = aggregate_windows(combined_sentences, window_size) if combined_sentences else []
        # Merge per-metric windowed summaries so everything lives under `windows`.
        window_slices = list(sliding_windows(combined_sentences, window_size))
        for idx, window in enumerate(windows):
            if idx < len(clause_windows):
                window.update(clause_windows[idx])
            if idx < len(depth_windows):
                window.update(depth_windows[idx])
            if idx < len(dep_windows):
                window.update(dep_windows[idx])
            if idx < len(entropy_windows):
                window.update(entropy_windows[idx])
            if idx < len(pos_entropy_windows):
                window.update(pos_entropy_windows[idx])
            if idx < len(rhythm_windows):
                window.update(rhythm_windows[idx])
            if idx < len(window_slices):
                window_sents = window_slices[idx]
                window["max_depth"] = max(
                    (sent.get("max_depth", 0) for sent in window_sents),
                    default=0,
                )
                total_tokens = sum(sent.get("token_count", 0) for sent in window_sents)
                total_punctuation = sum(sent.get("punctuation_count", 0) for sent in window_sents)
                total_non_space = total_tokens + total_punctuation
                window["token_count"] = total_tokens
                window["avg_tokens_per_sentence"] = round(
                    total_tokens / len(window_sents), 6
                ) if window_sents else 0.0
                window["punctuation_count"] = total_punctuation
                window["punctuation_per_token"] = (
                    round(total_punctuation / total_non_space, 6) if total_non_space else 0.0
                )
                clause_counts_total = {"main": 0, "subordinate": 0, "coordinate": 0}
                breath_unit_totals = {"comma": 0, "semicolon": 0, "dash": 0}
                total_parataxis = 0
                total_hypotaxis = 0
                total_punctuation_linked_main = 0
                for sent in window_sents:
                    for key, value in sent.get("clause_counts", {}).items():
                        clause_counts_total[key] = clause_counts_total.get(key, 0) + value
                    for key, value in sent.get("breath_unit_counts", {}).items():
                        breath_unit_totals[key] = breath_unit_totals.get(key, 0) + value
                    parataxis_payload = sent.get("parataxis_hypotaxis", {})
                    if isinstance(parataxis_payload, dict):
                        total_parataxis += parataxis_payload.get("parataxis_count", 0)
                        total_hypotaxis += parataxis_payload.get("hypotaxis_count", 0)
                        total_punctuation_linked_main += parataxis_payload.get(
                            "punctuation_linked_main_count", 0
                        )
                if total_tokens > 0:
                    window["clause_counts_per_token"] = {
                        k: round(v / total_tokens, 6) for k, v in clause_counts_total.items()
                    }
                else:
                    window["clause_counts_per_token"] = {k: 0.0 for k in clause_counts_total}
                total_main = clause_counts_total.get("main", 0)
                total_sub = clause_counts_total.get("subordinate", 0)
                total_coord = clause_counts_total.get("coordinate", 0)
                sub_ratio = total_sub / total_main if total_main else 0.0
                coord_ratio = total_coord / total_main if total_main else 0.0
                parataxis_ratio = total_parataxis / total_hypotaxis if total_hypotaxis else 0.0
                window["clause_ratios"] = {
                    "subordination_ratio": round(sub_ratio, 2),
                    "coordination_ratio": round(coord_ratio, 2),
                    "parataxis_to_hypotaxis_ratio": round(parataxis_ratio, 6),
                }
                window["parataxis_hypotaxis"] = {
                    "parataxis_count": int(total_parataxis),
                    "hypotaxis_count": int(total_hypotaxis),
                    "punctuation_linked_main_count": int(total_punctuation_linked_main),
                    "parataxis_to_hypotaxis_ratio": round(parataxis_ratio, 6),
                    "parataxis_per_token": round(total_parataxis / total_tokens, 6) if total_tokens else 0.0,
                    "hypotaxis_per_token": round(total_hypotaxis / total_tokens, 6) if total_tokens else 0.0,
                }
                breath_unit_per_1000_words = {
                    key: round(value / total_tokens * 1000, 6) if total_tokens else 0.0
                    for key, value in breath_unit_totals.items()
                }
                breath_unit_per_1000_words["total"] = round(
                    sum(breath_unit_totals.values()) / total_tokens * 1000, 6
                ) if total_tokens else 0.0
                window["breath_unit_counts"] = breath_unit_totals
                window["breath_unit_per_1000_words"] = breath_unit_per_1000_words
                window["clause_ratios_per_main"] = window["clause_ratios"]
                window["clause_counts_count"] = clause_counts_total
                # Keep avg_* aliases aligned to token-weighted window metrics.
                window["avg_counts_per_token"] = window["clause_counts_per_token"]
                window["avg_ratios"] = window["clause_ratios"]

        return {
            "meta": {
                "window_size": window_size,
                "num_sentences": len(sentences),
            },
            "sentences": combined_sentences,
            "windows": windows,
        }
