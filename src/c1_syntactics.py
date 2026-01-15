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

import statistics

from .x_configs import DEFAULT_WINDOW_SIZE
from .z_utils import aggregate_windows, sliding_windows


class SyntaxAnalyzer:
    def __init__(self, nlp):
        self.nlp = nlp

    def compute_clause_metrics(self, doc, window_size=DEFAULT_WINDOW_SIZE):
        sentence_metrics = []

        for sent in doc.sents:
            tokens = [t for t in sent if not t.is_punct]
            token_count = len(tokens)
            main_counts = sub_counts = coord_counts = 0
            for token in tokens:
                if token.dep_ == "ROOT":
                    main_counts += 1
                elif token.dep_ in ("advcl", "ccomp", "xcomp"):
                    sub_counts += 1
                elif token.dep_ == "conj":
                    coord_counts += 1

            sub_to_main_ratio = sub_counts / main_counts if main_counts else 0
            coord_to_main_ratio = coord_counts / main_counts if main_counts else 0

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
                    },
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
            dependents_per_head = {"main_clause": [], "subordinate_clause": [], "coordinate_clause": []}
            dependency_distances = []

            for token in sent:
                num_dependents = len(list(token.children))
                dependency_distances.extend([abs(token.i - child.i) for child in token.children])

                if token.dep_ == "ROOT":
                    dependents_per_head["main_clause"].append(num_dependents)
                elif token.dep_ in ("advcl", "ccomp", "xcomp"):
                    dependents_per_head["subordinate_clause"].append(num_dependents)
                elif token.dep_ == "conj":
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

    def analyze_document(self, doc, window_size=DEFAULT_WINDOW_SIZE):
        """
        Convenience wrapper to compute all syntax metrics for a spaCy Doc.

        Returns:
            dict with meta, sentences (per-sentence combined syntax), windows (aggregated).
            Window rows are built with aggregate_windows: contiguous spans of size `window_size`
            are averaged across numeric fields and tagged with inclusive start/end sentence indices.

        Example:
            >>> from .x_configs import load_spacy_model
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

        combined_sentences = []
        for idx, sent in enumerate(sentences):
            clause_payload = clause_sent[idx] if idx < len(clause_sent) else {}
            depth_payload = depth_sent[idx] if idx < len(depth_sent) else {}
            dep_payload = dep_sent[idx] if idx < len(dep_sent) else {}
            token_count = clause_payload.get("token_count", 0)
            punctuation_count = punctuation_counts[idx] if idx < len(punctuation_counts) else 0
            total_non_space = token_count + punctuation_count
            punctuation_per_token = round(punctuation_count / total_non_space, 6) if total_non_space else 0.0

            combined_sentences.append(
                {
                    "sentence_id": idx,
                    "clause_counts": clause_payload.get("avg_counts", {}),
                    "clause_counts_per_token": clause_payload.get("avg_counts_per_token", {}),
                    "clause_ratios": clause_payload.get("avg_ratios", {}),
                    "max_depth": depth_payload.get("max_depth", 0),
                    "mean_depth": depth_payload.get("mean_depth", 0),
                    "median_depth": depth_payload.get("median_depth", 0),
                    "depth_skew": depth_payload.get("depth_skew", 0),
                    "avg_dependents_per_head": dep_payload.get("avg_dependents_per_head", {}),
                    "avg_max_dependents_per_head": dep_payload.get("avg_max_dependents_per_head", 0),
                    "avg_mean_dependency_distance": dep_payload.get("avg_mean_dependency_distance", 0),
                    "token_count": clause_payload.get("token_count", 0),
                    "punctuation_count": punctuation_count,
                    "punctuation_per_token": punctuation_per_token,
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
                for sent in window_sents:
                    for key, value in sent.get("clause_counts", {}).items():
                        clause_counts_total[key] = clause_counts_total.get(key, 0) + value
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
                window["clause_ratios"] = {
                    "subordination_ratio": round(sub_ratio, 2),
                    "coordination_ratio": round(coord_ratio, 2),
                }
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
