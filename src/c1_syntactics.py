import statistics
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from x_configs import DEFAULT_WINDOW_SIZE
from .z_utils import aggregate_windows, graph_path, load_json

"""
Sentence-level grammar metrics (clauses, syntactic depth, dependency complexity).
Only computation/visualisation helpers live here; the d-layer handles reading/writing.

Example output snippet from `analyze_document`:
{
  "window_size": 3,
  "num_sentences": 120,
  "clause_metrics": [
    {"avg_counts": {"main": 1, "subordinate": 0, "coordinate": 1},
     "avg_ratios": {"subordination_ratio": 0.0, "coordination_ratio": 1.0},
     "sentences": ["S1...", "S2...", "S3..."],
     "start_sentence": 0, "end_sentence": 2},
    ...
  ],
  "clause_embedding_metrics": [
    {"avg_max_depth": 4, "avg_mean_depth": 2.1, "avg_median_depth": 2.0, "avg_depth_skew": 0.1,
     "sentences": [...], "start_sentence": 0, "end_sentence": 2},
    ...
  ],
  "dependency_metrics": [
    {"avg_dependents_per_head": {"main_clause": 2.1, "subordinate_clause": 1.0, "coordinate_clause": 0.5},
     "avg_max_dependents_per_head": 5, "avg_mean_dependency_distance": 1.4,
     "sentences": [...], "start_sentence": 0, "end_sentence": 2},
    ...
  ]
}
"""


class SyntaxAnalyzer:
    def __init__(self, nlp):
        self.nlp = nlp

    def compute_clause_metrics(self, doc, window_size=DEFAULT_WINDOW_SIZE):
        sentence_metrics = []

        for sent in doc.sents:
            main_counts = sub_counts = coord_counts = 0
            for token in sent:
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
                    "sentence_text": sent.text,
                    "avg_counts": {
                        "main": main_counts,
                        "subordinate": sub_counts,
                        "coordinate": coord_counts,
                    },
                    "avg_ratios": {
                        "subordination_ratio": round(sub_to_main_ratio, 2),
                        "coordination_ratio": round(coord_to_main_ratio, 2),
                    },
                }
            )

        return aggregate_windows(sentence_metrics, window_size)

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
                        "sentence_text": sent.text,
                        "max_depth": max(sent_depths),
                        "mean_depth": round(statistics.mean(sent_depths), 2),
                        "median_depth": round(statistics.median(sent_depths), 2),
                        "depth_skew": round(statistics.mean(sent_depths) - statistics.median(sent_depths), 2),
                    }
                )

        aggregated = aggregate_windows(sentence_depths, window_size)
        for window in aggregated:
            window["avg_max_depth"] = window.pop("max_depth", 0)
            window["avg_mean_depth"] = window.pop("mean_depth", 0)
            window["avg_median_depth"] = window.pop("median_depth", 0)
            window["avg_depth_skew"] = window.pop("depth_skew", 0)

        return aggregated

    def compute_dependency_complexity(self, doc, window_size=DEFAULT_WINDOW_SIZE):
        sentence_metrics = []

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

            sentence_metrics.append(
                {
                    "sentence_text": sent.text,
                    "avg_dependents_per_head": {
                        "main_clause": round(statistics.mean(dependents_per_head["main_clause"]), 2)
                        if dependents_per_head["main_clause"]
                        else 0,
                        "subordinate_clause": round(statistics.mean(dependents_per_head["subordinate_clause"]), 2)
                        if dependents_per_head["subordinate_clause"]
                        else 0,
                        "coordinate_clause": round(statistics.mean(dependents_per_head["coordinate_clause"]), 2)
                        if dependents_per_head["coordinate_clause"]
                        else 0,
                    },
                    "avg_max_dependents_per_head": max(all_dependents, default=0),
                    "avg_mean_dependency_distance": round(statistics.mean(dependency_distances), 2)
                    if dependency_distances
                    else 0,
                }
            )

        return aggregate_windows(sentence_metrics, window_size)

    def analyze_document(self, doc, window_size=DEFAULT_WINDOW_SIZE):
        """
        Convenience wrapper to compute all syntax metrics for a spaCy Doc.

        Returns:
            dict with clause_metrics, clause_embedding_metrics, dependency_metrics, window_size, num_sentences.

        Example:
            >>> from x_configs import load_spacy_model
            >>> nlp = load_spacy_model()
            >>> doc = nlp("One. Two. Three.")
            >>> SyntaxAnalyzer(nlp).analyze_document(doc)["clause_metrics"][0]["avg_counts"]["main"]
            1
        """
        sentences = list(doc.sents)
        clause_metrics = self.compute_clause_metrics(doc, window_size=window_size)
        clause_embed_metrics = self.compute_clause_embedding_depth(doc, window_size=window_size)
        dependency_metrics = self.compute_dependency_complexity(doc, window_size=window_size)

        return {
            "window_size": window_size,
            "num_sentences": len(sentences),
            "clause_metrics": clause_metrics,
            "clause_embedding_metrics": clause_embed_metrics,
            "dependency_metrics": dependency_metrics,
        }


class SyntaxVisualiser:
    def __init__(self, json_file: str):
        self.json_file = Path(json_file)
        self.data = load_json(self.json_file)

    def plot_clause_complexity(self, save_path: Optional[Path] = None):
        snippets = [(c["start_sentence"] + c["end_sentence"]) // 2 for c in self.data["clause_metrics"]]
        sub_counts = [c["avg_counts"]["subordinate"] for c in self.data["clause_metrics"]]
        coord_counts = [c["avg_counts"]["coordinate"] for c in self.data["clause_metrics"]]

        plt.figure(figsize=(12, 6))
        plt.bar(snippets, sub_counts, label="Subordinate", color="#377eb8")
        plt.bar(snippets, coord_counts, bottom=sub_counts, label="Coordinate", color="#e41a1c")

        plt.xlabel("Snippet midpoint (sentence index)")
        plt.ylabel("Average clause count")
        plt.title(f"Clause Composition: {self.data['filename']}")
        plt.legend()
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_clause_depth_metrics(self, save_path: Optional[Path] = None):
        metrics = self.data["clause_embedding_metrics"]
        snippets = [(c["start_sentence"] + c["end_sentence"]) // 2 for c in metrics]

        plt.figure(figsize=(12, 6))
        plt.plot(snippets, [c["avg_max_depth"] for c in metrics], label="Max Depth", linewidth=2)
        plt.plot(snippets, [c["avg_mean_depth"] for c in metrics], label="Mean Depth", linestyle="--")
        plt.plot(snippets, [c["avg_median_depth"] for c in metrics], label="Median Depth", linestyle=":")
        plt.plot(snippets, [c["avg_depth_skew"] for c in metrics], label="Depth Skew", linestyle="-.", alpha=0.7)

        plt.xlabel("Snippet midpoint (sentence index)")
        plt.ylabel("Depth Value")
        plt.title(f"Syntactic Depth: {self.data['filename']}")
        plt.legend()
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_clause_depth_area(self, save_path: Optional[Path] = None):
        metrics = self.data["clause_embedding_metrics"]
        snippets = [(c["start_sentence"] + c["end_sentence"]) // 2 for c in metrics]

        plt.figure(figsize=(12, 6))
        plt.stackplot(
            snippets,
            [c["avg_median_depth"] for c in metrics],
            [c["avg_mean_depth"] for c in metrics],
            [c["avg_max_depth"] for c in metrics],
            labels=["Median Depth", "Mean Depth", "Max Depth"],
            alpha=0.7,
        )

        plt.xlabel("Snippet midpoint (sentence index)")
        plt.ylabel("Depth Value")
        plt.title(f"Stacked Syntactic Depth: {self.data['filename']}")
        plt.legend(loc="upper left")
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_dependency_complexity(self, save_path: Optional[Path] = None):
        metrics = self.data["dependency_metrics"]
        snippets = [(c["start_sentence"] + c["end_sentence"]) // 2 for c in metrics]

        mean_dep_dist = [c["avg_mean_dependency_distance"] for c in metrics]
        max_dep = [c["avg_max_dependents_per_head"] for c in metrics]
        main_dep = [c["avg_dependents_per_head"]["main_clause"] for c in metrics]
        sub_dep = [c["avg_dependents_per_head"]["subordinate_clause"] for c in metrics]
        coord_dep = [c["avg_dependents_per_head"]["coordinate_clause"] for c in metrics]

        plt.figure(figsize=(12, 6))
        plt.bar(snippets, mean_dep_dist, color="#b2df8a", alpha=0.6, label="Mean Dependency Distance")
        plt.plot(snippets, main_dep, label="Main Clause", color="#1f78b4", linewidth=2)
        plt.plot(snippets, sub_dep, label="Subordinate Clause", color="#33a02c", linestyle="--")
        plt.plot(snippets, coord_dep, label="Coordinate Clause", color="#e31a1c", linestyle=":")
        plt.plot(snippets, max_dep, label="Max Dependents/Head", color="#ff7f00", linestyle="-.", alpha=0.7)

        plt.xlabel("Snippet midpoint (sentence index)")
        plt.ylabel("Complexity Metric Value")
        plt.title(f"Dependency Complexity: {self.data['filename']}")
        plt.legend(loc="upper left")
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
        plt.close()
