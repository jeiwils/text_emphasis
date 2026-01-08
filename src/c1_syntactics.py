import statistics
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from x_configs import DEFAULT_WINDOW_SIZE
from z_utils import aggregate_windows, graph_path, load_json, sliding_windows

"""
Sentence-level grammar metrics (clauses, syntactic depth, dependency complexity).


{
  "meta": {"window_size": 3, "num_sentences": 120},

  "sentences": [
    {
      "sentence_id": 0,
      "clause_counts": {
        "main": 1, 
        ...
        },
      "clause_ratios": {
        "subordination_ratio": 0.0, 
        ...
        },
      "max_depth": 4, 
      "mean_depth": 2.1, 
      "median_depth": 2.0, 
      "depth_skew": 0.1,
      "avg_dependents_per_head": {
        "main_clause": 2.1, 
        ...
        },
      "avg_max_dependents_per_head": 5, 
      "avg_mean_dependency_distance": 1.4
    },
    ...
  ],

  "windows": [
    {
      "start_sentence": 0,
      "end_sentence": 2,
      "clause_counts": { # averaged over the window
        ...
        }, 
      "clause_ratios": { # averaged over the window
        ...
        },  
      "max_depth": 3.3, 
      "mean_depth": 1.9, 
      "median_depth": 1.8, 
      "depth_skew": 0.2,
      "avg_dependents_per_head": {
        ...
        }, 
      "avg_max_dependents_per_head": 4.3,
      "avg_mean_dependency_distance": 1.5,
      "avg_counts": {
        ...
        }, 
      "avg_ratios": {
        ...
        }, 
      "avg_max_depth": 3.3, 
      "avg_mean_depth": 1.9,
      "avg_median_depth": 1.8, 
      "avg_depth_skew": 0.2
    },
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
            token_count = len([t for t in sent if not t.is_punct])
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

        return sentence_depths, aggregated

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

        return sentence_metrics, aggregate_windows(sentence_metrics, window_size)

    def analyze_document(self, doc, window_size=DEFAULT_WINDOW_SIZE):
        """
        Convenience wrapper to compute all syntax metrics for a spaCy Doc.

        Returns:
            dict with meta, sentences (per-sentence combined syntax), windows (aggregated).
            Window rows are built with aggregate_windows: contiguous spans of size `window_size`
            are averaged across numeric fields and tagged with inclusive start/end sentence indices.

        Example:
            >>> from x_configs import load_spacy_model
            >>> nlp = load_spacy_model()
            >>> doc = nlp("One. Two. Three.")
            >>> SyntaxAnalyzer(nlp).analyze_document(doc)["sentences"][0]["clause_counts"]["main"]
            1.0
        """
        sentences = list(doc.sents)
        clause_sent, clause_windows = self.compute_clause_metrics(doc, window_size=window_size)
        depth_sent, depth_windows = self.compute_clause_embedding_depth(doc, window_size=window_size)
        dep_sent, dep_windows = self.compute_dependency_complexity(doc, window_size=window_size)

        combined_sentences = []
        for idx, sent in enumerate(sentences):
            clause_payload = clause_sent[idx] if idx < len(clause_sent) else {}
            depth_payload = depth_sent[idx] if idx < len(depth_sent) else {}
            dep_payload = dep_sent[idx] if idx < len(dep_sent) else {}

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
                total_tokens = sum(sent.get("token_count", 0) for sent in window_sents)
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

        return {
            "meta": {
                "window_size": window_size,
                "num_sentences": len(sentences),
            },
            "sentences": combined_sentences,
            "windows": windows,
        }


class SyntaxVisualiser:
    def __init__(self, json_file: str):
        self.json_file = Path(json_file)
        self.data = load_json(self.json_file)
        self.windows = self.data.get("windows", [])

    def _text_label(self) -> str:
        return self.data.get("filename") or self.data.get("meta", {}).get("filename", "")

    def _clause_metrics(self):
        if "clause_metrics" in self.data:
            return self.data["clause_metrics"]
        return [w for w in self.windows if "avg_counts" in w and "avg_ratios" in w]

    def _depth_metrics(self):
        if "clause_embedding_metrics" in self.data:
            return self.data["clause_embedding_metrics"]
        return [w for w in self.windows if "avg_max_depth" in w]

    def _dependency_metrics(self):
        if "dependency_metrics" in self.data:
            return self.data["dependency_metrics"]
        return [w for w in self.windows if "avg_mean_dependency_distance" in w]

    def plot_clause_complexity(self, save_path: Optional[Path] = None):
        clause_metrics = self._clause_metrics()
        if not clause_metrics:
            return

        snippets = [(c["start_sentence"] + c["end_sentence"]) // 2 for c in clause_metrics]
        sub_counts = [c["avg_counts"]["subordinate"] for c in clause_metrics]
        coord_counts = [c["avg_counts"]["coordinate"] for c in clause_metrics]

        plt.figure(figsize=(12, 6))
        plt.bar(snippets, sub_counts, label="Subordinate", color="#377eb8")
        plt.bar(snippets, coord_counts, bottom=sub_counts, label="Coordinate", color="#e41a1c")

        plt.xlabel("Snippet midpoint (sentence index)")
        plt.ylabel("Average clause count")
        plt.title(f"Clause Composition: {self._text_label()}")
        plt.legend()
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_clause_depth_metrics(self, save_path: Optional[Path] = None):
        metrics = self._depth_metrics()
        if not metrics:
            return
        snippets = [(c["start_sentence"] + c["end_sentence"]) // 2 for c in metrics]

        plt.figure(figsize=(12, 6))
        plt.plot(snippets, [c["avg_max_depth"] for c in metrics], label="Max Depth", linewidth=2)
        plt.plot(snippets, [c["avg_mean_depth"] for c in metrics], label="Mean Depth", linestyle="--")
        plt.plot(snippets, [c["avg_median_depth"] for c in metrics], label="Median Depth", linestyle=":")
        plt.plot(snippets, [c["avg_depth_skew"] for c in metrics], label="Depth Skew", linestyle="-.", alpha=0.7)

        plt.xlabel("Snippet midpoint (sentence index)")
        plt.ylabel("Depth Value")
        plt.title(f"Syntactic Depth: {self._text_label()}")
        plt.legend()
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_clause_depth_area(self, save_path: Optional[Path] = None):
        metrics = self._depth_metrics()
        if not metrics:
            return
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
        plt.title(f"Stacked Syntactic Depth: {self._text_label()}")
        plt.legend(loc="upper left")
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_dependency_complexity(self, save_path: Optional[Path] = None):
        metrics = self._dependency_metrics()
        if not metrics:
            return
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
        plt.title(f"Dependency Complexity: {self._text_label()}")
        plt.legend(loc="upper left")
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
        plt.close()
