import spacy
import statistics
from itertools import islice
from .z_utils import sliding_windows, processed_text_path, load_json, graph_path, aggregate_windows
from pathlib import Path
from typing import Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from statistics import mean



"""


"""



class SyntaxAnalyzer:
    def __init__(self, nlp):
        self.nlp = nlp

    # ----------------------------
    # Clause Counts Metrics
    # ----------------------------
    def compute_clause_metrics(self, doc, window_size=3):
        sentence_metrics = []

        for sent in doc.sents:
            main_counts = sub_counts = coord_counts = 0
            for token in sent:
                if token.dep_ == 'ROOT':
                    main_counts += 1
                elif token.dep_ in ('advcl', 'ccomp', 'xcomp'):
                    sub_counts += 1
                elif token.dep_ == 'conj':
                    coord_counts += 1

            sub_to_main_ratio = sub_counts / main_counts if main_counts else 0
            coord_to_main_ratio = coord_counts / main_counts if main_counts else 0

            sentence_metrics.append({
                "avg_counts": {
                    "main": main_counts,
                    "subordinate": sub_counts,
                    "coordinate": coord_counts
                },
                "avg_ratios": {
                    "subordination_ratio": round(sub_to_main_ratio, 2),
                    "coordination_ratio": round(coord_to_main_ratio, 2)
                }
            })

        # Aggregate over windows
        return aggregate_windows(sentence_metrics, window_size)

    # ----------------------------
    # Clause Embedding Depth (syntactic depth)
    # ----------------------------
    def compute_clause_embedding_depth(self, doc, window_size=3):
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
                sentence_depths.append({
                    "max_depth": max(sent_depths),
                    "mean_depth": round(statistics.mean(sent_depths), 2),
                    "median_depth": round(statistics.median(sent_depths), 2),
                    "depth_skew": round(statistics.mean(sent_depths) - statistics.median(sent_depths), 2)
                })

        # Aggregate over windows
        return aggregate_windows(sentence_depths, window_size)

    # ----------------------------
    # Dependency Complexity Metrics
    # ----------------------------
    def compute_dependency_complexity(self, doc, window_size=3):
        sentence_metrics = []

        for sent in doc.sents:
            dependents_per_head = {"main_clause": [], "subordinate_clause": [], "coordinate_clause": []}
            dependency_distances = []

            for token in sent:
                num_dependents = len(list(token.children))
                dependency_distances.extend([abs(token.i - child.i) for child in token.children])

                if token.dep_ == 'ROOT':
                    dependents_per_head['main_clause'].append(num_dependents)
                elif token.dep_ in ('advcl', 'ccomp', 'xcomp'):
                    dependents_per_head['subordinate_clause'].append(num_dependents)
                elif token.dep_ == 'conj':
                    dependents_per_head['coordinate_clause'].append(num_dependents)

            all_dependents = dependents_per_head['main_clause'] + dependents_per_head['subordinate_clause'] + dependents_per_head['coordinate_clause']

            sentence_metrics.append({
                "avg_dependents_per_head": {
                    "main_clause": round(statistics.mean(dependents_per_head['main_clause']), 2) if dependents_per_head['main_clause'] else 0,
                    "subordinate_clause": round(statistics.mean(dependents_per_head['subordinate_clause']), 2) if dependents_per_head['subordinate_clause'] else 0,
                    "coordinate_clause": round(statistics.mean(dependents_per_head['coordinate_clause']), 2) if dependents_per_head['coordinate_clause'] else 0
                },
                "avg_max_dependents_per_head": max(all_dependents, default=0),
                "avg_mean_dependency_distance": round(statistics.mean(dependency_distances), 2) if dependency_distances else 0
            })

        # Aggregate over windows
        return aggregate_windows(sentence_metrics, window_size)












class SyntaxVisualiser:
    def __init__(self, json_file: str):
        self.json_file = Path(json_file)
        self.data = load_json(self.json_file)

    def plot_clause_complexity(self, save_path: Optional[Path] = None):
        """Stacked bar chart of subordinate vs coordinate clause counts."""
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
            save_path.parent.mkdir(parents=True, exist_ok=True)  # ✅ ensure folder exists
            plt.savefig(save_path, dpi=300)
        plt.close()



    def plot_clause_depth_metrics(self, save_path: Optional[Path] = None):
        """Line plot of syntactic depth metrics over text."""
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
            save_path.parent.mkdir(parents=True, exist_ok=True)  # ✅ ensure folder exists
            plt.savefig(save_path, dpi=300)
        plt.close()



    def plot_clause_depth_area(self, save_path: Optional[Path] = None):
        """Stacked area plot of max/mean/median syntactic depth."""
        metrics = self.data["clause_embedding_metrics"]
        snippets = [(c["start_sentence"] + c["end_sentence"]) // 2 for c in metrics]

        plt.figure(figsize=(12, 6))
        plt.stackplot(
            snippets,
            [c["avg_median_depth"] for c in metrics],
            [c["avg_mean_depth"] for c in metrics],
            [c["avg_max_depth"] for c in metrics],
            labels=["Median Depth", "Mean Depth", "Max Depth"],
            alpha=0.7
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
        """Line + bar plot showing dependency complexity across windows."""
        metrics = self.data["dependency_metrics"]
        snippets = [(c["start_sentence"] + c["end_sentence"]) // 2 for c in metrics]

        # Extract values
        mean_dep_dist = [c["avg_mean_dependency_distance"] for c in metrics]
        max_dep = [c["avg_max_dependents_per_head"] for c in metrics]
        main_dep = [c["avg_dependents_per_head"]["main_clause"] for c in metrics]
        sub_dep = [c["avg_dependents_per_head"]["subordinate_clause"] for c in metrics]
        coord_dep = [c["avg_dependents_per_head"]["coordinate_clause"] for c in metrics]

        plt.figure(figsize=(12, 6))

        # Bar for dependency distance (overall)
        plt.bar(snippets, mean_dep_dist, color="#b2df8a", alpha=0.6, label="Mean Dependency Distance")

        # Lines for dependents per head (clauses)
        plt.plot(snippets, main_dep, label="Main Clause", color="#1f78b4", linewidth=2)
        plt.plot(snippets, sub_dep, label="Subordinate Clause", color="#33a02c", linestyle="--")
        plt.plot(snippets, coord_dep, label="Coordinate Clause", color="#e31a1c", linestyle=":")

        # Line for max dependents per head
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








def run_syntax_analysis(window_size=3, use_existing=True):
    """
    Efficient syntax analysis for all cleaned texts.
    Computes sentence-level metrics once per doc, then aggregates windows.
    Saves JSON to 'window_metrics' directory.
    """
    nlp = spacy.load("en_core_web_sm")
    analyzer = SyntaxAnalyzer(nlp)
    cleaned_root = processed_text_path("cleaned")
    output_root = processed_text_path("window")

    for subdir in cleaned_root.iterdir():
        if not subdir.is_dir():
            continue
        print(f"Processing category: {subdir.name}")
        out_subdir = output_root / subdir.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        for file in subdir.glob("*.txt"):
            output_file = out_subdir / f"{file.stem}_syntax.json"
            if use_existing and output_file.exists():
                print(f"Skipping {file.name} (exists)")
                continue

            text = file.read_text(encoding="utf-8")
            doc = analyzer.nlp(text)
            sentences = list(doc.sents)
            num_sentences = len(sentences)

            print(f"→ Computing syntax metrics for {file.name} ({num_sentences} sentences)...")

            # --- Compute metrics with window aggregation ---
            clause_metrics = analyzer.compute_clause_metrics(doc, window_size=window_size)
            clause_embed_metrics = analyzer.compute_clause_embedding_depth(doc, window_size=window_size)
            dependency_metrics = analyzer.compute_dependency_complexity(doc, window_size=window_size)




            # --- Save separate JSONs ---
            (output_file.parent / f"{file.stem}_clause_counts.json").write_text(
                json.dumps(clause_metrics, indent=2), encoding="utf-8"
            )
            (output_file.parent / f"{file.stem}_clause_depth.json").write_text(
                json.dumps(clause_embed_metrics, indent=2), encoding="utf-8"
            )
            (output_file.parent / f"{file.stem}_clause_dependencies.json").write_text(
                json.dumps(dependency_metrics, indent=2), encoding="utf-8"
            )



            print(f"✅ Saved clause_counts, clause_depth, and clause_dependencies for {file.name}")


    print("🎉 All done.")






if __name__ == "__main__":

    # Step 1: Compute and save syntax metrics
    run_syntax_analysis(window_size=3, use_existing=True)

    window_folder = processed_text_path("window")
    json_files = list(window_folder.rglob("*.json"))

    for jf in json_files:
        visualiser = SyntaxVisualiser(jf)

        # Preserve relative folder structure
        rel_path = jf.relative_to(window_folder).with_suffix("")
        subdir = rel_path.parent

        # Clause complexity plots
        clause_dir = graph_path("syntactic", subfolder="clause") / subdir
        clause_dir.mkdir(parents=True, exist_ok=True) 
        visualiser.plot_clause_complexity(save_path=clause_dir / f"{jf.stem}_clause_complexity_bar.png")

        # Clause embedding depth plots
        dep_dir = graph_path("syntactic", subfolder="clause") / subdir
        dep_dir.mkdir(parents=True, exist_ok=True)  
        visualiser.plot_clause_depth_metrics(save_path=dep_dir / f"{jf.stem}_clause_depth_line.png")
        visualiser.plot_clause_depth_area(save_path=dep_dir / f"{jf.stem}_clause_depth_area.png")

        # Dependency complexity plots
        dep_complex_dir = graph_path("syntactic", subfolder="dependency") / subdir
        dep_complex_dir.mkdir(parents=True, exist_ok=True)
        visualiser.plot_dependency_complexity(save_path=dep_complex_dir / f"{jf.stem}_dependency_complexity.png")


        

        print(f"✅ Saved all plots for {jf.name} in {subdir}")

