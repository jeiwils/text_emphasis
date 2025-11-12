import spacy
import statistics
from itertools import islice
from .z_utils import sliding_windows, processed_text_path
import json
from pathlib import Path
from typing import Optional

from statistics import mean



"""
takes .txt from each subdir of cleaned_texts
processed them
outputs 


"""



class SyntaxAnalyzer:
    def __init__(self, model='en_core_web_sm'):
        self.nlp = spacy.load(model)


    # ----------------------------
    # Clause Embedding Metrics
    # ----------------------------
    def compute_clause_embedding_metrics(self, doc, window_size=3):
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
                    "mean_depth": statistics.mean(sent_depths),
                    "median_depth": statistics.median(sent_depths),
                    "depth_skew": statistics.mean(sent_depths) - statistics.median(sent_depths)
                })

        # Compute sliding window averages
        windowed_metrics = []
        for window in sliding_windows(sentence_depths, window_size):
            avg_max = round(statistics.mean(d["max_depth"] for d in window), 2)
            avg_mean = round(statistics.mean(d["mean_depth"] for d in window), 2)
            avg_median = round(statistics.mean(d["median_depth"] for d in window), 2)
            avg_skew = round(statistics.mean(d["depth_skew"] for d in window), 2)
            windowed_metrics.append({
                "avg_max_depth": avg_max,
                "avg_mean_depth": avg_mean,
                "avg_median_depth": avg_median,
                "avg_depth_skew": avg_skew
            })

        return windowed_metrics

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
                "main": main_counts,
                "subordinate": sub_counts,
                "coordinate": coord_counts,
                "subordination_ratio": round(sub_to_main_ratio, 2),
                "coordination_ratio": round(coord_to_main_ratio, 2)
            })

        # Sliding window averages
        windowed_metrics = []
        for window in sliding_windows(sentence_metrics, window_size):
            avg_main = round(statistics.mean(d["main"] for d in window), 2)
            avg_sub = round(statistics.mean(d["subordinate"] for d in window), 2)
            avg_coord = round(statistics.mean(d["coordinate"] for d in window), 2)
            avg_sub_ratio = round(statistics.mean(d["subordination_ratio"] for d in window), 2)
            avg_coord_ratio = round(statistics.mean(d["coordination_ratio"] for d in window), 2)

            windowed_metrics.append({
                "avg_counts": {
                    "main": avg_main,
                    "subordinate": avg_sub,
                    "coordinate": avg_coord
                },
                "avg_ratios": {
                    "subordination_ratio": avg_sub_ratio,
                    "coordination_ratio": avg_coord_ratio
                }
            })

        return windowed_metrics

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
                "max_dependents_per_head": max(all_dependents, default=0),
                "mean_dependency_distance": round(statistics.mean(dependency_distances), 2) if dependency_distances else 0
            })

        # Sliding window averages
        windowed_metrics = []
        for window in sliding_windows(sentence_metrics, window_size):
            avg_main = round(statistics.mean(d["avg_dependents_per_head"]["main_clause"] for d in window), 2)
            avg_sub = round(statistics.mean(d["avg_dependents_per_head"]["subordinate_clause"] for d in window), 2)
            avg_coord = round(statistics.mean(d["avg_dependents_per_head"]["coordinate_clause"] for d in window), 2)
            avg_max = round(statistics.mean(d["max_dependents_per_head"] for d in window), 2)
            avg_dep_dist = round(statistics.mean(d["mean_dependency_distance"] for d in window), 2)

            windowed_metrics.append({
                "avg_dependents_per_head": {
                    "main_clause": avg_main,
                    "subordinate_clause": avg_sub,
                    "coordinate_clause": avg_coord
                },
                "avg_max_dependents_per_head": avg_max,
                "avg_mean_dependency_distance": avg_dep_dist
            })

        return windowed_metrics






def run_syntax_analysis(window_size=3, use_existing=True):
    """
    Efficient syntax analysis for all cleaned texts.
    Computes sentence-level metrics once per doc, then aggregates windows.
    Saves JSON to 'window_metrics' directory.
    """
    analyzer = SyntaxAnalyzer()
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

            # --- Compute sentence-level metrics once ---
            clause_sent_metrics = analyzer.compute_clause_metrics(doc, window_size=1)
            clause_embed_sent_metrics = analyzer.compute_clause_embedding_metrics(doc, window_size=1)
            dependency_sent_metrics = analyzer.compute_dependency_complexity(doc, window_size=1)

            # --- Aggregate metrics in sliding windows ---
            def aggregate_windows(sent_metrics):
                windows = []
                for i in range(0, num_sentences - window_size + 1):
                    window_sents = sent_metrics[i:i + window_size]

                    # Aggregate metrics
                    agg = {}
                    for key in window_sents[0]:
                        if isinstance(window_sents[0][key], dict):
                            # Nested dict
                            agg[key] = {k: round(mean(d[key][k] for d in window_sents), 2)
                                        for k in window_sents[0][key]}
                        else:
                            # Scalar metric
                            agg[key] = round(mean(d[key] for d in window_sents), 2)

                    # Add metadata
                    agg["start_sentence"] = i
                    agg["end_sentence"] = i + window_size - 1
                    agg["text_snippet"] = " ".join([s.text for s in sentences[i:i+window_size]])[:200]
                    windows.append(agg)
                return windows

            clause_metrics = aggregate_windows(clause_sent_metrics)
            clause_embed_metrics = aggregate_windows(clause_embed_sent_metrics)
            dependency_metrics = aggregate_windows(dependency_sent_metrics)

            # --- Save JSON ---
            result = {
                "filename": file.name,
                "num_sentences": num_sentences,
                "clause_metrics": clause_metrics,
                "clause_embedding_metrics": clause_embed_metrics,
                "dependency_metrics": dependency_metrics
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(f"✅ Saved syntax metrics to {output_file}")

    print("🎉 All done.")


# Example call
if __name__ == "__main__":
    run_syntax_analysis(window_size=3, use_existing=True)