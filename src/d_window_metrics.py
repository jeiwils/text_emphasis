import json
from pathlib import Path

from x_configs import DEFAULT_WINDOW_SIZE, load_spacy_model
from .c0_log_prob_metrics import WholeTextMetrics
from .c1_syntactics import SyntaxAnalyzer
from .c2_lexico_semantics import LexicoSemanticsAnalyzer
from .c3_discourse import DiscourseAnalyzer
from .e_variance_report import (
    build_topic_window_metrics,
    collect_topic_mentions,
    load_topics_json,
)
from .z_utils import processed_text_path

"""
Orchestrator for running all c-layer metrics and saving outputs.

Flow:
1) `run_corpus_metrics` reads cleaned texts from `data/texts/cleaned_texts/<category>/*.txt`
   and writes corpus metrics JSON to `data/texts/corpus_analytics/<category>/<name>_metrics.json`.
   Each file matches the c0 example dict (log-probs, surprisal, top_words, etc.).
2) `run_windowed_metrics` reads those corpus JSONs, recomputes sentence-level spaCy docs,
   and writes combined window metrics to `data/texts/window_metrics/<category>/<name>_metrics.json`
   with keys: syntax (clause/depth/dependency), lexical density/frequency/MATTR, information content,
   discourse window metrics, semantic roles/structures, and topic metrics.

Example window metrics JSON (truncated):
{
  "filename": "book1.txt",
  "model": "gpt2",
  "window_size": 3,
  "num_sentences": 120,
  "clause_metrics": [...],
  "dependency_complexity_metrics": [...],
  "lexical_density_metrics": [...],
  "information_content_metrics": [...],
  "discourse_metrics": [...],
  "topic_metrics": [{"topic_id": 3, "count": 4, "start_sentence": 0, "end_sentence": 2}, ...],
  "lexical_diversity_windowed": [{"mattr_score": 0.71, "start_sentence": 0, "end_sentence": 2}, ...]
}
"""


def run_corpus_metrics(window_size=DEFAULT_WINDOW_SIZE, use_existing=True):
    """
    Compute and save corpus-level log-prob/surprisal metrics for all cleaned texts.
    """
    metrics = WholeTextMetrics()
    nlp = load_spacy_model()

    cleaned_root = processed_text_path("cleaned")
    output_root = processed_text_path("corpus")
    output_root.mkdir(parents=True, exist_ok=True)

    for subdir in cleaned_root.iterdir():
        if not subdir.is_dir():
            continue
        print(f"Processing category: {subdir.name}")

        out_subdir = output_root / subdir.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        for file in subdir.glob("*.txt"):
            output_file = out_subdir / f"{file.stem}_metrics.json"
            if use_existing and output_file.exists():
                print(f"Skipping {file.name} (exists)")
                continue

            text = file.read_text(encoding="utf-8")
            result = metrics.build_metrics_for_text(text, file.name, nlp=nlp, window_size=window_size)
            output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Saved corpus metrics for {file.name}")


def _load_text_from_cleaned(category: Path, metrics_file: Path) -> str:
    """
    Derive the cleaned text path from a metrics filename (assumes *_metrics.json convention).
    """
    cleaned_root = processed_text_path("cleaned")
    base_name = metrics_file.stem.replace("_metrics", "")
    candidate = cleaned_root / category.name / f"{base_name}.txt"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return ""


def run_windowed_metrics(window_size=DEFAULT_WINDOW_SIZE, mattr_window_size=50, use_existing=True):
    """
    Compute window-level metrics by combining syntax, lexico-semantic, discourse, and topic metrics.
    Requires corpus outputs from run_corpus_metrics.
    """
    corpus_root = processed_text_path("corpus")
    output_root = processed_text_path("window")
    output_root.mkdir(parents=True, exist_ok=True)

    nlp = load_spacy_model()
    syntax_analyzer = SyntaxAnalyzer(nlp)
    discourse_analyzer = DiscourseAnalyzer(nlp)

    for subdir in corpus_root.iterdir():
        if not subdir.is_dir():
            continue
        print(f"Processing category: {subdir.name}")
        out_subdir = output_root / subdir.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        for file in subdir.glob("*.json"):
            output_file = out_subdir / file.name
            if use_existing and output_file.exists():
                print(f"Skipping {file.name} (exists)")
                continue

            data = json.load(file.open("r", encoding="utf-8"))
            text_content = data.get("text") or _load_text_from_cleaned(subdir, file)

            corpus_word_freqs = data.get("word_frequencies") or {w: f for w, f in data.get("top_words", [])}

            lex_analyzer = LexicoSemanticsAnalyzer(nlp, corpus_freqs=corpus_word_freqs)
            doc = nlp(text_content or "")
            num_sentences = len(list(doc.sents))
            doc = nlp(text_content or "")  # reset iterator after counting
            windowed_mattr_metrics = lex_analyzer.compute_windowed_mattr(
                doc,
                window_size=window_size,
                mattr_window_size=mattr_window_size,
            )

            syntax_metrics = syntax_analyzer.analyze_document(doc, window_size=window_size)
            clause_metrics = syntax_metrics["clause_metrics"]
            clause_embed_metrics = syntax_metrics["clause_embedding_metrics"]
            dep_complexity_metrics = syntax_metrics["dependency_metrics"]

            avg_word_freq_metrics = lex_analyzer.compute_avg_word_frequency(doc, window_size=window_size)
            lexical_density_metrics = lex_analyzer.analyze_lexical_density(doc, window_size=window_size)
            lexical_information_content = lex_analyzer.analyze_information_content(
                doc, word_frequencies=corpus_word_freqs, window_size=window_size
            )
            cohesion_metrics = discourse_analyzer.analyze_cohesion(doc, window_size=window_size)
            semantic_role_metrics = lex_analyzer.analyze_semantic_roles(doc, window_size=window_size)

            info_content_metrics = data.get("sentence_surprisal_metrics_windowed", [])

            semantic_structures = lex_analyzer.extract_semantic_structures(doc, window_size=window_size)
            _, _, discourse_metrics = discourse_analyzer.compute_sentence_metrics(
                doc,
                window_size=window_size,
            )

            topics_data = load_topics_json(file)
            topic_mentions = collect_topic_mentions(topics_data)
            topic_metrics = build_topic_window_metrics(topic_mentions, clause_metrics)

            result = {
                "filename": data.get("filename", file.name),
                "model": data.get("model", ""),
                "window_size": window_size,
                "num_sentences": num_sentences,
                "clause_metrics": clause_metrics,
                "clause_embedding_metrics": clause_embed_metrics,
                "dependency_complexity_metrics": dep_complexity_metrics,
                "avg_word_freq_metrics": avg_word_freq_metrics,
                "lexical_density_metrics": lexical_density_metrics,
                "lexical_information_content": lexical_information_content,
                "cohesion_metrics": cohesion_metrics,
                "semantic_role_metrics": semantic_role_metrics,
                "information_content_metrics": info_content_metrics,
                "semantic_structures": semantic_structures,
                "discourse_metrics": discourse_metrics,
                "topic_metrics": topic_metrics,
                "lexical_diversity_windowed": windowed_mattr_metrics,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(f"Saved windowed metrics for {file.name}")
    print("All done.")


def run_all_metrics(window_size=DEFAULT_WINDOW_SIZE, mattr_window_size=50, use_existing=True):
    """
    Full orchestrator: corpus metrics first, then windowed metrics.
    """
    run_corpus_metrics(window_size=window_size, use_existing=use_existing)
    run_windowed_metrics(window_size=window_size, mattr_window_size=mattr_window_size, use_existing=use_existing)


if __name__ == "__main__":
    run_all_metrics()
