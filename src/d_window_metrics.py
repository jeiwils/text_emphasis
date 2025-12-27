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
   Each file matches the c0 meta/sentences/windows schema (log-probs, surprisal, etc.).
2) `run_windowed_metrics` reads those corpus JSONs, recomputes sentence-level spaCy docs,
   and writes combined window metrics to `data/texts/window_metrics/<category>/<name>_metrics.json`
   with nested blocks: meta, syntax (meta/sentences/windows + heavy), lexico_semantics (same shape),
   discourse (same shape), information_content_metrics (c0 windows), and topic_metrics.
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
            meta_block = data.get("meta", {}) if isinstance(data, dict) else {}
            text_content = data.get("text") or _load_text_from_cleaned(subdir, file)

            lex_analyzer = LexicoSemanticsAnalyzer(nlp)
            doc = nlp(text_content or "")
            num_sentences = len(list(doc.sents))
            doc = nlp(text_content or "")  # reset iterator after counting
            syntax_metrics = syntax_analyzer.analyze_document(doc, window_size=window_size)
            lex_metrics = lex_analyzer.analyze_document(
                doc,
                window_size=window_size,
                mattr_window_size=mattr_window_size,
            )
            discourse_metrics = discourse_analyzer.analyze_text(text_content or "", window_size=window_size)

            info_content_metrics = data.get("windows", [])

            topics_data = load_topics_json(file)
            topic_mentions = collect_topic_mentions(topics_data)
            topic_metrics = build_topic_window_metrics(
                topic_mentions,
                syntax_metrics.get("windows", []),
            )

            result = {
                "meta": {
                    "filename": meta_block.get("filename", file.name),
                    "model": meta_block.get("model", ""),
                    "window_size": window_size,
                    "num_sentences": num_sentences,
                },
                "syntax": syntax_metrics,
                "lexico_semantics": lex_metrics,
                "information_content_metrics": info_content_metrics,
                "discourse": discourse_metrics,
                "topic_metrics": topic_metrics,
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
