import json
from pathlib import Path
from typing import List, Tuple
from spacy.tokens import Doc

from x_configs import DEFAULT_WINDOW_SIZE, load_spacy_model
from .a_preprocessing_cleaning import preprocess_all_pdfs
from .b1_concept_embeddings import generate_embeddings
from .b2_log_prob_metrics import WholeTextMetrics
from .b3_topic_modeling import run_topic_modelling
from .c1_syntactics import SyntaxAnalyzer
from .c2_lexico_semantics import LexicoSemanticsAnalyzer
from .c3_discourse import DiscourseAnalyzer
from .z_utils import processed_text_path



"""


Orchestrator for running all b- and c-layer metrics and saving outputs.

Flow:
1) `run_preprocessing` converts raw PDFs to cleaned/normalised text and segmented JSONL under data/texts/.
2) `run_concept_embeddings` reads normalised texts from `data/texts/normalised_texts/<category>/*_normalised.json`
   and writes phrases + embeddings to `data/embeddings/concept_embeddings/<name>/`.
3) `run_topic_modelling` reads normalised-segmented texts from
   `data/texts/normalised_segmented_texts/<category>/*.jsonl`
   and writes topics JSON to `data/topic_modelling/<category>/<name>_topics.json`.
4) `run_corpus_metrics` reads cleaned texts from `data/texts/cleaned_texts/<category>/*.json`
   and writes corpus metrics JSON to `data/texts/corpus_analytics/<category>/<name>_metrics.json`.
   Each file matches the c0 meta/sentences/windows schema (log-probs, surprisal, etc.).
5) `run_windowed_metrics` reads those corpus JSONs, recomputes sentence-level spaCy docs
   from cleaned-segmented texts,
   and writes combined window metrics to `data/texts/window_metrics/<category>/<name>_metrics.json`
   with nested blocks: meta, syntax (meta/sentences/windows + heavy), lexico_semantics (same shape),
   discourse (same shape), information_content_metrics (c0 windows)
"""


def run_concept_embeddings(top_n=100, use_existing=True):
    """
    Extract noun-phrase concepts and embeddings for all normalised texts.
    """
    normalised_root = processed_text_path("normalised")
    if not normalised_root.exists():
        print(f"No normalised texts found at {normalised_root}")
        return

    for subdir in normalised_root.iterdir():
        if not subdir.is_dir():
            continue
        print(f"Processing concept embeddings: {subdir.name}")

        for file in subdir.glob("*_normalised.json"):
            generate_embeddings(file, top_n=top_n, use_existing=use_existing)

    print("Concept embeddings complete.")


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

        for file in subdir.glob("*.json"):
            output_file = out_subdir / f"{file.stem}_metrics.json"
            if use_existing and output_file.exists():
                print(f"Skipping {file.name} (exists)")
                continue

            try:
                text = json.load(file.open("r", encoding="utf-8")).get("text", "")
            except json.JSONDecodeError:
                text = ""
            result = metrics.build_metrics_for_text(text, file.name, nlp=nlp, window_size=window_size)
            output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Saved corpus metrics for {file.name}")


def _load_segmented_jsonl(path: Path) -> List[str]:
    sentences: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = entry.get("text") if isinstance(entry, dict) else None
            if text:
                sentences.append(str(text).strip())
    return [s for s in sentences if s]


def _load_text_for_windowing(category: Path, metrics_file: Path) -> Tuple[str, List[str]]:
    """
    Derive the cleaned-segmented text path from a metrics filename
    (assumes *_metrics.json convention).
    """
    segmented_root = processed_text_path("cleaned_segmented")
    base_name = metrics_file.stem.replace("_metrics", "")
    candidate = segmented_root / category.name / f"{base_name}.jsonl"
    if candidate.exists():
        sentences = _load_segmented_jsonl(candidate)
        return "\n".join(sentences), sentences
    cleaned_root = processed_text_path("cleaned")
    fallback = cleaned_root / category.name / f"{metrics_file.stem.replace('_metrics', '')}.json"
    if fallback.exists():
        try:
            data = json.load(fallback.open("r", encoding="utf-8"))
            text = data.get("text", "")
        except json.JSONDecodeError:
            text = ""
        return text, []
    return "", []


def run_windowed_metrics(window_size=DEFAULT_WINDOW_SIZE, mattr_window_size=50, use_existing=True):
    """
    Compute window-level metrics by combining syntax, lexico-semantic, discourse
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
            text_content, segmented_sentences = _load_text_for_windowing(subdir, file)
            if data.get("text"):
                text_content = data.get("text")
                segmented_sentences = []

            lex_analyzer = LexicoSemanticsAnalyzer(nlp)
            if segmented_sentences:
                docs = list(nlp.pipe(segmented_sentences))
                doc = Doc.from_docs(docs)
                num_sentences = len(segmented_sentences)
            else:
                doc = nlp(text_content or "")
                num_sentences = len(list(doc.sents))
                doc = nlp(text_content or "")  # reset iterator after counting
            syntax_metrics = syntax_analyzer.analyze_document(doc, window_size=window_size)
            lex_metrics = lex_analyzer.analyze_document(
                doc,
                window_size=window_size,
                mattr_window_size=mattr_window_size,
            )
            if segmented_sentences:
                discourse_sent, discourse_windows = discourse_analyzer.compute_sentence_metrics(
                    doc, window_size=window_size
                )
                discourse_metrics = {
                    "meta": {"window_size": window_size, "num_sentences": num_sentences},
                    "sentences": discourse_sent,
                    "windows": discourse_windows,
                }
            else:
                discourse_metrics = discourse_analyzer.analyze_text(
                    text_content or "", window_size=window_size
                )

            info_content_metrics = data.get("windows", [])

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
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(f"Saved windowed metrics for {file.name}")
    print("All done.")


def run_preprocessing(process_unknown=True):
    """
    Run PDF preprocessing to produce cleaned/normalised corpora.
    """
    preprocess_all_pdfs(process_unknown=process_unknown)


def run_all_metrics(window_size=DEFAULT_WINDOW_SIZE, mattr_window_size=50, use_existing=True, process_unknown=True):
    """
    Full orchestrator: preprocessing, concept embeddings, topics, corpus metrics, then windowed metrics.
    """
    run_preprocessing(process_unknown=process_unknown)
    run_concept_embeddings(use_existing=use_existing)
    run_topic_modelling(
        use_existing=use_existing,
        base_window_size=window_size,
    )
    run_corpus_metrics(window_size=window_size, use_existing=use_existing)
    run_windowed_metrics(window_size=window_size, mattr_window_size=mattr_window_size, use_existing=use_existing)


if __name__ == "__main__":
    run_all_metrics()
