"""
Orchestrator for running all b- and c-layer metrics and saving outputs.

Flow:
1) `run_preprocessing` converts raw PDFs to cleaned/normalised text and segmented JSONL under data/texts/processed/.
2) `run_concept_embeddings` reads normalised texts from `data/texts/processed/normalised_texts/<genre>/<author>/*_normalised.json`
   and writes phrases + embeddings to `data/analytics/embeddings/<genre>/<author>/<name>/`.
3) `run_topic_modelling` reads normalised-segmented texts from
   `data/texts/processed/normalised_segmented_texts/<genre>/<author>/*_normalised_segmented.jsonl`
   and writes topics JSON to `data/analytics/topic_modelling/<genre>/<author>/<name>/<name>_clustered_topics.json`.
4) `run_corpus_metrics` reads cleaned texts from `data/texts/processed/cleaned_texts/<genre>/<author>/*.json`
   and writes corpus metrics JSON to `data/analytics/corpus_analytics/<genre>/<author>/<name>/<name>_corpus_metrics.json`
   plus per-text frequencies at `data/analytics/corpus_analytics/<genre>/<author>/<name>/<name>_corpus_frequencies.json`.
   Each file matches the c0 meta/sentences/windows schema (log-probs, surprisal, etc.).
5) `run_windowed_metrics` reads those corpus JSONs, recomputes sentence-level spaCy docs
   from cleaned-segmented texts,
   and writes combined window metrics to `data/analytics/window_metrics/<genre>/<author>/<name>/<name>_window_metrics.json`
   with nested blocks: meta, syntax (meta/sentences/windows + heavy), lexico_semantics (same shape),
   discourse (same shape), log_prob (c0 windows)
"""

import json
from pathlib import Path
from typing import List, Tuple

from spacy.tokens import Doc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .x_configs import (
    DEFAULT_PIPELINE_TOPIC_WINDOW_STRIDE_MULTIPLE,
    DEFAULT_TOPIC_WINDOW_MULTIPLE,
    DEFAULT_WINDOW_SIZE,
    GENRES,
    MODEL_CONFIGS,
    WEB_CONFIGS,
    load_spacy_model,
)
from .a_preprocessing_cleaning import (
    TextPreprocessor,
    preprocess_all_pdfs,
    preprocess_web_story,
)
from .b1_concept_embeddings import ConceptExtractor, generate_embeddings
from .b3_log_prob_metrics import WholeTextMetrics
from .b2_topic_modeling import run_topic_modelling
from .c1_syntactics import SyntaxAnalyzer
from .c2_lexico_semantics import LexicoSemanticsAnalyzer
from .c3_discourse import DiscourseAnalyzer
from .d2_dashboard import run_dashboard
from .z_utils import analytics_path, iter_dirs, text_path

def run_concept_embeddings(top_n=100, use_existing=True, authors=None, encoder=None):
    """
    Extract noun-phrase concepts and embeddings for all normalised texts.
    """
    normalised_root = text_path("processed", "normalised_texts")
    if not normalised_root.exists():
        tqdm.write(f"Concept embeddings: no normalised texts found at {normalised_root}")
        return

    extractor = ConceptExtractor(encoder=encoder)
    categories = list(iter_dirs(normalised_root, genres=GENRES, authors=authors, depth=2))
    for category_key, subdir in tqdm(categories, desc="Concept embeddings", ascii=True):
        genre, author = category_key.split("/", 1)
        files = sorted(subdir.glob("*_normalised.json"))
        if not files:
            continue
        for file in tqdm(files, desc=f"Concept embeddings: {genre}/{author}", leave=False, ascii=True):
            generate_embeddings(
                file,
                top_n=top_n,
                use_existing=use_existing,
                extractor=extractor,
                quiet=True,
            )

    tqdm.write("Concept embeddings complete.")


def run_corpus_metrics(use_existing=True, authors=None):
    """
    Compute and save corpus-level log-prob/surprisal metrics for all cleaned texts.
    """
    window_size = DEFAULT_WINDOW_SIZE
    metrics = WholeTextMetrics()
    nlp = load_spacy_model()

    cleaned_root = text_path("processed", "cleaned_texts")
    segmented_root = text_path("processed", "cleaned_segmented_texts")
    output_root = analytics_path("corpus")
    output_root.mkdir(parents=True, exist_ok=True)

    categories = list(iter_dirs(cleaned_root, genres=GENRES, authors=authors, depth=2))
    processed = 0
    skipped = 0
    for category_key, subdir in tqdm(categories, desc="Corpus metrics", ascii=True):
        genre, author = category_key.split("/", 1)

        out_subdir = output_root / genre / author
        out_subdir.mkdir(parents=True, exist_ok=True)

        cleaned_files = sorted(subdir.glob("*.json"))

        file_texts = []
        for file in cleaned_files:
            try:
                text = json.load(file.open("r", encoding="utf-8")).get("text", "")
            except json.JSONDecodeError:
                text = ""
            file_texts.append((file, text))

        for file, text in tqdm(
            file_texts,
            desc=f"Corpus metrics: {category_key}",
            leave=False,
            ascii=True,
        ):
            base_name = file.stem.replace("_cleaned", "")
            text_dir = out_subdir / base_name
            text_dir.mkdir(parents=True, exist_ok=True)

            freq_path = text_dir / f"{base_name}_corpus_frequencies.json"
            if use_existing and freq_path.exists():
                try:
                    freqs = json.load(freq_path.open("r", encoding="utf-8"))
                except json.JSONDecodeError:
                    freqs = {}
            else:
                freqs = metrics.compute_corpus_frequencies([text], nlp=nlp)
                freq_path.write_text(json.dumps(freqs, indent=2), encoding="utf-8")

            output_file = text_dir / f"{base_name}_corpus_metrics.json"
            if use_existing and output_file.exists():
                skipped += 1
                continue

            segmented_path = segmented_root / genre / author / f"{base_name}_cleaned_segmented.jsonl"
            if segmented_path.exists():
                segmented_sentences = _load_segmented_jsonl(segmented_path)
                text = "\n".join(segmented_sentences)
                spans = []
                cursor = 0
                for sent in segmented_sentences:
                    end = cursor + len(sent)
                    spans.append((cursor, end))
                    cursor = end + 1
            else:
                spans = None

            result = metrics.build_metrics_for_text(
                text,
                file.name,
                nlp=nlp,
                window_size=window_size,
                sentence_spans=spans,
            )
            output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
            processed += 1
    tqdm.write(f"Corpus metrics complete: {processed} processed, {skipped} skipped.")


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


def _load_text_for_windowing(category_key: str, metrics_file: Path) -> Tuple[str, List[str]]:
    """
    Derive the cleaned-segmented text path from a metrics filename
    (assumes *_corpus_metrics.json convention).
    Raises FileNotFoundError if the segmented file is absent.
    """
    segmented_root = text_path("processed", "cleaned_segmented_texts", category_key)
    base_name = metrics_file.stem.replace("_corpus_metrics", "")
    candidate = segmented_root / f"{base_name}_cleaned_segmented.jsonl"
    if candidate.exists():
        sentences = _load_segmented_jsonl(candidate)
        return "\n".join(sentences), sentences
    raise FileNotFoundError(
        f"Segmented text not found for {metrics_file.name}: expected {candidate}. "
        "Generate cleaned segmented texts before running window metrics."
    )


def _load_corpus_frequencies(text_dir: Path, base_name: str) -> dict:
    freq_path = text_dir / f"{base_name}_corpus_frequencies.json"
    if not freq_path.exists():
        return {}
    try:
        data = json.load(freq_path.open("r", encoding="utf-8"))
        if isinstance(data, dict) and "word_frequencies" in data:
            return data.get("word_frequencies") or {}
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def run_windowed_metrics(mattr_window_size=50, use_existing=True, authors=None):
    """
    Compute window-level metrics by combining syntax, lexico-semantic, discourse
    Requires corpus outputs from run_corpus_metrics.
    """
    window_size = DEFAULT_WINDOW_SIZE
    corpus_root = analytics_path("corpus")
    output_root = analytics_path("window")
    output_root.mkdir(parents=True, exist_ok=True)

    nlp = load_spacy_model()
    syntax_analyzer = SyntaxAnalyzer(nlp)
    discourse_analyzer = DiscourseAnalyzer(nlp)

    categories = list(iter_dirs(corpus_root, genres=GENRES, authors=authors, depth=2))
    processed = 0
    skipped = 0
    for category_key, author_dir in tqdm(categories, desc="Window metrics", ascii=True):
        genre, author = category_key.split("/", 1)
        out_category_dir = output_root / genre / author
        out_category_dir.mkdir(parents=True, exist_ok=True)

        metric_files = []
        for text_dir in author_dir.iterdir():
            if not text_dir.is_dir():
                continue
            metric_files.extend(sorted(text_dir.glob("*_corpus_metrics.json")))
        for file in tqdm(
            metric_files,
            desc=f"Window metrics: {category_key}",
            leave=False,
            ascii=True,
        ):
            text_dir = file.parent
            base_name = file.stem.replace("_corpus_metrics", "")
            corpus_freqs = _load_corpus_frequencies(text_dir, base_name)
            global_avg_freq = (sum(corpus_freqs.values()) / len(corpus_freqs)) if corpus_freqs else None
            lex_analyzer = LexicoSemanticsAnalyzer(nlp, corpus_freqs=corpus_freqs)

            output_text_dir = out_category_dir / base_name
            output_text_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_text_dir / f"{base_name}_window_metrics.json"
            if use_existing and output_file.exists():
                skipped += 1
                continue

            data = json.load(file.open("r", encoding="utf-8"))
            meta_block = data.get("meta", {}) if isinstance(data, dict) else {}
            _, segmented_sentences = _load_text_for_windowing(category_key, file)
            if not segmented_sentences:
                raise ValueError(
                    f"No segmented sentences found for {file.name}; expected cleaned segmented JSONL."
                )

            # Build a single Doc while preserving provided sentence boundaries.
            tokenized_docs = [nlp.make_doc(text) for text in segmented_sentences]
            for sent_doc in tokenized_docs:
                for i, token in enumerate(sent_doc):
                    token.is_sent_start = i == 0

            doc = Doc.from_docs(tokenized_docs)
            for name, proc in nlp.pipeline:
                if name == "senter":  # keep our manual sentence boundaries
                    continue
                doc = proc(doc)
            num_sentences = len(segmented_sentences)
            doc_sentence_count = len(list(doc.sents))
            if doc_sentence_count != num_sentences:
                raise ValueError(
                    f"Sentence count mismatch after spaCy pipeline for {file.name}: "
                    f"segmented={num_sentences}, parsed={doc_sentence_count}. "
                    "Check sentence boundary drift or adjust pipeline components."
                )

            syntax_metrics = syntax_analyzer.analyze_document(doc, window_size=window_size)
            lex_metrics = lex_analyzer.analyze_document(
                doc,
                window_size=window_size,
                mattr_window_size=mattr_window_size,
                global_avg_freq=global_avg_freq,
            )
            discourse_sent, discourse_windows = discourse_analyzer.compute_sentence_metrics(
                doc, window_size=window_size
            )
            discourse_metrics = {
                "meta": {"window_size": window_size, "num_sentences": num_sentences},
                "sentences": discourse_sent,
                "windows": discourse_windows,
            }

            log_prob_sentences = data.get("sentences", [])
            log_prob_windows = data.get("windows", [])
            log_prob_meta = {
                "filename": meta_block.get("filename", file.name),
                "model": meta_block.get("model", ""),
                "window_size": meta_block.get("window_size", window_size),
                "num_sentences": len(log_prob_sentences) if log_prob_sentences else num_sentences,
                "avg_log_prob": meta_block.get("avg_log_prob"),
            }

            # Keep full window metrics; dashboard-level filtering happens downstream.
            syntax_windows = syntax_metrics.get("windows", [])
            lex_windows = lex_metrics.get("windows", [])
            discourse_windows = discourse_metrics.get("windows", [])

            result = {
                "meta": {
                    "filename": meta_block.get("filename", file.name),
                    "model": meta_block.get("model", ""),
                    "window_size": window_size,
                    "num_sentences": num_sentences,
                },
                "syntax": {
                    "meta": syntax_metrics.get("meta", {}),
                    "sentences": syntax_metrics.get("sentences", []),
                    "windows": syntax_windows,
                },
                "lexico_semantics": {
                    "meta": lex_metrics.get("meta", {}),
                    "sentences": lex_metrics.get("sentences", []),
                    "windows": lex_windows,
                },
                "log_prob": {
                    "meta": log_prob_meta,
                    "sentences": log_prob_sentences,
                    "windows": log_prob_windows,
                },
                "discourse": {
                    "meta": discourse_metrics.get("meta", {}),
                    "sentences": discourse_metrics.get("sentences", []),
                    "windows": discourse_windows,
                },
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            processed += 1
    tqdm.write(f"Window metrics complete: {processed} processed, {skipped} skipped.")


def run_preprocessing(process_unknown=True, use_existing=True, authors=None):
    """
    Run PDF preprocessing to produce cleaned/normalised corpora.
    """
    preproc = TextPreprocessor()
    for story_key, config in WEB_CONFIGS.items():
        if authors and config.get("author") not in authors:
            continue
        preprocess_web_story(
            story_key,
            preproc,
            config,
            use_existing=use_existing,
        )
    preprocess_all_pdfs(
        process_unknown=process_unknown,
        use_existing=use_existing,
        authors=authors,
    )


def run_all_metrics(
    mattr_window_size=50,
    use_existing=True,
    process_unknown=True,
    authors=None,
):
    """
    Full orchestrator: preprocessing, concept embeddings, topics, corpus metrics, then windowed metrics.
    """
    window_size = DEFAULT_WINDOW_SIZE
    tqdm.write("Stage 1/6: preprocessing")
    run_preprocessing(
        process_unknown=process_unknown,
        use_existing=use_existing,
        authors=authors,
    )
    tqdm.write("Stage 2/6: concept embeddings")
    shared_encoder = SentenceTransformer(MODEL_CONFIGS["sentence_embedding"])
    run_concept_embeddings(
        use_existing=use_existing,
        encoder=shared_encoder,
        authors=authors,
    )
    tqdm.write("Stage 3/6: topic modelling")
    run_topic_modelling(
        use_existing=use_existing,
        base_window_size=window_size,
        window_multiple=DEFAULT_TOPIC_WINDOW_MULTIPLE,
        window_stride=window_size * DEFAULT_PIPELINE_TOPIC_WINDOW_STRIDE_MULTIPLE,
        encoder=shared_encoder,
        authors=authors,
    )
    tqdm.write("Stage 4/6: corpus metrics")
    run_corpus_metrics(use_existing=use_existing, authors=authors)
    tqdm.write("Stage 5/6: window metrics")
    run_windowed_metrics(
        mattr_window_size=mattr_window_size,
        use_existing=use_existing,
        authors=authors,
    )
    tqdm.write("Stage 6/6: dashboard correlations")
    run_dashboard(use_existing=use_existing, authors=authors)
    tqdm.write("All stages complete.")

if __name__ == "__main__":
    run_all_metrics()
