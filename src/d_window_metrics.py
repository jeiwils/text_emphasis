import json
from pathlib import Path
from typing import List, Optional, Tuple
from spacy.tokens import Doc

from x_configs import DEFAULT_WINDOW_SIZE, GENRES, load_spacy_model
from a_preprocessing_cleaning import preprocess_all_pdfs
from b_concept_embeddings import ConceptExtractor, generate_embeddings
from c0_log_prob_metrics import WholeTextMetrics
from c4_topic_modeling import run_topic_modelling
from c1_syntactics import SyntaxAnalyzer
from c2_lexico_semantics import LexicoSemanticsAnalyzer
from c3_discourse import DiscourseAnalyzer
from w_dashboard import run_dashboard
from z_utils import analytics_path, iter_genre_author_dirs, text_path



"""


Orchestrator for running all b- and c-layer metrics and saving outputs.

Flow:
1) `run_preprocessing` converts raw PDFs to cleaned/normalised text and segmented JSONL under data/texts/processed/.
2) `run_concept_embeddings` reads normalised texts from `data/texts/processed/normalised_texts/<category>/*_normalised.json`
   and writes phrases + embeddings to `data/embeddings/concept_embeddings/<category>/<name>/`.
3) `run_topic_modelling` reads normalised-segmented texts from
   `data/texts/processed/normalised_segmented_texts/<category>/*_normalised_segmented.jsonl`
   and writes topics JSON to `data/analytics/topic_modelling/<category>/<name>_topics.json`.
4) `run_corpus_metrics` reads cleaned texts from `data/texts/processed/cleaned_texts/<category>/*.json`
   and writes corpus metrics JSON to `data/analytics/corpus_analytics/<category>/<name>/<name>_corpus_metrics.json`
   plus per-text frequencies at `data/analytics/corpus_analytics/<category>/<name>/<name>_corpus_frequencies.json`.
   Each file matches the c0 meta/sentences/windows schema (log-probs, surprisal, etc.).
5) `run_windowed_metrics` reads those corpus JSONs, recomputes sentence-level spaCy docs
   from cleaned-segmented texts,
   and writes combined window metrics to `data/analytics/window_metrics/<category>/<name>/<name>_window_metrics.json`
   with nested blocks: meta, syntax (meta/sentences/windows + heavy), lexico_semantics (same shape),
   discourse (same shape), information_content_metrics (c0 windows)
"""

_DASHBOARD_METRICS = {
    "discourse": {
        "explicit_connectives_per_token",
        "connective_counts_per_token",
        "entity_overlap_ratio",
        "content_overlap_ratio",
        "pronoun_ratio",
    },
    "lexico_semantics": {
        "lexical_density",
        "content_function_ratio",
        "num_clauses_per_token",
        "num_agents_per_token",
        "num_patients_per_token",
        "role_count_per_token",
        "role_counts_per_token",
    },
    "syntax": {
        "clause_counts_per_token",
        "clause_ratios",
        "avg_dependents_per_head",
        "avg_mean_dependency_distance",
        "mean_depth",
        "median_depth",
        "max_depth",
        "depth_skew",
        "avg_tokens_per_sentence",
    },
    "log_prob": {
        "token_weighted_mean_surprisal",
        "token_weighted_surprisal_variance",
    },
}


def _prune_window_metrics(
    metrics: dict,
    keep_keys: set,
    *,
    nested_keys: Optional[set] = None,
) -> dict:
    if not isinstance(metrics, dict):
        return {}
    pruned = {}
    for key in ("start_sentence", "end_sentence", "token_count"):
        if key in metrics:
            pruned[key] = metrics.get(key)
    for key in keep_keys:
        value = metrics.get(key)
        if key in (nested_keys or set()) and isinstance(value, dict):
            pruned[key] = value
        elif value is not None:
            pruned[key] = value
    return pruned


def run_concept_embeddings(top_n=100, use_existing=True, authors=None):
    """
    Extract noun-phrase concepts and embeddings for all normalised texts.
    """
    normalised_root = text_path("processed", "normalised_texts")
    if not normalised_root.exists():
        print(f"No normalised texts found at {normalised_root}")
        return

    extractor = ConceptExtractor()
    for genre, author, subdir in iter_genre_author_dirs(normalised_root, GENRES, authors):
        print(f"Processing concept embeddings: {genre}/{author}")
        for file in subdir.glob("*_normalised.json"):
            generate_embeddings(
                file,
                top_n=top_n,
                use_existing=use_existing,
                extractor=extractor,
            )

    print("Concept embeddings complete.")


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

    for genre, author, subdir in iter_genre_author_dirs(cleaned_root, GENRES, authors):
        category_key = f"{genre}/{author}"
        print(f"Processing category: {category_key}")

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

        for file, text in file_texts:
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
                freqs = metrics.compute_corpus_frequencies([text])
                freq_path.write_text(json.dumps(freqs, indent=2), encoding="utf-8")

            output_file = text_dir / f"{base_name}_corpus_metrics.json"
            if use_existing and output_file.exists():
                print(f"Skipping {output_file.name} (exists)")
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

    for genre, author, author_dir in iter_genre_author_dirs(corpus_root, GENRES, authors):
        category_key = f"{genre}/{author}"
        print(f"Processing category: {category_key}")
        out_category_dir = output_root / genre / author
        out_category_dir.mkdir(parents=True, exist_ok=True)

        for text_dir in author_dir.iterdir():
            if not text_dir.is_dir():
                continue

            for file in text_dir.glob("*_corpus_metrics.json"):
                base_name = file.stem.replace("_corpus_metrics", "")
                corpus_freqs = _load_corpus_frequencies(text_dir, base_name)
                global_avg_freq = (sum(corpus_freqs.values()) / len(corpus_freqs)) if corpus_freqs else None
                lex_analyzer = LexicoSemanticsAnalyzer(nlp, corpus_freqs=corpus_freqs)

                output_text_dir = out_category_dir / base_name
                output_text_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_text_dir / f"{base_name}_window_metrics.json"
                if use_existing and output_file.exists():
                    print(f"Skipping {output_file.name} (exists)")
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

                # Prune window metrics to the dashboard-relevant fields only.
                syntax_windows = [
                    _prune_window_metrics(
                        window,
                        _DASHBOARD_METRICS["syntax"],
                        nested_keys={"clause_counts_per_token", "clause_ratios", "avg_dependents_per_head"},
                    )
                    for window in syntax_metrics.get("windows", [])
                ]
                lex_windows = [
                    _prune_window_metrics(
                        window,
                        _DASHBOARD_METRICS["lexico_semantics"],
                        nested_keys={"role_counts_per_token"},
                    )
                    for window in lex_metrics.get("windows", [])
                ]
                discourse_windows = [
                    _prune_window_metrics(
                        window,
                        _DASHBOARD_METRICS["discourse"],
                        nested_keys={"connective_counts_per_token"},
                    )
                    for window in discourse_metrics.get("windows", [])
                ]
                log_prob_windows = [
                    _prune_window_metrics(window, _DASHBOARD_METRICS["log_prob"])
                    for window in log_prob_windows
                ]

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
                    "information_content_metrics": log_prob_windows,
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

                print(f"Saved windowed metrics for {file.name}")
    print("All done.")


def run_preprocessing(process_unknown=True, use_existing=True):
    """
    Run PDF preprocessing to produce cleaned/normalised corpora.
    """
    preprocess_all_pdfs(process_unknown=process_unknown, use_existing=use_existing)


def run_all_metrics(mattr_window_size=50, use_existing=True, process_unknown=True):
    """
    Full orchestrator: preprocessing, concept embeddings, topics, corpus metrics, then windowed metrics.
    """
    window_size = DEFAULT_WINDOW_SIZE
    run_preprocessing(process_unknown=process_unknown, use_existing=use_existing)
    run_concept_embeddings(use_existing=use_existing)
    run_topic_modelling(
        use_existing=use_existing,
        base_window_size=window_size,
        window_multiple=5,
        window_stride=window_size,
    )
    run_corpus_metrics(use_existing=use_existing)
    run_windowed_metrics(mattr_window_size=mattr_window_size, use_existing=use_existing)
    run_dashboard(use_existing=use_existing)

if __name__ == "__main__":
    run_all_metrics()
