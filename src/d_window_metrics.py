import json
from pathlib import Path
from typing import List, Tuple
from spacy.tokens import Doc

from x_configs import DEFAULT_WINDOW_SIZE, load_spacy_model
from a_preprocessing_cleaning import preprocess_all_pdfs
from b1_concept_embeddings import generate_embeddings
from c0_log_prob_metrics import WholeTextMetrics
from b2_topic_modeling import run_topic_modelling
from c1_syntactics import SyntaxAnalyzer
from c2_lexico_semantics import LexicoSemanticsAnalyzer
from c3_discourse import DiscourseAnalyzer
from z_utils import processed_text_path



"""


Orchestrator for running all b- and c-layer metrics and saving outputs.

Flow:
1) `run_preprocessing` converts raw PDFs to cleaned/normalised text and segmented JSONL under data/processed/.
2) `run_concept_embeddings` reads normalised texts from `data/processed/normalised_texts/<category>/*_normalised.json`
   and writes phrases + embeddings to `data/embeddings/concept_embeddings/<category>/<name>/`.
3) `run_topic_modelling` reads normalised-segmented texts from
   `data/processed/normalised_segmented_texts/<category>/*_normalised_segmented.jsonl`
   and writes topics JSON to `data/topic_modelling/<category>/<name>_topics.json`.
4) `run_corpus_metrics` reads cleaned texts from `data/processed/cleaned_texts/<category>/*.json`
   and writes corpus metrics JSON to `data/processed/corpus_analytics/<category>/<name>_corpus_metrics.json`
   plus per-text frequencies at `data/processed/corpus_analytics/<category>/<name>_corpus_frequencies.json`.
   Each file matches the c0 meta/sentences/windows schema (log-probs, surprisal, etc.).
5) `run_windowed_metrics` reads those corpus JSONs, recomputes sentence-level spaCy docs
   from cleaned-segmented texts,
   and writes combined window metrics to `data/processed/window_metrics/<category>/<name>_window_metrics.json`
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


def run_corpus_metrics(use_existing=True):
    """
    Compute and save corpus-level log-prob/surprisal metrics for all cleaned texts.
    """
    window_size = DEFAULT_WINDOW_SIZE
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
            freq_path = out_subdir / f"{base_name}_corpus_frequencies.json"
            if use_existing and freq_path.exists():
                try:
                    freqs = json.load(freq_path.open("r", encoding="utf-8"))
                except json.JSONDecodeError:
                    freqs = {}
            else:
                freqs = metrics.compute_corpus_frequencies([text])
                freq_path.write_text(json.dumps(freqs, indent=2), encoding="utf-8")

            output_file = out_subdir / f"{base_name}_corpus_metrics.json"
            if use_existing and output_file.exists():
                print(f"Skipping {output_file.name} (exists)")
                continue

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
    (assumes *_corpus_metrics.json convention).
    Raises FileNotFoundError if the segmented file is absent.
    """
    segmented_root = processed_text_path("cleaned_segmented")
    base_name = metrics_file.stem.replace("_corpus_metrics", "")
    candidate = segmented_root / category.name / f"{base_name}_cleaned_segmented.jsonl"
    if candidate.exists():
        sentences = _load_segmented_jsonl(candidate)
        return "\n".join(sentences), sentences
    raise FileNotFoundError(
        f"Segmented text not found for {metrics_file.name}: expected {candidate}. "
        "Generate cleaned segmented texts before running window metrics."
    )


def _load_corpus_frequencies(category_dir: Path, base_name: str) -> dict:
    freq_path = category_dir / f"{base_name}_corpus_frequencies.json"
    if not freq_path.exists():
        return {}
    try:
        return json.load(freq_path.open("r", encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def run_windowed_metrics(mattr_window_size=50, use_existing=True):
    """
    Compute window-level metrics by combining syntax, lexico-semantic, discourse
    Requires corpus outputs from run_corpus_metrics.
    """
    window_size = DEFAULT_WINDOW_SIZE
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
            if file.name.endswith("_corpus_frequencies.json"):
                continue
            base_name = file.stem.replace("_corpus_metrics", "")
            corpus_freqs = _load_corpus_frequencies(subdir, base_name)
            global_avg_freq = (sum(corpus_freqs.values()) / len(corpus_freqs)) if corpus_freqs else None
            lex_analyzer = LexicoSemanticsAnalyzer(nlp, corpus_freqs=corpus_freqs)

            output_file = out_subdir / f"{base_name}_window_metrics.json"
            if use_existing and output_file.exists():
                print(f"Skipping {output_file.name} (exists)")
                continue

            data = json.load(file.open("r", encoding="utf-8"))
            meta_block = data.get("meta", {}) if isinstance(data, dict) else {}
            text_content, segmented_sentences = _load_text_for_windowing(subdir, file)
            if not segmented_sentences:
                raise ValueError(
                    f"No segmented sentences found for {file.name}; expected cleaned segmented JSONL."
                )

            docs = list(nlp.pipe(segmented_sentences))
            doc = Doc.from_docs(docs)
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


def run_all_metrics(mattr_window_size=50, use_existing=True, process_unknown=True):
    """
    Full orchestrator: preprocessing, concept embeddings, topics, corpus metrics, then windowed metrics.
    """
    window_size = DEFAULT_WINDOW_SIZE
    run_preprocessing(process_unknown=process_unknown)
    run_concept_embeddings(use_existing=use_existing)
    run_topic_modelling(
        use_existing=use_existing,
        base_window_size=window_size,
        window_multiple=5,
        window_stride=window_size,
    )
    run_corpus_metrics(use_existing=use_existing)
    run_windowed_metrics(mattr_window_size=mattr_window_size, use_existing=use_existing)


if __name__ == "__main__":
    run_all_metrics()
