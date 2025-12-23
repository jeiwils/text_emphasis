import json
import spacy
from .z_utils import processed_text_path
from .c1_syntactics import SyntaxAnalyzer
from .c2_lexico_semantics_TODO_DEPENDS_ON_CORPUS import LexicoSemanticsAnalyzer

def run_windowed_metrics(window_size=3, use_existing=True):
    """
    Computes sentence/window-level metrics for all texts
    using precomputed corpus-level metrics.
    
    Reads from: processed_text_paths('corpus')
    Saves to:  processed_text_paths('window')
    """
    corpus_root = processed_text_path("corpus")
    output_root = processed_text_path("window")
    output_root.mkdir(parents=True, exist_ok=True)
    nlp = spacy.load("en_core_web_sm")
    syntax_analyzer = SyntaxAnalyzer(nlp)

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

            # Load precomputed corpus metrics
            data = json.load(file.open("r", encoding="utf-8"))
            text_content = data.get("text")  # make sure you saved raw text or chunks
            if not text_content:
                # Fallback: stitch chunk text if raw text was not stored
                chunks = data.get("chunks", [])
                text_content = " ".join(chunk.get("text", "") for chunk in chunks if isinstance(chunk, dict))

            corpus_word_freqs = data.get("word_frequencies") or {w: f for w, f in data.get("top_words", [])}

            # Initialize analyzers
            lex_analyzer = LexicoSemanticsAnalyzer(nlp, corpus_freqs=corpus_word_freqs)
            doc = nlp(text_content or "")

            # ------------------------
            # Syntax metrics
            # ------------------------
            clause_metrics = syntax_analyzer.compute_clause_metrics(doc, window_size=window_size)
            clause_embed_metrics = syntax_analyzer.compute_clause_embedding_depth(doc, window_size=window_size)
            dep_complexity_metrics = syntax_analyzer.compute_dependency_complexity(doc, window_size=window_size)

            # ------------------------
            # Lexico-semantic metrics
            # ------------------------
            avg_word_freq_metrics = lex_analyzer.compute_avg_word_frequency(doc, window_size=window_size)
            lexical_information_content = lex_analyzer.analyze_information_content(
                doc, word_frequencies=corpus_word_freqs, window_size=window_size
            )
            
            # Use token log-probs if available
            log_probs_list = []
            for chunk in data.get("chunks", []):
                log_probs_list.append(chunk.get("log_probs", []))
            info_content_metrics = lex_analyzer.compute_information_content(log_probs_list, window_size=window_size)

            semantic_structures = lex_analyzer.extract_semantic_structures(doc, window_size=window_size)

            # Combine into result
            result = {
                "filename": data["filename"],
                "model": data.get("model", ""),
                "clause_metrics": clause_metrics,
                "clause_embedding_metrics": clause_embed_metrics,
                "dependency_complexity_metrics": dep_complexity_metrics,
                "avg_word_freq_metrics": avg_word_freq_metrics,
                "lexical_information_content": lexical_information_content,
                "information_content_metrics": info_content_metrics,
                "semantic_structures": semantic_structures
            }

            # Save
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(f"✅ Saved windowed metrics for {file.name}")

    print("🎉 All done.")
