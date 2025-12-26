import json
import re
import statistics

from x_configs import load_spacy_model
from .z_utils import processed_text_path, topic_modelling_path
from .c1_syntactics import SyntaxAnalyzer
from .c2_lexico_semantics import LexicoSemanticsAnalyzer





def load_topics_json(corpus_file):
    candidate_paths = [
        corpus_file.with_name(f"{corpus_file.stem}_topics.json"),
        corpus_file.with_name(f"{corpus_file.stem}.topics.json"),
        corpus_file.with_name(f"{corpus_file.stem}-topics.json"),
    ]
    candidate_paths.extend(sorted(corpus_file.parent.glob(f"{corpus_file.stem}*topic*.json")))

    seen = set()
    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def collect_topic_mentions(topics_data):
    if not topics_data:
        return []

    if isinstance(topics_data, dict):
        topics = (
            topics_data.get("topics")
            or topics_data.get("topic_results")
            or topics_data.get("results")
            or []
        )
    elif isinstance(topics_data, list):
        topics = topics_data
    else:
        topics = []

    mentions = []
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        topic_id = topic.get("topic_id", topic.get("id"))
        for mention in topic.get("mentions", []):
            if not isinstance(mention, dict):
                continue
            sentence_index = mention.get("sentence_index")
            if sentence_index is None:
                continue
            mentions.append(
                {
                    "topic_id": topic_id,
                    "sentence_index": sentence_index,
                }
            )
    return mentions


def build_topic_window_metrics(topic_mentions, window_entries):
    metrics = []
    for window in window_entries:
        start_sentence = window.get("start_sentence", 0)
        end_sentence = window.get("end_sentence", 0)
        window_mentions = [
            mention
            for mention in topic_mentions
            if start_sentence <= mention["sentence_index"] <= end_sentence
        ]
        topic_counts = {}
        for mention in window_mentions:
            topic_id = mention["topic_id"]
            if topic_id is None:
                continue
            topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
        sorted_topics = sorted(
            topic_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        top_topic_ids = [topic_id for topic_id, _ in sorted_topics]
        metrics.append(
            {
                "start_sentence": start_sentence,
                "end_sentence": end_sentence,
                "topic_mention_count": len(window_mentions),
                "unique_topic_count": len(topic_counts),
                "top_topic_ids": top_topic_ids,
            }
        )
    return metrics


def _tokenize_words(text: str, lowercase: bool = True):
    """Lightweight tokenizer for lexical diversity (MATTR)."""
    tokens = re.findall(r"[A-Za-z0-9']+", text)
    if lowercase:
        tokens = [t.lower() for t in tokens]
    return tokens


def _moving_average_type_token_ratio(tokens, window_size: int = 50) -> float:
    """Compute Moving Average Type-Token Ratio (MATTR) over a sliding window."""
    tokens = [t for t in tokens if t]
    total_tokens = len(tokens)

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if total_tokens == 0:
        return 0.0
    if total_tokens < window_size:
        return round(len(set(tokens)) / total_tokens, 3)

    ttr_values = []
    for i in range(total_tokens - window_size + 1):
        window = tokens[i : i + window_size]
        ttr_values.append(len(set(window)) / window_size)

    return round(statistics.mean(ttr_values), 3)


def compute_mattr_metrics(text: str, window_size: int = 50, lowercase: bool = True):
    """Compute MATTR over the whole text for inclusion in window metrics."""
    words = _tokenize_words(text, lowercase=lowercase)
    mattr = _moving_average_type_token_ratio(words, window_size=window_size)
    return {
        "mattr_score": mattr,
        "window_size": min(window_size, len(words)),
        "total_tokens": len(words),
    }



def run_windowed_metrics(window_size=3, mattr_window_size=50, use_existing=True):
    """
    Computes sentence/window-level metrics for all texts
    using precomputed corpus-level metrics.
    
    Reads from: processed_text_paths('corpus')
    Saves to:  processed_text_paths('window')
    """
    corpus_root = processed_text_path("corpus")
    output_root = processed_text_path("window")
    output_root.mkdir(parents=True, exist_ok=True)
    nlp = load_spacy_model()
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
            mattr_metrics = compute_mattr_metrics(text_content or "", window_size=mattr_window_size)

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

            topics_data = load_topics_json(file)
            topic_mentions = collect_topic_mentions(topics_data)
            topic_metrics = build_topic_window_metrics(topic_mentions, clause_metrics)

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
                "semantic_structures": semantic_structures,
                "topic_metrics": topic_metrics,
                "lexical_diversity": mattr_metrics,
            }

            # Save
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(f"✅ Saved windowed metrics for {file.name}")

    print("🎉 All done.")
