import json
import statistics
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import spacy

from .z_utils import aggregate_windows, processed_text_path


"""
Discourse-level metrics built on simple heuristics.

This module avoids model training and instead offers:
    - Explicit connective detection mapped to PDTB Level-1 senses.
    - Cohesion signals via entity/content overlap across adjacent sentences.
    - Shallow tense tracking to flag temporal shifts.

Outputs are sentence-level annotations, sliding-window aggregates,
and a short summary block that can be written to JSON.
"""


CONNECTIVE_LEXICON: Dict[str, Sequence[str]] = {
    # PDTB-inspired level-1 groupings
    "Temporal": [
        "before",
        "after",
        "when",
        "while",
        "once",
        "until",
        "meanwhile",
        "as soon as",
    ],
    "Contingency": [
        "because",
        "since",
        "therefore",
        "thus",
        "so",
        "hence",
        "consequently",
        "if",
        "unless",
    ],
    "Comparison": [
        "but",
        "however",
        "although",
        "though",
        "whereas",
        "instead",
        "yet",
        "nevertheless",
        "on the other hand",
    ],
    "Expansion": [
        "and",
        "also",
        "moreover",
        "furthermore",
        "besides",
        "in addition",
        "for example",
        "for instance",
        "indeed",
    ],
}


class DiscourseAnalyzer:
    def __init__(self, nlp=None):
        self.nlp = nlp or spacy.load("en_core_web_sm")
        self._lexicon_phrases = self._prepare_lexicon()

    def _prepare_lexicon(self) -> List[Tuple[str, List[str], str]]:
        phrases = []
        for category, markers in CONNECTIVE_LEXICON.items():
            for marker in markers:
                phrases.append((category, marker.split(), marker))
        # Sort longer markers first to avoid partial matches (e.g., "as soon as" before "as")
        return sorted(phrases, key=lambda x: len(x[1]), reverse=True)

    def _overlap(self, previous: Iterable[str], current: Iterable[str]) -> Tuple[int, float]:
        prev_set, curr_set = set(previous), set(current)
        if not prev_set or not curr_set:
            return 0, 0.0
        overlap = len(prev_set & curr_set)
        ratio = overlap / max(len(curr_set), 1)
        return overlap, round(ratio, 3)

    def _infer_tense(self, sent) -> Optional[str]:
        """Very small heuristic for tense: majority vote of verb POS tags and modals."""
        past_tags = {"VBD", "VBN"}
        present_tags = {"VBP", "VBZ", "VBG"}
        counts = Counter()

        tokens = list(sent)
        for token in tokens:
            if token.tag_ in past_tags:
                counts["past"] += 1
            elif token.tag_ in present_tags:
                counts["present"] += 1

        # Simple future proxy
        if any(t.lower_ in {"will", "shall", "gonna", "going"} for t in tokens):
            counts["future"] += 1

        if not counts:
            return None

        return counts.most_common(1)[0][0]

    def _find_connectives(self, sent) -> List[Dict[str, object]]:
        tokens = [t.text.lower() for t in sent if not t.is_space]
        matches: List[Dict[str, object]] = []

        for category, marker_tokens, marker_str in self._lexicon_phrases:
            span_len = len(marker_tokens)
            if span_len == 0 or len(tokens) < span_len:
                continue
            for i in range(len(tokens) - span_len + 1):
                if tokens[i : i + span_len] == marker_tokens:
                    matches.append(
                        {
                            "marker": marker_str,
                            "category": category,
                            "start": i,
                            "end": i + span_len - 1,
                        }
                    )
                    break  # avoid double-counting the same marker
        return matches

    def _empty_connective_counts(self) -> Dict[str, int]:
        return {category: 0 for category in CONNECTIVE_LEXICON}

    def _dominant_relation(self, counts: Dict[str, int]) -> Optional[str]:
        if not counts:
            return None
        category, value = max(counts.items(), key=lambda item: item[1])
        return category if value > 0 else None

    def compute_sentence_metrics(self, doc, window_size: Optional[int] = None):
        """
        Compute per-sentence discourse markers and cohesion signals.
        """
        sent_metrics: List[Dict[str, object]] = []
        annotations: List[Dict[str, object]] = []

        prev_entities: set = set()
        prev_content: set = set()
        prev_tense: Optional[str] = None

        for idx, sent in enumerate(doc.sents):
            tokens = [t for t in sent if not t.is_space]
            connectives = self._find_connectives(sent)
            connective_counts = self._empty_connective_counts()
            for c in connectives:
                connective_counts[c["category"]] += 1

            # Cohesion signals: noun overlap and content-word overlap
            noun_lemmas = {t.lemma_.lower() for t in sent if t.pos_ in {"NOUN", "PROPN"} and t.is_alpha}
            content_lemmas = {
                t.lemma_.lower()
                for t in sent
                if t.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"} and t.is_alpha
            }

            entity_overlap, entity_ratio = self._overlap(prev_entities, noun_lemmas)
            content_overlap, content_ratio = self._overlap(prev_content, content_lemmas)

            pronoun_count = sum(1 for t in sent if t.pos_ == "PRON")
            pronoun_ratio = round(pronoun_count / len(tokens), 3) if tokens else 0.0

            tense = self._infer_tense(sent)
            tense_shift = int(prev_tense is not None and tense is not None and tense != prev_tense)

            sent_metrics.append(
                {
                    "sentence_index": idx,
                    "num_tokens": len(tokens),
                    "explicit_connectives": len(connectives),
                    "connective_counts": connective_counts,
                    "entity_overlap": entity_overlap,
                    "entity_overlap_ratio": entity_ratio,
                    "content_overlap": content_overlap,
                    "content_overlap_ratio": content_ratio,
                    "pronoun_ratio": pronoun_ratio,
                    "tense_shift": tense_shift,
                }
            )

            annotations.append(
                {
                    "sentence_index": idx,
                    "text": sent.text.strip(),
                    "connectives": connectives,
                    "dominant_relation": self._dominant_relation(connective_counts),
                    "verb_tense": tense,
                }
            )

            prev_entities = noun_lemmas
            prev_content = content_lemmas
            prev_tense = tense

        windowed_metrics = aggregate_windows(sent_metrics, window_size) if window_size and window_size > 1 else []

        return sent_metrics, annotations, windowed_metrics

    def summarize(self, sentence_metrics: List[Dict[str, object]]) -> Dict[str, object]:
        if not sentence_metrics:
            return {
                "total_sentences": 0,
                "total_connectives": 0,
                "relation_totals": {},
                "avg_pronoun_ratio": 0.0,
                "avg_entity_overlap": 0.0,
                "avg_content_overlap": 0.0,
                "tense_switch_rate": 0.0,
            }

        relation_counter: Counter = Counter()
        for m in sentence_metrics:
            relation_counter.update(m["connective_counts"])

        total_sentences = len(sentence_metrics)
        avg_pronoun = statistics.mean(m["pronoun_ratio"] for m in sentence_metrics)
        avg_entity_overlap = statistics.mean(m["entity_overlap_ratio"] for m in sentence_metrics)
        avg_content_overlap = statistics.mean(m["content_overlap_ratio"] for m in sentence_metrics)

        tense_switches = sum(m["tense_shift"] for m in sentence_metrics)
        tense_switch_rate = tense_switches / max(total_sentences - 1, 1)

        return {
            "total_sentences": total_sentences,
            "total_connectives": sum(m["explicit_connectives"] for m in sentence_metrics),
            "relation_totals": dict(relation_counter),
            "avg_pronoun_ratio": round(avg_pronoun, 3),
            "avg_entity_overlap": round(avg_entity_overlap, 3),
            "avg_content_overlap": round(avg_content_overlap, 3),
            "tense_switch_rate": round(tense_switch_rate, 3),
        }

    def analyze_text(self, text: str, window_size: int = 3) -> Dict[str, object]:
        """
        Analyze a single text and return discourse metrics.

        Output shape (dict):
        {
          "sentence_metrics": [ ... ],    # One row per sentence with numeric metrics.
          "sentence_annotations": [ ... ],# One row per sentence with text + discourse labels.
          "window_metrics": [ ... ],      # One row per window with averaged numeric metrics.
          "summary": { ... },             # Corpus-level summary aggregated over all sentences.
          ...
        }

        Detailed fields:
        {
          "sentence_metrics": [
            {
              "sentence_index": int,  # 0-based index of the sentence in the document.
              "num_tokens": int,  # Non-space token count for the sentence.
              "explicit_connectives": int,  # Number of matched connective phrases.
              "connective_counts": {"Temporal": int, "Contingency": int, ...},  # Count by relation.
              "entity_overlap": int,  # Shared noun/proper-noun lemmas vs. previous sentence.
              "entity_overlap_ratio": float,  # Overlap ratio for noun/proper-noun lemmas.
              "content_overlap": int,  # Shared content-word lemmas vs. previous sentence.
              "content_overlap_ratio": float,  # Overlap ratio for content-word lemmas.
              "pronoun_ratio": float,  # Pronoun count divided by total tokens.
              "tense_shift": int  # 1 if tense changed vs. previous sentence, else 0.
            },
            ...
          ],
          "sentence_annotations": [
            {
              "sentence_index": int,  # 0-based index of the sentence in the document.
              "text": str,  # Raw sentence text.
              "connectives": [
                {
                  "marker": str,  # Matched connective phrase.
                  "category": str,  # PDTB-inspired relation category.
                  "start": int,  # Start token index within the sentence.
                  "end": int  # End token index within the sentence.
                },
                ...
              ],
              "dominant_relation": Optional[str],  # Relation with highest count, if any.
              "verb_tense": Optional[str]  # Heuristic tense label ("past"/"present"/"future").
            },
            ...
          ],
          "window_metrics": [
            {
              "...": "averaged numeric fields from sentence_metrics",
              "start_sentence": int,  # 0-based index of the first sentence in the window.
              "end_sentence": int  # 0-based index of the last sentence in the window.
            },
            ...
          ],
          "summary": {
            "total_sentences": int,  # Number of sentences in the document.
            "total_connectives": int,  # Total explicit connective matches.
            "relation_totals": {"Temporal": int, ...},  # Totals by relation category.
            "avg_pronoun_ratio": float,  # Mean pronoun ratio across sentences.
            "avg_entity_overlap": float,  # Mean noun/proper-noun overlap ratio.
            "avg_content_overlap": float,  # Mean content-word overlap ratio.
            "tense_switch_rate": float  # Tense shifts divided by sentence transitions.
          }
        }
        """
        doc = self.nlp(text)
        sentence_metrics, annotations, windowed_metrics = self.compute_sentence_metrics(doc, window_size=window_size)
        summary = self.summarize(sentence_metrics)

        return {
            "sentence_metrics": sentence_metrics,
            "sentence_annotations": annotations,
            "window_metrics": windowed_metrics,
            "summary": summary,
        }


def run_discourse_analysis(window_size: int = 3, use_existing: bool = True):
    """
    Run discourse analysis across cleaned texts and save JSON outputs alongside other window metrics.
    """
    cleaned_root = processed_text_path("cleaned")
    output_root = processed_text_path("window")
    analyzer = DiscourseAnalyzer()

    for subdir in cleaned_root.iterdir():
        if not subdir.is_dir():
            continue

        out_subdir = output_root / subdir.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        for file in subdir.glob("*.txt"):
            output_file = out_subdir / f"{file.stem}_discourse.json"
            if use_existing and output_file.exists():
                print(f"Skipping {file.name} (exists)")
                continue

            text = file.read_text(encoding="utf-8")
            result = analyzer.analyze_text(text, window_size=window_size)
            result["filename"] = file.name
            result["window_size"] = window_size

            output_file.write_text(
                json.dumps(result, indent=2),
                encoding="utf-8",
            )
            print(f"Saved discourse metrics for {file.name}")











def main():
    run_discourse_analysis()


if __name__ == "__main__":
    main()