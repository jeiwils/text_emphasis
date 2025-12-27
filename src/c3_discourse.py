import statistics
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from x_configs import DEFAULT_WINDOW_SIZE, load_spacy_model
from .z_utils import aggregate_windows

"""
Discourse-level metrics and cohesion across sentences (heuristic, no training).


Output shape (no IO):
{
  "meta": {"window_size": 3, "num_sentences": 80},

  "sentences": [
    {
      "sentence_index": 0, 
      "num_tokens": 12,
      "explicit_connectives": 1, 
      "connective_counts": {
        "Temporal": 1, 
        ...
        },
      "entity_overlap": 0, 
      "entity_overlap_ratio": 0.0,
      "content_overlap": 0, 
      "content_overlap_ratio": 0.0,
      "pronoun_ratio": 0.08, 
      "tense_shift": 0,
      "dominant_relation": "Temporal", 
      "verb_tense": "past"
    },
    ...
  ],

  "windows": [ # averaged over the window
    {
      "start_sentence": 0,
      "end_sentence": 2,
      "num_tokens": 11, 
      "explicit_connectives": 0.3, 
      "connective_counts": {
        "Temporal": 0.3, 
        ...
        },
      "entity_overlap": 0.3, 
      "entity_overlap_ratio": 0.1,
      "content_overlap": 0.6, 
      "content_overlap_ratio": 0.2,
      "pronoun_ratio": 0.05, 
      "tense_shift": 0.3
    },
    ...
  ]
}


"""


CONNECTIVE_LEXICON: Dict[str, Sequence[str]] = {
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
        self.nlp = nlp or load_spacy_model()
        self._lexicon_phrases = self._prepare_lexicon()

    def _prepare_lexicon(self) -> List[Tuple[str, List[str], str]]:
        phrases = []
        for category, markers in CONNECTIVE_LEXICON.items():
            for marker in markers:
                phrases.append((category, marker.split(), marker))
        return sorted(phrases, key=lambda x: len(x[1]), reverse=True)

    def _overlap(self, previous: Iterable[str], current: Iterable[str]) -> Tuple[int, float]:
        prev_set, curr_set = set(previous), set(current)
        if not prev_set or not curr_set:
            return 0, 0.0
        overlap = len(prev_set & curr_set)
        ratio = overlap / max(len(curr_set), 1)
        return overlap, round(ratio, 3)

    def _infer_tense(self, sent) -> Optional[str]:
        past_tags = {"VBD", "VBN"}
        present_tags = {"VBP", "VBZ", "VBG"}
        counts = Counter()

        tokens = list(sent)
        for token in tokens:
            if token.tag_ in past_tags:
                counts["past"] += 1
            elif token.tag_ in present_tags:
                counts["present"] += 1

        has_modal_future = any(t.lower_ in {"will", "shall", "gonna"} for t in tokens)
        has_going_to = any(
            tokens[i].lower_ == "going"
            and i + 2 < len(tokens)
            and tokens[i + 1].lower_ == "to"
            and tokens[i + 2].pos_ == "VERB"
            for i in range(len(tokens) - 2)
        )
        if has_modal_future or has_going_to:
            counts["future"] += 1

        if not counts:
            return None

        return counts.most_common(1)[0][0]

    def analyze_cohesion(self, doc, window_size: Optional[int] = None):
        prev_sent_content = None
        sent_metrics: List[Dict[str, object]] = []

        for sent in doc.sents:
            words = [t.lemma_.lower() for t in sent if t.is_alpha]
            overlap = None
            if prev_sent_content is not None:
                overlap = len(set(words) & set(prev_sent_content))

            sent_metrics.append(
                {
                    "cohesion_overlap": overlap,
                }
            )

            prev_sent_content = words

        if window_size and window_size > 1:
            return aggregate_windows(sent_metrics, window_size)

        return sent_metrics

    def find_connectives(self, sent) -> List[Dict[str, object]]:
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
        return matches

    def _empty_connective_counts(self) -> Dict[str, int]:
        return {category: 0 for category in CONNECTIVE_LEXICON}

    def _dominant_relation(self, counts: Dict[str, int]) -> Optional[str]:
        if not counts:
            return None
        category, value = max(counts.items(), key=lambda item: item[1])
        return category if value > 0 else None

    def compute_sentence_metrics(self, doc, window_size: Optional[int] = None):
        sent_metrics: List[Dict[str, object]] = []

        prev_entities: set = set()
        prev_content: set = set()
        prev_tense: Optional[str] = None

        for idx, sent in enumerate(doc.sents):
            tokens = [t for t in sent if not t.is_space]
            connectives = self.find_connectives(sent)
            connective_counts = self._empty_connective_counts()
            for c in connectives:
                connective_counts[c["category"]] += 1

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
                    "dominant_relation": self._dominant_relation(connective_counts),
                    "verb_tense": tense,
                }
            )

            prev_entities = noun_lemmas
            prev_content = content_lemmas
            prev_tense = tense

        windowed_metrics = aggregate_windows(sent_metrics, window_size) if window_size and window_size > 1 else []

        return sent_metrics, windowed_metrics

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

    def analyze_text(self, text: str, window_size: int = DEFAULT_WINDOW_SIZE) -> Dict[str, object]:
        """
        Analyze discourse cohesion for a raw string and return sentence + window payloads.
        Window metrics are built with aggregate_windows: contiguous spans of `window_size` sentences are
        averaged for numeric fields (including nested dicts) and tagged with inclusive start/end indices.
        Raw sentence text and connective strings are not emitted.
        """
        doc = self.nlp(text)
        sentence_metrics, windowed_metrics = self.compute_sentence_metrics(doc, window_size=window_size)

        return {
            "meta": {
                "window_size": window_size,
                "num_sentences": len(sentence_metrics),
            },
            "sentences": sentence_metrics,
            "windows": windowed_metrics,
        }
