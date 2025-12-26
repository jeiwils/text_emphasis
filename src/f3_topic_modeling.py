"""

Neural topic modeling for long-form text.

Input:
- a raw text string (document, chapter, article, etc.)

Output:
- list of TopicResult objects:
  - topic_id (int)
  - keywords (top TF-IDF terms for the cluster)
  - mentions (sentence spans w/ character offsets for localisation)


"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from .z_utils import processed_text_path, topic_modelling_path
from x_configs import load_spacy_model


@dataclass
class TopicMention:
    sentence_index: int
    text: str
    start_char: int
    end_char: int


@dataclass
class TopicResult:
    topic_id: int
    keywords: List[str]
    mentions: List[TopicMention]


class NeuralTopicModeler:
    """
    Clusters sentence embeddings, extracts keywords, and returns
    localized mentions (sentence index + char offsets).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        language: str = "en_core_web_sm",
        stop_words: str = "english",
    ):
        self.encoder = SentenceTransformer(model_name)
        self.nlp = load_spacy_model(language)
        self.stop_words = stop_words

    def segment_sentences(self, text: str) -> List[TopicMention]:
        """Split text into sentences with char offsets."""
        doc = self.nlp(text)
        sentences = []
        for idx, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            sentences.append(
                TopicMention(
                    sentence_index=idx,
                    text=sent_text,
                    start_char=sent.start_char,
                    end_char=sent.end_char,
                )
            )
        return sentences

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """Encode sentences into embeddings."""
        return self.encoder.encode(sentences)

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Cluster embeddings with HDBSCAN."""
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        return clusterer.fit_predict(embeddings)

    def _build_topic_keywords(
        self,
        cluster_docs: List[str],
        labels: List[int],
        top_n: int = 8,
        ngram_range: Tuple[int, int] = (1, 2),
    ) -> Dict[int, List[str]]:
        """Build TF-IDF keywords for each topic cluster."""
        vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            ngram_range=ngram_range,
        )
        tfidf = vectorizer.fit_transform(cluster_docs)
        feature_names = np.array(vectorizer.get_feature_names_out())

        keywords: Dict[int, List[str]] = {}
        for row_idx, label in enumerate(labels):
            row = tfidf[row_idx]
            if row.nnz == 0:
                keywords[label] = []
                continue
            scores = row.toarray().ravel()
            top_indices = scores.argsort()[::-1][:top_n]
            keywords[label] = feature_names[top_indices].tolist()
        return keywords

    def extract_topics(
        self,
        text: str,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        top_n: int = 8,
        ngram_range: Tuple[int, int] = (1, 2),
    ) -> List[TopicResult]:
        """Main entrypoint: returns clustered topics + localized mentions."""
        sentences = self.segment_sentences(text)
        if not sentences:
            return []

        sentence_texts = [s.text for s in sentences]
        if len(sentence_texts) < min_cluster_size:
            labels = np.zeros(len(sentence_texts), dtype=int)
        else:
            embeddings = self.embed_sentences(sentence_texts)
            labels = self.cluster_embeddings(
                embeddings,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )

        topic_docs: Dict[int, List[str]] = {}
        topic_mentions: Dict[int, List[TopicMention]] = {}
        for sentence, label in zip(sentences, labels):
            if label == -1:
                continue
            topic_docs.setdefault(label, []).append(sentence.text)
            topic_mentions.setdefault(label, []).append(sentence)

        if not topic_docs:
            return []

        cluster_labels = sorted(topic_docs.keys())
        cluster_docs = [" ".join(topic_docs[label]) for label in cluster_labels]
        keywords = self._build_topic_keywords(
            cluster_docs,
            cluster_labels,
            top_n=top_n,
            ngram_range=ngram_range,
        )

        results = []
        for label in cluster_labels:
            results.append(
                TopicResult(
                    topic_id=label,
                    keywords=keywords.get(label, []),
                    mentions=topic_mentions.get(label, []),
                )
            )
        return results


def serialize_topic_results(topic_results: List[TopicResult]) -> List[Dict[str, Any]]:
    """Convert TopicResult objects to plain dicts for JSON output."""
    return [
        {
            "topic_id": result.topic_id,
            "keywords": result.keywords,
            "mentions": [
                {
                    "sentence_index": mention.sentence_index,
                    "start_char": mention.start_char,
                    "end_char": mention.end_char,
                    "text": mention.text,
                }
                for mention in result.mentions
            ],
        }
        for result in topic_results
    ]


def count_mentions_per_sentence(topic_results: List[TopicResult]) -> Dict[int, int]:
    """Count how many topic mentions occur in each sentence index."""
    counts: Dict[int, int] = {}
    for result in topic_results:
        for mention in result.mentions:
            counts[mention.sentence_index] = counts.get(mention.sentence_index, 0) + 1
    return counts


def run_topic_modelling(use_existing: bool = True):
    """Batch topic modelling across all cleaned text files."""
    modeler = NeuralTopicModeler()
    cleaned_root = processed_text_path("cleaned")
    output_root = topic_modelling_path()
    output_root.mkdir(parents=True, exist_ok=True)

    for subdir in cleaned_root.iterdir():
        if not subdir.is_dir():
            continue
        print(f"Processing category: {subdir.name}")

        out_subdir = output_root / subdir.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        for file in subdir.glob("*.txt"):
            output_file = out_subdir / f"{file.stem}_topics.json"
            if use_existing and output_file.exists():
                print(f"Skipping {file.name} (exists)")
                continue

            text = file.read_text(encoding="utf-8")
            print(f"Extracting topics for {file.name}...")

            topic_results = modeler.extract_topics(text)
            result = {
                "filename": file.name,
                "topics": serialize_topic_results(topic_results),
                "mentions_per_sentence": count_mentions_per_sentence(topic_results),
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            print(f"Saved topic modelling to {output_file.name}")

    print("All done.")


if __name__ == "__main__":
    run_topic_modelling()
