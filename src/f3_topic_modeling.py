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

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import HDBSCAN
import spacy


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
        self.nlp = spacy.load(language)
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