from typing import List, Dict, Tuple
from collections import Counter
from pathlib import Path
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from nltk.corpus import stopwords
import nltk
import re

from x_configs import load_spacy_model
from .z_utils import embeddings_path


class ConceptExtractor:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        language: str = "en_core_web_sm",
    ):
        """Initialize with specified models."""
        self.encoder = SentenceTransformer(model_name)
        self.nlp = load_spacy_model(language)
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            self.stop_words = set(stopwords.words("english"))

    def extract_noun_phrases(self, text: str, lemmatize: bool = True) -> List[str]:
        """Extract noun phrases from text, optionally lemmatized, deduplicated in order."""
        doc = self.nlp(text)
        phrases = [chunk.text for chunk in doc.noun_chunks]

        if lemmatize:
            lemmatized = []
            for phrase in phrases:
                phrase_doc = self.nlp(phrase)
                tokens = []
                for token in phrase_doc:
                    if token.is_punct or token.is_space:
                        continue
                    if token.text.lower() in {"'s"}:
                        continue
                    lemma = token.lemma_.lower().strip("-'")
                    if lemma and any(c.isalnum() for c in lemma):
                        tokens.append(lemma)
                if tokens:
                    lemmatized.append(" ".join(tokens))
            phrases = lemmatized

        # Remove stopwords (phrase-level)
        filtered = []
        for phrase in phrases:
            tokens = [t for t in phrase.split() if t.lower() not in self.stop_words]
            if len(tokens) > 1 or (tokens and tokens[0].isalpha()):
                filtered.append(" ".join(tokens))

        phrases = filtered

        # Remove stray punctuation and possessives
        cleaned_phrases = []
        for phrase in phrases:
            clean_phrase = phrase.strip().lower()
            if re.fullmatch(r"'s", clean_phrase):
                continue
            if re.fullmatch(r"[-\"']+", clean_phrase):
                continue
            if not re.search(r"[a-zA-Z]", clean_phrase):
                continue
            cleaned_phrases.append(clean_phrase)

        # Deduplicate while preserving order
        seen = set()
        unique_phrases = []
        for p in cleaned_phrases:
            if p not in seen:
                unique_phrases.append(p)
                seen.add(p)

        return unique_phrases

    def embed_phrases(self, phrases: List[str]) -> np.ndarray:
        """Encode phrases into embeddings."""
        if not phrases:
            dim = self.encoder.get_sentence_embedding_dimension()
            return np.empty((0, dim))
        return self.encoder.encode(phrases)

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
    ) -> Dict[int, List[int]]:
        """Cluster embeddings using HDBSCAN and remove noise (-1)."""
        if embeddings is None or len(embeddings) == 0:
            return {}
        if len(embeddings) < min_cluster_size:
            return {}

        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(embeddings)

        # Organize results
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(cluster_labels):
            if label == -1:
                continue  # skip noise
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        return clusters

    def extract_clusters(
        self,
        text: str,
        min_cluster_size: int = 5,
        lemmatize: bool = True,
    ) -> Dict[int, List[str]]:
        """End-to-end helper: text -> noun phrases -> embeddings -> clustered phrase lists."""
        phrases = self.extract_noun_phrases(text, lemmatize=lemmatize)
        embeddings = self.embed_phrases(phrases)
        index_clusters = self.cluster_embeddings(
            embeddings, min_cluster_size=min_cluster_size
        )

        phrase_clusters: Dict[int, List[str]] = {}
        for label, indices in index_clusters.items():
            phrase_clusters[label] = [phrases[i] for i in indices]
        return phrase_clusters


def filter_top_n_phrases(phrases: List[str], n: int = 100) -> Tuple[List[str], List[int]]:
    """Keep only the top-n most frequent phrases and return them with their indices."""
    counts = Counter(phrases)
    top_phrases = [phrase for phrase, _ in counts.most_common(n)]
    filtered_indices = [i for i, p in enumerate(phrases) if p in top_phrases]
    filtered_phrases = [phrases[i] for i in filtered_indices]
    return filtered_phrases, filtered_indices


def generate_embeddings(normalised_text_path: Path, top_n: int = 100, use_existing: bool = True):
    """Extract top-N noun phrases and generate or load embeddings."""
    extractor = ConceptExtractor()
    base_name = normalised_text_path.stem.replace("_normalised", "")

    with open(normalised_text_path, "r", encoding="utf-8") as f:
        normalised_text = f.read()

    all_phrases = extractor.extract_noun_phrases(normalised_text, lemmatize=True)
    phrases, _ = filter_top_n_phrases(all_phrases, n=top_n)

    concept_dir = embeddings_path("concept") / base_name
    concept_dir.mkdir(parents=True, exist_ok=True)
    phrases_path = concept_dir / f"{base_name}_phrases.pkl"
    with open(phrases_path, "wb") as f:
        pickle.dump(phrases, f)

    embeddings_file = concept_dir / f"{base_name}_embeddings.pkl"
    if use_existing and embeddings_file.exists():
        with open(embeddings_file, "rb") as f:
            embeddings = pickle.load(f)
    else:
        embeddings = extractor.embed_phrases(phrases)
        with open(embeddings_file, "wb") as f:
            pickle.dump(embeddings, f)

    return normalised_text, phrases, embeddings
