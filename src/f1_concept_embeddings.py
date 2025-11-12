"""
TO DO:
- lemmatize the tokens?
- is there any way to do this with a language model, so that it's not just surface level? So that it's with IDEAS/CONCEPTS rather than WORDS


"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import spacy
from nltk.corpus import stopwords




class ConceptExtractor:
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 language: str = "en_core_web_sm"):
        """Initialize with specified models."""
        self.encoder = SentenceTransformer(model_name)
        self.nlp = spacy.load(language)
        self.stop_words = set(stopwords.words('english'))  # load once here



    def extract_noun_phrases(self, text: str, lemmatize: bool = True) -> List[str]:
        """Extract noun phrases from text, optionally lemmatized, deduplicated in order."""
        doc = self.nlp(text)
        phrases = [chunk.text for chunk in doc.noun_chunks]

        if lemmatize:
            lemmatized = []
            for phrase in phrases:
                phrase_doc = self.nlp(phrase)
                lemmatized.append(' '.join([token.lemma_ for token in phrase_doc]))
            phrases = lemmatized

        # Remove stopwords (phrase-level)
        filtered = []
        for phrase in phrases:
            # skip if any token is a stopword
            if all(token.lower() not in self.stop_words for token in phrase.split()):
                # skip if phrase is too short or just punctuation
                clean_phrase = phrase.strip()
                if len(clean_phrase) > 1 and any(c.isalnum() for c in clean_phrase):
                    filtered.append(clean_phrase)
        phrases = filtered

        # Deduplicate while preserving order
        seen = set()
        unique_phrases = []
        for p in phrases:
            if p not in seen:
                unique_phrases.append(p)
                seen.add(p)

        return unique_phrases


    

        
    def embed_phrases(self, phrases: List[str]) -> np.ndarray:
        """Encode phrases into embeddings."""
        return self.encoder.encode(phrases)
    

        
    def cluster_embeddings(self, 
                        embeddings: np.ndarray,
                        min_cluster_size: int = 5) -> Dict[int, List[int]]:
        """Cluster embeddings using HDBSCAN and remove noise (-1)."""
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(embeddings)

        # Organize results
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label == -1:
                continue  # skip noise
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        return clusters






