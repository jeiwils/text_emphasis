"""






"""

from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import spacy

class ConceptExtractor:
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 language: str = "en_core_web_sm"):
        """Initialize with specified models."""
        self.encoder = SentenceTransformer(model_name)
        self.nlp = spacy.load(language)



        
    def extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases from text."""
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]
    

        
    def embed_phrases(self, phrases: List[str]) -> np.ndarray: #### seems like a ridiculous thing to have as an individual function?
        """Encode phrases into embeddings."""
        return self.encoder.encode(phrases)
    

        
    def cluster_embeddings(self, 
                          embeddings: np.ndarray,
                          min_cluster_size: int = 5) -> Dict[str, List[int]]:
        """Cluster embeddings using HDBSCAN."""
        clusterer = HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Organize results
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
            
        return clusters