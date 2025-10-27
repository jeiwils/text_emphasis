"""






"""

from typing import List, Dict
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

class LexicalAnalyzer:
    def __init__(self):
        """Initialize the lexical analyzer."""
        self.tfidf = TfidfVectorizer()
        
    def compute_tfidf(self, corpus: List[str]) -> np.ndarray:
        """
        
        word frequency/rarity

        Compute TF-IDF scores for documents in corpus.
        
        
        """
        return self.tfidf.fit_transform(corpus)
    
    def compute_lexical_entropy(self, tokens: List[str]) -> float:
        """
        
        word variation within certain window 

        Compute Shannon entropy of token distribution.
        
        """
        freq = Counter(tokens)
        total = sum(freq.values())
        probs = [count/total for count in freq.values()]
        return -sum(p * np.log2(p) for p in probs)
    
    def word_length_stats(self, tokens: List[str]) -> Dict[str, float]:
        """
        
        word length

        Compute word length statistics.
        
        """
        lengths = [len(token) for token in tokens]
        return {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'median': np.median(lengths)
        }
    
    def type_token_ratio(self, tokens: List[str]) -> float:
        """
        
        not sure what this is 

        Compute Type-Token Ratio (TTR).
        
        
        """
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)