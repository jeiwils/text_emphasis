"""



TF–IDF	
- Identify unusual or infrequent words/phrases to detect distinctive vocabulary marking central topics.	
- scikit-learn TfidfVectorizer or Gensim; tokenize with spaCy or NLTK; n-grams for multi-word phrases.
Lexical 
- Entropy / Diversity	Measure lexical variation; higher or lower diversity may indicate sections of emphasis.	
- Compute type–token ratio or Shannon entropy via NLTK or plain Python; optional: TextStat for additional metrics.
Vocabulary Length / Rhythm	
- Variations in word length can create rhythmic emphasis and affect attention.	
- Tokenize with spaCy; compute word lengths per sentence and calculate variance or z-scores across the corpus.





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
        
        word variation within certain window (i.e to see how varied vocabulary is in areas where there's a concentration of ndoes of a certain community in the networkx graph)

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