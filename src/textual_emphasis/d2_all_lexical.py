from typing import List, Dict, Optional
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

class EnhancedLexicalAnalyzer:
    """
    Enhanced lexical analysis:
    - TF-IDF with optional n-grams
    - Lexical entropy / diversity (Shannon entropy, TTR)
    - Lexical richness indices (Guiraud, Herdan's C)
    - Word length stats per sentence / document
    - POS-aware filtering
    - Optional frequency-aware rarity scoring
    """

    def __init__(self, language: str = "en_core_web_sm", ngram_range: tuple = (1,2), remove_stopwords: bool = True):
        self.tfidf = TfidfVectorizer(ngram_range=ngram_range)
        self.nlp = spacy.load(language)
        self.remove_stopwords = remove_stopwords

    # -------------------------------
    # TF-IDF
    # -------------------------------
    def compute_tfidf(self, corpus: List[str]) -> Dict[str, np.ndarray]:
        """
        Compute TF-IDF matrix and feature names
        """
        X = self.tfidf.fit_transform(corpus)
        return {"tfidf_matrix": X, "feature_names": self.tfidf.get_feature_names_out()}

    # -------------------------------
    # Token filtering
    # -------------------------------
    def tokenize(self, text: str, pos_filter: Optional[List[str]] = None) -> List[str]:
        doc = self.nlp(text)
        tokens = [
            t.text.lower() for t in doc
            if not t.is_punct and (not self.remove_stopwords or not t.is_stop)
            and (pos_filter is None or t.pos_ in pos_filter)
        ]
        return tokens

    # -------------------------------
    # Lexical diversity metrics
    # -------------------------------
    def type_token_ratio(self, tokens: List[str]) -> float:
        if not tokens: return 0.0
        return len(set(tokens)) / len(tokens)

    def shannon_entropy(self, tokens: List[str]) -> float:
        freq = Counter(tokens)
        total = sum(freq.values())
        if total == 0: return 0.0
        probs = [count / total for count in freq.values()]
        return -sum(p * np.log2(p) for p in probs)

    def guiraud_index(self, tokens: List[str]) -> float:
        return len(set(tokens)) / np.sqrt(len(tokens)) if tokens else 0.0

    def herdan_c(self, tokens: List[str]) -> float:
        return len(set(tokens)) / (len(tokens) ** (2/3)) if tokens else 0.0

    # -------------------------------
    # Word length statistics
    # -------------------------------
    def word_length_stats(self, tokens: List[str]) -> Dict[str, float]:
        lengths = [len(t) for t in tokens]
        if not lengths:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
        return {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'median': np.median(lengths)
        }

    # -------------------------------
    # Sentence-level analysis
    # -------------------------------
    def analyze_sentences(self, text: str, pos_filter: Optional[List[str]] = None) -> List[Dict[str, float]]:
        """
        Return a list of sentence-level lexical metrics
        """
        doc = self.nlp(text)
        results = []
        for sent in doc.sents:
            tokens = self.tokenize(sent.text, pos_filter=pos_filter)
            results.append({
                "sentence": sent.text,
                "ttr": self.type_token_ratio(tokens),
                "entropy": self.shannon_entropy(tokens),
                "guiraud": self.guiraud_index(tokens),
                "herdan_c": self.herdan_c(tokens),
                **self.word_length_stats(tokens),
                "n_tokens": len(tokens)
            })
        return results

    # -------------------------------
    # Corpus-level summary
    # -------------------------------
    def analyze_corpus(self, texts: List[str], pos_filter: Optional[List[str]] = None) -> List[Dict[str, float]]:
        """
        Aggregate lexical metrics over entire corpus
        """
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text, pos_filter=pos_filter))
        return [{
            "ttr": self.type_token_ratio(all_tokens),
            "entropy": self.shannon_entropy(all_tokens),
            "guiraud": self.guiraud_index(all_tokens),
            "herdan_c": self.herdan_c(all_tokens),
            **self.word_length_stats(all_tokens),
            "n_tokens": len(all_tokens)
        }]
