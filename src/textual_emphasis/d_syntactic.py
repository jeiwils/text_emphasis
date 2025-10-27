"""

TO DO:
- 'grammar' version of TF-IDF (parse sentence units, structures etc... get a score for less common structures)





"""

from typing import List, Dict
import pandas as pd
import spacy
from spacy.tokens import Doc

class SyntacticAnalyzer:
    def __init__(self, language: str = "en_core_web_sm"):
        """Initialize with specified language model."""
        self.nlp = spacy.load(language)
    
    def sentence_metrics(self, text: str) -> pd.DataFrame:
        """Compute metrics for each sentence."""
        doc = self.nlp(text)
        metrics = []
        
        for sent in doc.sents:
            metrics.append({
                'text': sent.text,
                'length': len(sent),
                'n_tokens': len([t for t in sent if not t.is_punct]),
                'n_punctuation': len([t for t in sent if t.is_punct]),
                'depth': self._get_parse_depth(sent),
                'n_clauses': len([t for t in sent if t.dep_ == "ROOT"]),
            })
            
        return pd.DataFrame(metrics)
    
    def _get_parse_depth(self, span) -> int:
        """Calculate maximum depth of dependency parse tree."""
        def get_depth(token):
            if not list(token.children):
                return 0
            return 1 + max(get_depth(child) for child in token.children)
        
        return get_depth(span.root)
    
    def syntactic_rarity(self, doc: Doc) -> Dict[str, float]:
        """Compute metrics for unusual syntactic patterns."""
        metrics = {
            'n_passive': len([t for t in doc if t.dep_ == "nsubjpass"]),
            'n_questions': len([t for t in doc if t.tag_ == "WRB"]),
            'n_subordinate': len([t for t in doc if t.dep_ == "mark"]),
            'n_relative': len([t for t in doc if t.dep_ == "relcl"]),
        }
        
        # Normalize by document length
        doc_len = len([t for t in doc if not t.is_punct])
        return {k: v/doc_len for k, v in metrics.items()}