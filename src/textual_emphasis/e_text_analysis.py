from typing import List
import pandas as pd

# Syntactic module (combined)
from d_all_syntactic import FullSyntacticAnalyzer
# Lexical module
from textual_emphasis.d2_all_lexical import EnhancedLexicalAnalyzer

class TextAnalyser:
    """
    Unified wrapper that combines syntactic and lexical analysis
    using the separate modules.
    """

    def __init__(self, language: str = "en_core_web_sm"):
        self.syntactic_analyzer = FullSyntacticAnalyzer(language=language)
        self.lexical_analyzer = EnhancedLexicalAnalyzer(language=language)

def analyze_corpus(self, texts: List[str], pos_filter=None):
    """
    Returns two DataFrames:
    - sentence_df: one row per sentence
    - text_df: aggregated features per text
    """
    sentence_rows = []
    text_rows = []

    syntactic_results = self.syntactic_analyzer.analyze_corpus(texts)

    for text, syn_res in zip(texts, syntactic_results):
        # Lexical analysis at sentence level
        doc = self.lexical_analyzer.nlp(text)
        for i, sent in enumerate(doc.sents):
            tokens = self.lexical_analyzer.tokenize(sent.text, pos_filter=pos_filter)
            sent_lexical = {
                "ttr": self.lexical_analyzer.type_token_ratio(tokens),
                "entropy": self.lexical_analyzer.shannon_entropy(tokens),
                "guiraud": self.lexical_analyzer.guiraud_index(tokens),
                "herdan_c": self.lexical_analyzer.herdan_c(tokens),
                **self.lexical_analyzer.word_length_stats(tokens),
                "n_tokens": len(tokens)
            }
            # Combine with sentence-level syntactic features
            sent_row = {**syn_res["sentences"][i], **sent_lexical, "sentence_index": i, "text": text}
            sentence_rows.append(sent_row)

        # Lexical analysis at text level
        tokens = self.lexical_analyzer.tokenize(text, pos_filter=pos_filter)
        text_lexical = {
            "ttr": self.lexical_analyzer.type_token_ratio(tokens),
            "entropy": self.lexical_analyzer.shannon_entropy(tokens),
            "guiraud": self.lexical_analyzer.guiraud_index(tokens),
            "herdan_c": self.lexical_analyzer.herdan_c(tokens),
            **self.lexical_analyzer.word_length_stats(tokens),
            "n_tokens": len(tokens)
        }
        # Combine with aggregated syntactic features
        text_row = {**syn_res["aggregated"], **text_lexical, "text": text}
        text_rows.append(text_row)

    sentence_df = pd.DataFrame(sentence_rows)
    text_df = pd.DataFrame(text_rows)
    return sentence_df, text_df