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

    def analyze_corpus(self, texts: List[str], pos_filter=None) -> pd.DataFrame:
        """
        Returns a DataFrame with:
        - syntactic features (from FullSyntacticAnalyzer)
        - lexical features (from EnhancedLexicalAnalyzer)
        """
        # 1. Syntactic analysis
        df_syntactic = self.syntactic_analyzer.analyze_corpus(texts)

        # 2. Lexical analysis
        lexical_results = []
        for text in texts:
            tokens = self.lexical_analyzer.tokenize(text, pos_filter=pos_filter)
            lexical_results.append({
                "ttr": self.lexical_analyzer.type_token_ratio(tokens),
                "entropy": self.lexical_analyzer.shannon_entropy(tokens),
                "guiraud": self.lexical_analyzer.guiraud_index(tokens),
                "herdan_c": self.lexical_analyzer.herdan_c(tokens),
                **self.lexical_analyzer.word_length_stats(tokens),
                "n_tokens": len(tokens)
            })
        df_lexical = pd.DataFrame(lexical_results)

        # 3. Merge syntactic + lexical features
        df_final = pd.concat([df_syntactic.reset_index(drop=True), df_lexical.reset_index(drop=True)], axis=1)
        return df_final
