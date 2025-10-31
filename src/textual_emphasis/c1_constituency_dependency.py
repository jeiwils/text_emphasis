from typing import List, Dict
import spacy
import benepar

class SyntacticAnalyzer:
    """
    Analyze the constituency and dependency structure of sentences.
    Captures:
    - Noun Phrases (NP)
    - Verb Phrases (VP)
    - Prepositional Phrases (PP)
    - Adjective Phrases (ADJP)
    - Adverb Phrases (ADVP)
    - Clauses (S, SBAR, CP, TP/IP)
    - Infinitival Clauses (S / VP[inf])
    """

    def __init__(self, language: str = "en_core_web_sm"):
        self.nlp = spacy.load(language)
        if not benepar.is_loaded("benepar_en3"):
            benepar.download("benepar_en3")
        self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    # -------------------------------
    # Constituency Extraction
    # -------------------------------
    def extract_constituency(self, sent) -> Dict[str, List[str]]:
        """
        Extract specific phrase types from a sentence using Benepar.
        Returns a dict mapping:
        'NP', 'VP', 'PP', 'ADJP', 'ADVP', 'CLAUSE', 'INFIN_CLAUSE' -> list of spans
        """
        tree_str = sent._.parse_string
        # Parse the string into a tree structure for recursive extraction
        # Benepar provides _._parse_tree which can be traversed
        tree = sent._.parse_tree
        phrases = {"NP": [], "VP": [], "PP": [], "ADJP": [], "ADVP": [], "CLAUSE": [], "INFIN_CLAUSE": []}

        def traverse(node):
            label = node.label()
            span_text = " ".join(node.leaves())
            if label in phrases:
                phrases[label].append(span_text)
            elif label in ("S", "SBAR", "CP", "TP", "IP"):
                phrases["CLAUSE"].append(span_text)
            elif label == "SINV" or (label == "S" and node[0].label() == "VP" and node[0][0].label() == "TO"):
                # heuristic for infinitival clauses
                phrases["INFIN_CLAUSE"].append(span_text)
            for child in node:
                if isinstance(child, benepar.Tree):
                    traverse(child)

        traverse(tree)
        return phrases

    def extract_constituency_corpus(self, text: str) -> List[Dict[str, List[str]]]:
        """Extract constituency phrases for all sentences in a text"""
        doc = self.nlp(text)
        return [self.extract_constituency(sent) for sent in doc.sents]

    # -------------------------------
    # Dependency Extraction
    # -------------------------------
    def extract_dependencies(self, sent) -> Dict[str, List[str]]:
        """
        Extract key dependency types for each sentence.
        Returns dict mapping:
        'nsubj', 'dobj', 'iobj', 'prep', 'amod', 'advmod', 'xcomp', 'relcl', etc. -> list of tokens
        """
        dep_dict = {}
        for token in sent:
            dep_dict.setdefault(token.dep_, []).append(token.text)
        return dep_dict

    def extract_dependencies_corpus(self, text: str) -> List[Dict[str, List[str]]]:
        """Extract dependencies for all sentences in a text"""
        doc = self.nlp(text)
        return [self.extract_dependencies(sent) for sent in doc.sents]

    # -------------------------------
    # Unified Sentence-Level Analysis
    # -------------------------------
    def analyze_text(self, text: str) -> List[Dict[str, Dict[str, List[str]]]]:
        """
        For each sentence, returns a dict with:
        {
            'constituency': {...},
            'dependencies': {...}
        }
        """
        doc = self.nlp(text)
        analysis = []
        for sent in doc.sents:
            constituency = self.extract_constituency(sent)
            dependencies = self.extract_dependencies(sent)
            analysis.append({"constituency": constituency, "dependencies": dependencies})
        return analysis
