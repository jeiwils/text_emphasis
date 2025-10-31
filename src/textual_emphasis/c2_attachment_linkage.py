from typing import List, Dict
import spacy

class AttachmentLinkageAnalyzer:
    """
    Analyze phrase attachment, modifiers, and coordination.
    Captures:
    - Adjectival modifiers (amod)
    - Adverbial modifiers (advmod)
    - Prepositional adjuncts (prep)
    - Relative clause modifiers (relcl)
    - Appositive modifiers (appos)
    - Participial clause modifiers (advcl)
    - Coordinated NPs / Clauses (conj)
    - Correlative constructions (cc + conj)
    - Conjunctive adverbs / connectives (advmod / cc)
    """

    def __init__(self, language: str = "en_core_web_sm"):
        self.nlp = spacy.load(language)

    # -------------------------------
    # Modifier / Adjunct Layer
    # -------------------------------
    def extract_modifiers(self, sent) -> Dict[str, List[str]]:
        """
        Extract modifiers and adjuncts for a single sentence.
        """
        modifiers = {
            "adjectival": [],
            "adverbial": [],
            "prepositional": [],
            "relative_clause": [],
            "appositive": [],
            "participial_clause": []
        }

        for token in sent:
            if token.dep_ == "amod":
                modifiers["adjectival"].append(token.text)
            elif token.dep_ == "advmod":
                modifiers["adverbial"].append(token.text)
            elif token.dep_ == "prep":
                modifiers["prepositional"].append(token.text)
            elif token.dep_ == "relcl":
                modifiers["relative_clause"].append(token.text)
            elif token.dep_ == "appos":
                modifiers["appositive"].append(token.text)
            elif token.dep_ == "advcl":
                # heuristically treat participial clauses as advcl
                modifiers["participial_clause"].append(token.text)
        return modifiers

    def extract_modifiers_corpus(self, text: str) -> List[Dict[str, List[str]]]:
        """
        Extract all modifiers/adjuncts for each sentence in the text.
        """
        doc = self.nlp(text)
        return [self.extract_modifiers(sent) for sent in doc.sents]

    # -------------------------------
    # Coordination and Conjunctions
    # -------------------------------
    def extract_coordination(self, sent) -> Dict[str, List[str]]:
        """
        Extract coordinated elements and connectives.
        """
        coordination = {
            "coordinated_conj": [],
            "correlative_constructions": [],
            "conjunctive_adverbs": []
        }

        for token in sent:
            # Coordinated NPs / Clauses
            if token.dep_ == "conj":
                coordination["coordinated_conj"].append(token.text)
            # Correlative constructions: detect cc + conj patterns
            if token.dep_ == "cc":
                # heuristic: check if sibling is conj
                if any(child.dep_ == "conj" for child in token.head.children):
                    coordination["correlative_constructions"].append(token.text)
            # Conjunctive adverbs / connectives
            if token.pos_ == "ADV" and token.dep_ in ("advmod", "cc"):
                coordination["conjunctive_adverbs"].append(token.text)

        return coordination

    def extract_coordination_corpus(self, text: str) -> List[Dict[str, List[str]]]:
        """
        Extract coordination/conjunctions for each sentence in the text.
        """
        doc = self.nlp(text)
        return [self.extract_coordination(sent) for sent in doc.sents]

    # -------------------------------
    # Unified Sentence-Level Analysis
    # -------------------------------
    def analyze_text(self, text: str) -> List[Dict[str, Dict[str, List[str]]]]:
        """
        For each sentence, returns a dict with:
        {
            'modifiers': {...},
            'coordination': {...}
        }
        """
        doc = self.nlp(text)
        analysis = []
        for sent in doc.sents:
            modifiers = self.extract_modifiers(sent)
            coordination = self.extract_coordination(sent)
            analysis.append({"modifiers": modifiers, "coordination": coordination})
        return analysis
