from typing import List, Dict
import spacy

class VerbalMorphosyntaxAnalyzer:
    """
    Analyze verbal morphosyntax for sentences.
    Captures:
    - Tense / Aspect
    - Voice (active/passive)
    - Modality
    - Negation
    - Agreement (Number / Person)
    """

    def __init__(self, language: str = "en_core_web_sm"):
        self.nlp = spacy.load(language)

    # -------------------------------
    # Tense / Aspect
    # -------------------------------
    def extract_tense_aspect(self, sent) -> List[Dict[str, List[str]]]:
        """
        Returns a list of dicts, one per verb/auxiliary:
        [{'verb': 'saw', 'tense_aspect': ['past']}, ...]
        """
        results = []
        for token in sent:
            if token.pos_ in ("AUX", "VERB"):
                descriptors = []
                if token.tag_ in ("VBD", "VBN"): descriptors.append("past")
                elif token.tag_ in ("VBZ", "VBP"): descriptors.append("present")
                if token.tag_ == "VBG": descriptors.append("progressive")
                if token.tag_ == "VBN" and any(child.lemma_ in ("have", "has", "had") for child in token.children):
                    descriptors.append("perfect")
                results.append({"verb": token.text, "tense_aspect": descriptors})
        return results

    def extract_voice(self, sent) -> List[Dict[str, str]]:
        """
        Returns a list of dicts, one per verb, with 'active' or 'passive'.
        """
        results = []
        for token in sent:
            if token.pos_ in ("AUX","VERB"):
                voice = "passive" if any(c.dep_ == "auxpass" for c in token.children) else "active"
                results.append({"verb": token.text, "voice": voice})
        return results

    # -------------------------------
    # Modality
    # -------------------------------
    def extract_modality(self, sent) -> List[str]:
        """
        Extract modal auxiliaries to capture modality (necessity, possibility, obligation)
        """
        modals = []
        for token in sent:
            if token.tag_ == "MD":  # modal auxiliary
                modals.append(token.text)
        return modals

    # -------------------------------
    # Negation
    # -------------------------------
    def extract_negation(self, sent) -> List[str]:
        """
        Extract negation markers for the sentence
        """
        negations = [token.text for token in sent if token.dep_ == "neg"]
        return negations

    # -------------------------------
    # Agreement / Number / Person
    # -------------------------------
    def extract_agreement(self, sent) -> List[str]:
        """
        Extract agreement features for verbs (number and person)
        """
        agreement = []
        for token in sent:
            if token.pos_ == "VERB":
                feats = token.morph
                agreement.append(f"{token.text}:{feats.get('Number')}:{feats.get('Person')}")
        return agreement

    # -------------------------------
    # Unified Sentence-Level Analysis
    # -------------------------------
    def analyze_text(self, text: str) -> List[Dict[str, object]]:
        """
        For each sentence, returns a dictionary with all verbal morphosyntax features.
        {
            'tense_aspect': [...],
            'voice': 'active'/'passive',
            'modality': [...],
            'negation': [...],
            'agreement': [...]
        }
        """
        doc = self.nlp(text)
        analysis = []
        for sent in doc.sents:
            tense_aspect = self.extract_tense_aspect(sent)
            voice = self.extract_voice(sent)
            modality = self.extract_modality(sent)
            negation = self.extract_negation(sent)
            agreement = self.extract_agreement(sent)
            analysis.append({
                "tense_aspect": tense_aspect,
                "voice": voice,
                "modality": modality,
                "negation": negation,
                "agreement": agreement
            })
        return analysis
