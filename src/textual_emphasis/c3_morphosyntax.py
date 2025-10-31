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
    def extract_tense_aspect(self, sent) -> List[str]:
        """
        Extract tense/aspect info based on auxiliaries and verb forms.
        Returns list of descriptors like 'past', 'present', 'progressive', 'perfect'
        """
        descriptors = []
        for token in sent:
            if token.pos_ in ("AUX", "VERB"):
                # Check auxiliary for tense/aspect
                if token.tag_ in ("VBD", "VBN"):
                    descriptors.append("past")
                elif token.tag_ in ("VBZ", "VBP"):
                    descriptors.append("present")
                # progressive aspect: 'ing'
                if token.tag_ == "VBG":
                    descriptors.append("progressive")
                # perfect aspect: 'have' + past participle
                if token.tag_ == "VBN" and any(child.lemma_ in ("have", "has", "had") for child in token.children):
                    descriptors.append("perfect")
        return list(set(descriptors))

    # -------------------------------
    # Voice
    # -------------------------------
    def extract_voice(self, sent) -> str:
        """
        Determine if the sentence uses active or passive voice.
        """
        for token in sent:
            if token.dep_ == "auxpass":
                return "passive"
        return "active"

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
