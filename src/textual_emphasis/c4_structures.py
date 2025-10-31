from typing import List, Dict
import spacy
import benepar

class StylisticStructuresAnalyzer:
    """
    Capture marked or noncanonical syntactic constructions:
    - Inversion
    - Fronting / Topicalization
    - Ellipsis / Gapping / Stripping / Sluicing
    - Apposition (expressive)
    - Parenthetical clauses
    """

    def __init__(self, language: str = "en_core_web_sm"):
        self.nlp = spacy.load(language)
        if not benepar.is_loaded("benepar_en3"):
            benepar.download("benepar_en3")
        self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    # -------------------------------
    # Inversion
    # -------------------------------
    def detect_inversion(self, sent) -> bool:
        """
        Detect subject-auxiliary inversion:
        e.g., "Never have I seen..."
        spaCy: auxiliary precedes subject
        """
        for token in sent:
            if token.pos_ in ("AUX", "VERB") and any(
                c.dep_ in ("nsubj", "nsubjpass") and c.i > token.i
                for c in token.children
            ):
                return True
        return False

    # -------------------------------
    # Fronting / Topicalization
    # -------------------------------
    def detect_fronting(self, sent) -> bool:
        """
        Detect fronted constituents: NP or adverb moved to sentence start
        spaCy: dislocated, Benepar: NP moved
        """
        first_token = sent[0]
        if first_token.dep_ in ("dobj", "obl", "advmod") and first_token.head.i > first_token.i:
            return True
        return False

    # -------------------------------
    # Ellipsis / Gapping / Stripping / Sluicing
    # -------------------------------
    def detect_ellipsis(self, sent) -> bool:
        """
        Heuristic: sentence lacks a main verb (ROOT) or uses 'orphan'
        """
        root_verbs = [t for t in sent if t.dep_ == "ROOT" and t.pos_ == "VERB"]
        orphan_tokens = [t for t in sent if t.dep_ == "orphan"]
        return len(root_verbs) == 0 or len(orphan_tokens) > 0

    # -------------------------------
    # Apposition
    # -------------------------------
    def detect_apposition(self, sent) -> bool:
        """
        Detect appositive constructions
        spaCy: appos
        """
        return any(t.dep_ == "appos" for t in sent)

    # -------------------------------
    # Parenthetical Clauses
    # -------------------------------
    def detect_parenthetical(self, sent) -> bool:
        """
        Detect parenthetical clauses
        spaCy: parataxis; Benepar: PRN
        """
        # spaCy approach: parataxis with commas
        return any(t.dep_ == "parataxis" for t in sent) or "(PRN" in sent._.parse_string

    # -------------------------------
    # Unified Sentence-Level Analysis
    # -------------------------------
    def analyze_text(self, text: str) -> List[Dict[str, bool]]:
        """
        For each sentence, returns a dictionary with stylistic constructions:
        {
            'inversion': bool,
            'fronting': bool,
            'ellipsis': bool,
            'apposition': bool,
            'parenthetical': bool
        }
        """
        doc = self.nlp(text)
        analysis = []
        for sent in doc.sents:
            analysis.append({
                "inversion": self.detect_inversion(sent),
                "fronting": self.detect_fronting(sent),
                "ellipsis": self.detect_ellipsis(sent),
                "apposition": self.detect_apposition(sent),
                "parenthetical": self.detect_parenthetical(sent)
            })
        return analysis
