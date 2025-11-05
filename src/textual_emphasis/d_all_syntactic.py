from typing import List, Dict
import spacy
import benepar

class FullSyntacticAnalyzer:
    """
    Full syntactic analysis pipeline combining:
    1. Constituency and Dependency
    2. Attachment and Linkage
    3. Verbal Morphosyntax
    4. Predefined Stylistic Structures
    """

    def __init__(self, language: str = "en_core_web_sm"):
        # Load spaCy and Benepar
        self.nlp = spacy.load(language)
        try:
            benepar.download("benepar_en3")
        except Exception:
            pass
        self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    # -------------------------------
    # Module 1: Constituency & Dependency
    # -------------------------------
    def extract_constituency(self, sent) -> Dict[str, List[str]]:
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
                phrases["INFIN_CLAUSE"].append(span_text)
            for child in node:
                if isinstance(child, benepar.Tree):
                    traverse(child)
        traverse(tree)
        return phrases

    def extract_dependencies(self, sent) -> Dict[str, List[str]]:
        dep_dict = {}
        for token in sent:
            dep_dict.setdefault(token.dep_, []).append(token.text)
        return dep_dict

    # -------------------------------
    # Module 2: Attachment & Linkage
    # -------------------------------
    def extract_modifiers(self, sent) -> Dict[str, List[str]]:
        modifiers = {
            "adjectival": [], "adverbial": [], "prepositional": [],
            "relative_clause": [], "appositive": [], "participial_clause": []
        }
        for token in sent:
            if token.dep_ == "amod": modifiers["adjectival"].append(token.text)
            elif token.dep_ == "advmod": modifiers["adverbial"].append(token.text)
            elif token.dep_ == "prep": modifiers["prepositional"].append(token.text)
            elif token.dep_ == "relcl": modifiers["relative_clause"].append(token.text)
            elif token.dep_ == "appos": modifiers["appositive"].append(token.text)
            elif token.dep_ == "advcl": modifiers["participial_clause"].append(token.text)
        return modifiers

    def extract_coordination(self, sent) -> Dict[str, List[str]]:
        coordination = {"coordinated_conj": [], "correlative_constructions": [], "conjunctive_adverbs": []}
        for token in sent:
            if token.dep_ == "conj": coordination["coordinated_conj"].append(token.text)
            if token.dep_ == "cc" and any(c.dep_ == "conj" for c in token.head.children):
                coordination["correlative_constructions"].append(token.text)
            if token.pos_ == "ADV" and token.dep_ in ("advmod", "cc"):
                coordination["conjunctive_adverbs"].append(token.text)
        return coordination

    # -------------------------------
    # Module 3: Verbal Morphosyntax
    # -------------------------------
    def extract_tense_aspect(self, sent) -> List[str]:
        descriptors = []
        for token in sent:
            if token.pos_ in ("AUX", "VERB"):
                if token.tag_ in ("VBD", "VBN"): descriptors.append("past")
                elif token.tag_ in ("VBZ", "VBP"): descriptors.append("present")
                if token.tag_ == "VBG": descriptors.append("progressive")
                if token.tag_ == "VBN" and any(child.lemma_ in ("have","has","had") for child in token.children):
                    descriptors.append("perfect")
        return list(set(descriptors))

    def extract_voice(self, sent) -> str:
        return "passive" if any(t.dep_=="auxpass" for t in sent) else "active"

    def extract_modality(self, sent) -> List[str]:
        return [t.text for t in sent if t.tag_ == "MD"]

    def extract_negation(self, sent) -> List[str]:
        return [t.text for t in sent if t.dep_=="neg"]

    def extract_agreement(self, sent) -> List[str]:
        agreement = []
        for t in sent:
            if t.pos_=="VERB":
                agreement.append(f"{t.text}:{t.morph.get('Number')}:{t.morph.get('Person')}")
        return agreement

    # -------------------------------
    # Module 4: Predefined Stylistic Structures
    # -------------------------------
    def detect_inversion(self, sent) -> bool:
        return any(t.pos_ in ("AUX","VERB") and any(c.dep_ in ("nsubj","nsubjpass") and c.i > t.i for c in t.children) for t in sent)

    def detect_fronting(self, sent) -> bool:
        first_token = sent[0]
        return first_token.dep_ in ("dobj","obl","advmod") and first_token.head.i > first_token.i

    def detect_ellipsis(self, sent) -> bool:
        root_verbs = [t for t in sent if t.dep_=="ROOT" and t.pos_=="VERB"]
        orphan_tokens = [t for t in sent if t.dep_=="orphan"]
        return len(root_verbs)==0 or len(orphan_tokens)>0

    def detect_apposition(self, sent) -> bool:
        return any(t.dep_=="appos" for t in sent)

    def detect_parenthetical(self, sent) -> bool:
        tree_str = str(sent._.parse_tree)
        return any(t.dep_=="parataxis" for t in sent) or "(PRN" in tree_str
    


    ###### ORDERING 


    def extract_constituent_sequence(self, sent):
        """Return a sequence of top-level phrase labels (NP, VP, PP...)"""
        tree = sent._.parse_tree
        return [child.label() for child in tree if isinstance(child, benepar.Tree)]


    def dependency_direction_stats(self, sent):
        """Count left- vs right-headed dependencies."""
        left, right = 0, 0
        for token in sent:
            for child in token.children:
                if child.i < token.i:
                    left += 1
                else:
                    right += 1
        return {"left_arcs": left, "right_arcs": right}


    # -------------------------------
    # Unified Sentence-Level Analysis
    # -------------------------------
    def analyze_text(self, text: str) -> List[Dict[str, object]]:
        """
        Returns a list of sentence-level analyses combining all four modules:
        {
            'constituency': {...},
            'dependencies': {...},
            'modifiers': {...},
            'coordination': {...},
            'tense_aspect': [...],
            'voice': 'active'/'passive',
            'modality': [...],
            'negation': [...],
            'agreement': [...],
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
                "constituency": self.extract_constituency(sent),
                "dependencies": self.extract_dependencies(sent),
                "modifiers": self.extract_modifiers(sent),
                "coordination": self.extract_coordination(sent),
                "tense_aspect": self.extract_tense_aspect(sent),
                "voice": self.extract_voice(sent),
                "modality": self.extract_modality(sent),
                "negation": self.extract_negation(sent),
                "agreement": self.extract_agreement(sent),
                "inversion": self.detect_inversion(sent),
                "fronting": self.detect_fronting(sent),
                "ellipsis": self.detect_ellipsis(sent),
                "apposition": self.detect_apposition(sent),
                "parenthetical": self.detect_parenthetical(sent),
                "constituent_sequence": self.extract_constituent_sequence(sent),
                "dependency_direction": self.dependency_direction_stats(sent),
            })
        return analysis
    

    def analyze_corpus(self, texts: List[str]) -> List[Dict[str, object]]:
        """
        For each text, return:
        {
            'sentences': [sentence-level dicts],
            'aggregated': aggregated counts/flags per text
        }
        Handles per-verb features (tense_aspect, voice) as lists of dicts.
        """
        corpus_results = []

        for text in texts:
            sent_results = self.analyze_text(text)  # sentence-level
            aggregated = {}

            for sent_res in sent_results:
                for key, val in sent_res.items():
                    if isinstance(val, list):
                        if val and isinstance(val[0], dict):  # list of dicts (per-verb)
                            for d in val:
                                for subkey, subval in d.items():
                                    if isinstance(subval, list):
                                        aggregated[f"{key}_{subkey}"] = aggregated.get(f"{key}_{subkey}", 0) + len(subval)
                                    else:
                                        aggregated[f"{key}_{subkey}"] = aggregated.get(f"{key}_{subkey}", 0) + 1
                        else:
                            aggregated[key] = aggregated.get(key, 0) + len(val)
                    elif isinstance(val, dict):
                        for subkey, subval in val.items():
                            aggregated[subkey] = aggregated.get(subkey, 0) + len(subval)
                    elif isinstance(val, bool):
                        aggregated[key] = aggregated.get(key, False) or val
                    else:
                        aggregated[key] = val

            corpus_results.append({
                "sentences": sent_results,
                "aggregated": aggregated
            })

        return corpus_results
    




