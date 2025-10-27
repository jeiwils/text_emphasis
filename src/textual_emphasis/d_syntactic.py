"""




Sentence Length / Punctuation	
- Abrupt, long, or heavily punctuated sentences can function as emphatic breaks or climaxes.	
- Split text into sentences with spaCy or NLTK; measure sentence lengths and punctuation counts; compute z-scores relative to corpus averages.
Marked Grammatical Constructions	
- Track predefined stylistically marked constructions (e.g., inversion, fronting, ellipsis, unusual tense or adverb placement) as signals of emphasis.	
- Parse with spaCy or Stanza; flag patterns via dependency labels or POS sequences; output as counts or binary features per sentence.
Dependency / Constituency Deviations	
- Quantify syntactic “surprise” relative to corpus; captures rare patterns that may indicate emergent or unexpected emphasis.	
- Parse sentences using spaCy/Stanza (dependency) or benepar/NLTK (constituency); encode structures; compute frequency deviations or TF–IDF–style rarity scores across corpus.




"""
from typing import List
import pandas as pd
import spacy
import benepar

class SyntacticAnalyzer:
    def __init__(self, language: str = "en_core_web_sm"):
        self.nlp = spacy.load(language)
        if not benepar.is_loaded('benepar_en3'):
            benepar.download('benepar_en3')
        self.nlp.add_pipe("benepar", config={"model": "benepar_en3"})

    # -------------------------------
    # Sentence Metrics + Z-scores
    # -------------------------------
    def sentence_metrics(self, texts: List[str]) -> pd.DataFrame:
        """Compute sentence metrics for all texts"""
        all_metrics = []
        for doc_id, text in enumerate(texts):
            doc = self.nlp(text)
            for sent in doc.sents:
                depth = self._compute_parse_depth_iterative(sent.root)
                all_metrics.append({
                    "doc_id": doc_id,
                    "sentence": sent.text,
                    "length": len(sent),
                    "n_tokens": len([t for t in sent if not t.is_punct]),
                    "n_punctuation": len([t for t in sent if t.is_punct]),
                    "depth": depth
                })
        df = pd.DataFrame(all_metrics)
        # Compute corpus z-scores
        for col in ["length", "n_tokens", "n_punctuation", "depth"]:
            df[f"{col}_z"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
        return df

    def _compute_parse_depth_iterative(self, root):
        max_depth = 0
        stack = [(root, 1)]
        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)
            for child in node.children:
                stack.append((child, depth + 1))
        return max_depth

    # -------------------------------
    # Marked Constructions
    # -------------------------------
    def marked_constructions(self, doc):
        n_tokens = len([t for t in doc if not t.is_punct])
        metrics = {
            "n_passive": len([t for t in doc if t.dep_ == "nsubjpass"]),
            "n_questions": len([t for t in doc if t.tag_ == "WRB"]),
            "n_subordinate": len([t for t in doc if t.dep_ == "mark"]),
            "n_relative": len([t for t in doc if t.dep_ == "relcl"]),
            "n_inversion": self._count_inversion(doc),
            "n_fronting": self._count_fronting(doc),
            "n_ellipsis": self._count_ellipsis(doc),
            "n_unusual_tense_adverb": self._count_unusual_tense_adverb(doc)
        }
        return {k: v / max(n_tokens,1) for k,v in metrics.items()}

    # --- heuristic methods ---
    def _count_inversion(self, doc): return sum(
        1 for sent in doc.sents if any(
            t.dep_ in ("ROOT", "aux") and any(c.dep_ in ("nsubj","nsubjpass") and c.i<t.i for c in t.children)
            for t in sent if t.pos_ in ("VERB","AUX")
        )
    )
    def _count_fronting(self, doc): return sum(
        1 for sent in doc.sents if sent[0].dep_ in ("dobj","obl","advmod") and sent[0].head.i > sent[0].i
    )
    def _count_ellipsis(self, doc): return sum(
        1 for sent in doc.sents if not any(t.dep_=="ROOT" and t.pos_=="VERB" for t in sent)
    )
    def _count_unusual_tense_adverb(self, doc): return sum(
        1 for sent in doc.sents for t in sent if t.tag_=="RB" and t.head.pos_ in ("AUX","VERB") and t.i < t.head.i
    )

    # -------------------------------
    # Dependency Patterns
    # -------------------------------
    def extract_syntactic_pattern(self, sent):
        return " ".join(f"{t.dep_}({t.head.pos_}->{t.pos_})" for t in sent if not t.is_punct)

    def parse_structures(self, text):
        doc = self.nlp(text)
        return [self.extract_syntactic_pattern(sent) for sent in doc.sents]

    # -------------------------------
    # Constituency Patterns
    # -------------------------------
    def extract_constituency_patterns(self, sent):
        tree = sent._.parse_string
        return [subtree for subtree in tree.split() if subtree]

    # -------------------------------
    # Unified DataFrame
    # -------------------------------
    def analyze_corpus(self, texts: List[str]) -> pd.DataFrame:
        """
        Return a single DataFrame with:
        - sentence metrics + z-scores
        - marked constructions
        - dependency TF–IDF
        - constituency TF–IDF
        - corpus deviation z-scores
        """
        # 1. Sentence metrics
        df_sent = self.sentence_metrics(texts)

        # 2. Marked constructions per doc
        constructions = []
        for doc_id, text in enumerate(texts):
            doc = self.nlp(text)
            mc = self.marked_constructions(doc)
            mc["doc_id"] = doc_id
            constructions.append(mc)
        df_constr = pd.DataFrame(constructions)

        # 3. Dependency TF–IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        dep_docs = [" ".join(self.parse_structures(text)) for text in texts]
        vectorizer_dep = TfidfVectorizer(analyzer="word", token_pattern=r"[^ ]+", min_df=1)
        dep_X = vectorizer_dep.fit_transform(dep_docs)
        df_dep = pd.DataFrame(dep_X.toarray(), columns=[f"dep_{c}" for c in vectorizer_dep.get_feature_names_out()])
        df_dep["doc_id"] = range(len(texts))

        # 4. Constituency TF–IDF
        cons_docs = []
        for text in texts:
            doc = self.nlp(text)
            patterns = []
            for sent in doc.sents:
                patterns.extend(self.extract_constituency_patterns(sent))
            cons_docs.append(" ".join(patterns))
        vectorizer_cons = TfidfVectorizer(analyzer="word", token_pattern=r"[^ ]+", min_df=1)
        cons_X = vectorizer_cons.fit_transform(cons_docs)
        df_cons = pd.DataFrame(cons_X.toarray(), columns=[f"cons_{c}" for c in vectorizer_cons.get_feature_names_out()])
        df_cons["doc_id"] = range(len(texts))

        # 5. Merge all at doc level
        df_final = df_constr.merge(df_dep, on="doc_id").merge(df_cons, on="doc_id")

        return df_final
