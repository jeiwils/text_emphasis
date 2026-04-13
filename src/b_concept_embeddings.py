"""
Concept embeddings from normalized text (noun phrase extraction + embeddings).

Input (generate_embeddings):
{
  "normalised_text_path": "data/texts/processed/normalised_texts/<category>/<name>_normalised.json",
  "file_contents": {"text": "<normalized text string>"},
  "top_n": 100
}

Output (return values):
{
  "normalised_text": "<string>",
  "phrases": ["noun phrase 1", "noun phrase 2", "..."],
  "embeddings": "numpy ndarray of shape (len(phrases), embedding_dim)"
}

Output files (written under data/analytics/embeddings/<genre>/<author>/<name>/):
- "<name>_phrases.pkl": List[str]
- "<name>_phrase_counts.json": Dict[str, int] (top-N phrase counts)
- "<name>_embeddings.pkl": numpy ndarray (raw)
- "<name>_embeddings_l2.pkl": numpy ndarray (L2-normalized)
"""

from typing import List, Dict, Tuple
from collections import Counter
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
import nltk
import re
import json

from .x_configs import (
    DEFAULT_CONCEPT_TOP_N,
    DEFAULT_SPACY_MODEL,
    DEFAULT_USE_EXISTING,
    MODEL_CONFIGS,
)
from .z_utils import (
    load_spacy_model,
    analytics_path,
    encode_texts,
    hdbscan_cluster_labels,
    labels_to_clusters,
    l2_normalize_embeddings,
)


class ConceptExtractor:
    def __init__(
        self,
        model_name: str = MODEL_CONFIGS["sentence_embedding"],
        language: str = DEFAULT_SPACY_MODEL,
        encoder: SentenceTransformer | None = None,
    ):
        """Initialize with specified models."""
        self.encoder = encoder or SentenceTransformer(model_name)
        self.nlp = load_spacy_model(language, disable=("ner",))
        try:
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            self.stop_words = set(stopwords.words("english"))

    def extract_noun_phrases(
        self,
        text: str,
        lemmatize: bool = True,
        *,
        dedupe: bool = True,
    ) -> List[str]:
        """Extract noun phrases from text, optionally lemmatized and deduped in order."""
        doc = self.nlp(text)
        phrases = [chunk.text for chunk in doc.noun_chunks]

        if lemmatize and phrases:
            lemmatized = []
            for phrase_doc in self.nlp.pipe(
                phrases,
                batch_size=256,
                disable=["parser", "ner"],
            ):
                tokens = []
                for token in phrase_doc:
                    if token.is_punct or token.is_space:
                        continue
                    if token.text.lower() in {"'s"}:
                        continue
                    lemma = token.lemma_.lower().strip("-'")
                    if lemma and any(c.isalnum() for c in lemma):
                        tokens.append(lemma)
                if tokens:
                    lemmatized.append(" ".join(tokens))
            phrases = lemmatized

        # Remove stopwords (phrase-level)
        filtered = []
        for phrase in phrases:
            tokens = [t for t in phrase.split() if t.lower() not in self.stop_words]
            if len(tokens) > 1 or (tokens and tokens[0].isalpha()):
                filtered.append(" ".join(tokens))

        phrases = filtered

        # Remove stray punctuation and possessives
        cleaned_phrases = []
        for phrase in phrases:
            clean_phrase = phrase.strip().lower()
            if re.fullmatch(r"'s", clean_phrase):
                continue
            if re.fullmatch(r"[-\"']+", clean_phrase):
                continue
            if not re.search(r"[a-zA-Z]", clean_phrase):
                continue
            cleaned_phrases.append(clean_phrase)

        if not dedupe:
            return cleaned_phrases

        # Deduplicate while preserving order
        seen = set()
        unique_phrases = []
        for p in cleaned_phrases:
            if p not in seen:
                unique_phrases.append(p)
                seen.add(p)

        return unique_phrases

    def extract_clusters(
        self,
        text: str,
        min_cluster_size: int = 5,
        lemmatize: bool = True,
    ) -> Dict[int, List[str]]:
        """End-to-end helper: text -> noun phrases -> embeddings -> clustered phrase lists."""
        phrases = self.extract_noun_phrases(text, lemmatize=lemmatize)
        embeddings = encode_texts(self.encoder, phrases)
        labels = hdbscan_cluster_labels(embeddings, min_cluster_size=min_cluster_size)
        index_clusters = labels_to_clusters(labels)

        phrase_clusters: Dict[int, List[str]] = {}
        for label, indices in index_clusters.items():
            phrase_clusters[label] = [phrases[i] for i in indices]
        return phrase_clusters


def filter_top_n_phrases(
    phrases: List[str], n: int = DEFAULT_CONCEPT_TOP_N
) -> Tuple[List[str], Dict[str, int]]:
    """Keep only the top-n most frequent unique phrases and return them with counts."""
    counts = Counter(phrases)
    if n is None or n <= 0:
        return [], {}
    top_phrases = [phrase for phrase, _ in counts.most_common(n)]
    top_counts = {phrase: int(counts[phrase]) for phrase in top_phrases}
    return top_phrases, top_counts


def generate_embeddings(
    normalised_text_path: Path,
    top_n: int = DEFAULT_CONCEPT_TOP_N,
    use_existing: bool = DEFAULT_USE_EXISTING,
    extractor: ConceptExtractor | None = None,
    *,
    quiet: bool = False,
):
    """Extract top-N noun phrases and generate or load embeddings."""
    base_name = normalised_text_path.stem.replace("_normalised", "")
    parent_category = normalised_text_path.parent.parent.name if normalised_text_path.parent.parent else ""
    author = normalised_text_path.parent.name
    category_parts = [author, base_name]
    if parent_category:
        category_parts.insert(0, parent_category)

    concept_dir = analytics_path("embeddings", category_parts)
    concept_dir.mkdir(parents=True, exist_ok=True)
    phrases_path = concept_dir / f"{base_name}_phrases.pkl"
    counts_path = concept_dir / f"{base_name}_phrase_counts.json"
    embeddings_raw_file = concept_dir / f"{base_name}_embeddings.pkl"
    embeddings_norm_file = concept_dir / f"{base_name}_embeddings_l2.pkl"

    if use_existing and phrases_path.exists() and (embeddings_norm_file.exists() or embeddings_raw_file.exists()):
        with open(normalised_text_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            normalised_text = data.get("text", "")

        with open(phrases_path, "rb") as f:
            phrases = pickle.load(f)

        if embeddings_norm_file.exists():
            with open(embeddings_norm_file, "rb") as f:
                embeddings = pickle.load(f)
        else:
            with open(embeddings_raw_file, "rb") as f:
                raw_embeddings = pickle.load(f)
            embeddings = l2_normalize_embeddings(raw_embeddings)
            with open(embeddings_norm_file, "wb") as f:
                pickle.dump(embeddings, f)

        if not quiet:
            print(f"[INFO] Skipping concept embeddings for {base_name} (exists)")
        return normalised_text, phrases, embeddings

    extractor = extractor or ConceptExtractor()
    with open(normalised_text_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        normalised_text = data.get("text", "")

    all_phrases = extractor.extract_noun_phrases(
        normalised_text,
        lemmatize=True,
        dedupe=False,
    )
    phrases, phrase_counts = filter_top_n_phrases(all_phrases, n=top_n)

    with open(phrases_path, "wb") as f:
        pickle.dump(phrases, f)
    if phrase_counts:
        with open(counts_path, "w", encoding="utf-8") as f:
            json.dump(phrase_counts, f, indent=2)

    if use_existing and embeddings_norm_file.exists():
        with open(embeddings_norm_file, "rb") as f:
            embeddings = pickle.load(f)
    elif use_existing and embeddings_raw_file.exists():
        with open(embeddings_raw_file, "rb") as f:
            raw_embeddings = pickle.load(f)
        embeddings = l2_normalize_embeddings(raw_embeddings)
        with open(embeddings_norm_file, "wb") as f:
            pickle.dump(embeddings, f)
    else:
        raw_embeddings = encode_texts(extractor.encoder, phrases, normalize=False)
        with open(embeddings_raw_file, "wb") as f:
            pickle.dump(raw_embeddings, f)
        embeddings = l2_normalize_embeddings(raw_embeddings)
        with open(embeddings_norm_file, "wb") as f:
            pickle.dump(embeddings, f)

    return normalised_text, phrases, embeddings
