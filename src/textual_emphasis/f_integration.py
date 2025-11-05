"""

TO DO:
- SORT ALL THIS OUT INTO ONE CLASS????

"""



from sklearn.feature_extraction.text import TfidfVectorizer

def flatten_syntax_features(sentence_dict):
    """Turn one sentence analysis dict into a string of syntactic features."""
    features = []

    # Constituents
    for label, spans in sentence_dict["constituency"].items():
        features += [label] * len(spans)

    # Dependencies
    features += list(sentence_dict["dependencies"].keys())

    # Modifiers / Coordination
    for group in ["modifiers", "coordination"]:
        for label, vals in sentence_dict[group].items():
            features += [label] * len(vals)

    # Verbal morphosyntax
    features += sentence_dict["tense_aspect"]
    features += [sentence_dict["voice"]]
    features += sentence_dict["modality"]
    features += sentence_dict["negation"]

    # Agreement patterns
    features += [a.split(":")[1] for a in sentence_dict["agreement"] if ":" in a]

    # Stylistic flags
    for flag in ["inversion","fronting","ellipsis","apposition","parenthetical"]:
        if sentence_dict[flag]:
            features.append(flag)

    # Ordering features
    features += sentence_dict["constituent_sequence"]
    features += ["left_arc"] * sentence_dict["dependency_direction"]["left_arcs"]
    features += ["right_arc"] * sentence_dict["dependency_direction"]["right_arcs"]

    return " ".join(features)




def compute_syntactic_tfidf(analyzer, text):
    analysis = analyzer.analyze_text(text)
    docs = [flatten_syntax_features(sent) for sent in analysis]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)

    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names, docs




import numpy as np

def most_unusual_sentence(tfidf_matrix):
    mean_scores = np.asarray(tfidf_matrix.mean(axis=1)).flatten()
    idx = np.argmax(mean_scores)
    return idx, mean_scores[idx]
