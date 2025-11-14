import statistics
import spacy
from .z_utils import sliding_windows, aggregate_windows
import numpy as np

"""

dependent on whole text analytics



"""



class LexicoSemanticsAnalyzer:
    def __init__(self, nlp):
        self.nlp = nlp

    # ---------------------
    # Lexical Density
    # ---------------------
    def analyze_lexical_density(self, doc, window_size=None):
        sent_metrics = []
        for sent in doc.sents:
            tokens = [t for t in sent if not t.is_punct]
            content_words = [t for t in tokens if t.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]]

            sent_metrics.append({
                "sentence_text": sent.text,
                "token_count": len(tokens),
                "content_count": len(content_words),
                "lexical_density": len(content_words) / len(tokens) if tokens else None,
            })

        return aggregate_windows(sent_metrics, window_size)


    # ---------------------
    # Information Content
    # ---------------------
    def analyze_information_content(self, doc, word_frequencies, window_size=None):
        sent_metrics = []

        for sent in doc.sents:
            ics = []
            for token in sent:
                if token.is_alpha:
                    freq = word_frequencies.get(token.text.lower())
                    if freq:
                        ics.append(-np.log(freq))

            sent_metrics.append({
                "sentence_text": sent.text,
                "information_content": float(np.mean(ics)) if ics else None,
                "ic_values": ics,
            })

        return aggregate_windows(sent_metrics, window_size)


    # ---------------------
    # Cohesion Metrics
    # ---------------------
    def analyze_cohesion(self, doc, window_size=None):
        prev_sent_content = None
        sent_metrics = []

        for sent in doc.sents:
            words = [t.lemma_.lower() for t in sent if t.is_alpha]
            overlap = None
            if prev_sent_content is not None:
                overlap = len(set(words) & set(prev_sent_content))

            sent_metrics.append({
                "sentence_text": sent.text,
                "cohesion_overlap": overlap,
                "content_words": words,
            })

            prev_sent_content = words

        return aggregate_windows(sent_metrics, window_size)


    # ---------------------
    # Semantic Roles / Arguments
    # ---------------------
    def analyze_semantic_roles(self, doc, window_size=None):
        sent_metrics = []

        for sent in doc.sents:
            roles = []
            for token in sent:
                if token.dep_ in ["nsubj", "dobj", "iobj", "pobj"]:
                    roles.append({
                        "role": token.dep_,
                        "text": token.text,
                        "head": token.head.text,
                    })

            sent_metrics.append({
                "sentence_text": sent.text,
                "semantic_roles": roles,
                "role_count": len(roles),
            })

        return aggregate_windows(sent_metrics, window_size)











# class LexicoSemanticsAnalyzer:
#     def __init__(self, spacy_model='en_core_web_sm', corpus_freqs=None):
#         self.nlp = spacy.load(spacy_model)
#         self.corpus_freqs = corpus_freqs or {}



#     # ----------------------------
#     # Average word frequency per sentence + sliding window
#     # ----------------------------
#     def compute_avg_word_frequency(self, doc, global_avg_freq=None, window_size=None):
#         """
#         Compute average word frequency and content/function ratio per sentence or window,
#         normalized by global frequency statistics if provided.
#         """
#         sent_metrics = []

#         for sent in doc.sents:
#             words = [token.text.lower() for token in sent if token.is_alpha]
#             total_tokens = len(words)
#             if self.corpus_freqs and words:
#                 freqs = [self.corpus_freqs.get(w, 1) for w in words]
#                 avg_word_freq = statistics.mean(freqs)
#             else:
#                 avg_word_freq = 0

#             # Normalization relative to global mean
#             if global_avg_freq and global_avg_freq > 0:
#                 norm_freq = round(avg_word_freq / global_avg_freq, 3)
#             else:
#                 norm_freq = round(avg_word_freq, 3)

#             content_words = [t for t in sent if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]
#             content_function_ratio = round(len(content_words)/total_tokens, 3) if total_tokens else 0

#             sent_metrics.append({
#                 "avg_word_freq": round(avg_word_freq, 3),
#                 "normalized_freq": norm_freq,
#                 "content_function_ratio": content_function_ratio
#             })

#         # Apply sliding window if requested
#         if window_size and window_size > 1:
#             windowed_metrics = []
#             for window in sliding_windows(sent_metrics, window_size):
#                 avg_freq = statistics.mean(d["avg_word_freq"] for d in window)
#                 avg_norm = statistics.mean(d["normalized_freq"] for d in window)
#                 avg_cfr = statistics.mean(d["content_function_ratio"] for d in window)
#                 windowed_metrics.append({
#                     "avg_word_freq": round(avg_freq, 3),
#                     "normalized_freq": round(avg_norm, 3),
#                     "content_function_ratio": round(avg_cfr, 3)
#                 })
#             return windowed_metrics

#         return sent_metrics



#     # ----------------------------
#     # Information content (surprisal) per sentence + sliding window
#     # ----------------------------
#     def compute_information_content(self, log_probs_list, global_avg_surprisal=None, window_size=None):
#         """
#         Compute per-sentence surprisal and variance, normalized by global average surprisal.
#         """
#         sent_metrics = []
#         for log_probs in log_probs_list:
#             if not log_probs:
#                 sent_metrics.append({"mean_surprisal": 0, "surprisal_variance": 0})
#                 continue

#             surprisals = [-lp for lp in log_probs]
#             mean_surprisal = statistics.mean(surprisals)
#             surprisal_variance = statistics.variance(surprisals) if len(surprisals) > 1 else 0

#             if global_avg_surprisal and global_avg_surprisal > 0:
#                 norm_surprisal = round(mean_surprisal / global_avg_surprisal, 3)
#             else:
#                 norm_surprisal = round(mean_surprisal, 3)

#             sent_metrics.append({
#                 "mean_surprisal": round(mean_surprisal, 3),
#                 "normalized_surprisal": norm_surprisal,
#                 "surprisal_variance": round(surprisal_variance, 3)
#             })

#         # Optional sliding window
#         if window_size and window_size > 1:
#             windowed_metrics = []
#             for window in sliding_windows(sent_metrics, window_size):
#                 avg_mean = statistics.mean(d["mean_surprisal"] for d in window)
#                 avg_norm = statistics.mean(d["normalized_surprisal"] for d in window)
#                 avg_var = statistics.mean(d["surprisal_variance"] for d in window)
#                 windowed_metrics.append({
#                     "mean_surprisal": round(avg_mean, 3),
#                     "normalized_surprisal": round(avg_norm, 3),
#                     "surprisal_variance": round(avg_var, 3)
#                 })
#             return windowed_metrics

#         return sent_metrics


#     # ----------------------------
#     # Extract semantic structures per clause + sliding window aggregation
#     # ----------------------------
#     def extract_semantic_structures(self, doc, window_size=None):
#         clause_metrics_per_sentence = []

#         for sent in doc.sents:
#             clauses = []
#             for token in sent:
#                 if token.pos_ != "VERB":
#                     continue

#                 # Determine clause type
#                 if token.dep_ == "ROOT":
#                     clause_type = "main"
#                 elif "advcl" in token.dep_ or "ccomp" in token.dep_ or "xcomp" in token.dep_:
#                     clause_type = "subordinate"
#                 elif token.dep_ == "conj":
#                     clause_type = "coordinate"
#                 else:
#                     continue  # skip verbs that are not part of a clause

#                 # Extract agent (subject) - full subtree
#                 subjects = [child for child in token.children if "subj" in child.dep_]
#                 agent_phrases = [" ".join([t.text for t in subj.subtree]) for subj in subjects]
#                 agent = "; ".join(agent_phrases) if agent_phrases else None

#                 # Extract patient (object) - full subtree
#                 objects = [child for child in token.children if "obj" in child.dep_]
#                 patient_phrases = [" ".join([t.text for t in obj.subtree]) for obj in objects]
#                 patient = "; ".join(patient_phrases) if patient_phrases else None

#                 clause_tokens = [t.text for t in token.subtree]

#                 clauses.append({
#                     "clause_level": clause_type,
#                     "predicate": token.lemma_,
#                     "agent": agent,
#                     "patient": patient,
#                     "clause_tokens": clause_tokens
#                 })

#             clause_metrics_per_sentence.append({
#                 "sentence": sent.text,
#                 "clauses": clauses,
#                 "num_clauses": len(clauses),
#                 "num_agents": sum(1 for c in clauses if c["agent"]),
#                 "num_patients": sum(1 for c in clauses if c["patient"])
#             })

#         # Sliding window aggregation
#         if window_size and window_size > 1:
#             windowed_metrics = []
#             for window in sliding_windows(clause_metrics_per_sentence, window_size):
#                 total_clauses = sum(d["num_clauses"] for d in window)
#                 total_agents = sum(d["num_agents"] for d in window)
#                 total_patients = sum(d["num_patients"] for d in window)

#                 windowed_metrics.append({
#                     "sentences": [d["sentence"] for d in window],
#                     "total_clauses": total_clauses,
#                     "total_agents": total_agents,
#                     "total_patients": total_patients
#                 })
#             return windowed_metrics

