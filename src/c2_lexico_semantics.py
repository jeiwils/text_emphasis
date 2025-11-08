import statistics
import spacy

"""






    "lexico_semantics": {
        "vocabulary": { # I NEED TO PASS ANALYTICS ABOUT THE WHOLE CORPUS TO CALCULATE THESE
            "mattr_score": "<moving_average_type_token_ratio>", # proxy for lexical diversity 
            "avg_word_freq": "<mean_corpus_frequency>", # proxy for vocabulary rareness - in relation to whole corpus (i.e individual story)
            "content_function_ratio": "<content_words / total>" # proxy for density of informational content vs descriptive
        },
        "information_content": {
            "mean_surprisal": "<avg_llm_logprob_or_entropy>",
            "surprisal_variance": "<variance_llm_logprob>"
        },
        "semantic_structures": [ 
            {
            "clause_level": "<main/subordinate/coordinate>",
            "predicate": "<main_verb_lemma>",
            "agent": "<subject_phrase_or_token>",
            "patient": "<object_phrase_or_token>",
            "clause_tokens": ["<list_of_tokens_in_clause>"]
            }
        ]
        }



"""




class LexicoSemanticsAnalyzer:
    def __init__(self, spacy_model='en_core_web_sm', corpus_freqs=None):
        self.nlp = spacy.load(spacy_model)
        self.corpus_freqs = corpus_freqs or {}




    def compute_vocabulary_metrics(self, doc):
        words = [token.text.lower() for token in doc if token.is_alpha]
        types = set(words)
        total_tokens = len(words)

        # Moving Average Type-Token Ratio (mattr)
        window_size = 50
        if total_tokens < window_size:
            mattr = len(types) / total_tokens if total_tokens else 0
        else:
            ttr_values = []
            for i in range(total_tokens - window_size + 1):
                window = words[i:i+window_size]
                ttr_values.append(len(set(window)) / window_size)
            mattr = round(statistics.mean(ttr_values), 3)

        # Average word frequency (proxy for rareness)
        if self.corpus_freqs:
            freqs = [self.corpus_freqs.get(w, 1) for w in words]  # default freq=1
            avg_word_freq = round(statistics.mean(freqs), 3)
        else:
            avg_word_freq = 0

        # Content vs function words
        content_words = [token for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}]
        content_function_ratio = round(len(content_words) / total_tokens, 3) if total_tokens else 0

        return {
            "mattr_score": mattr,
            "avg_word_freq": avg_word_freq,
            "content_function_ratio": content_function_ratio
        }




    def compute_information_content(self, text):
        """
        CHANGE THIS TO TAKE PRECOMPUTED LOG PROBS FROM PREVIOUS MODULE 
        
        
        """

        log_probs = self.compute_log_probs(text)
        mean_surprisal = round(statistics.mean([-lp for lp in log_probs]), 3)
        surprisal_variance = round(statistics.variance([-lp for lp in log_probs]), 3) if len(log_probs) > 1 else 0
        return {
            "mean_surprisal": mean_surprisal,
            "surprisal_variance": surprisal_variance
        }




    def extract_semantic_structures(self, doc):
        structures = []
        for sent in doc.sents:
            for token in sent:
                # Focus on verbs as predicates
                if token.pos_ == "VERB":
                    # Determine clause type
                    if token.dep_ == "ROOT":
                        clause_type = "main"
                    elif "advcl" in token.dep_ or "ccomp" in token.dep_ or "xcomp" in token.dep_:
                        clause_type = "subordinate"
                    elif "conj" in token.dep_:
                        clause_type = "coordinate"
                    else:
                        continue  # Skip verbs that are not clearly part of a clause

                    # Extract agent (subject) - full subtree
                    subjects = [child for child in token.children if "subj" in child.dep_]
                    agent_phrases = []
                    for subj in subjects:
                        agent_phrases.append(" ".join([t.text for t in subj.subtree]))
                    agent = "; ".join(agent_phrases) if agent_phrases else None

                    # Extract patient (object) - full subtree
                    objects = [child for child in token.children if "obj" in child.dep_]
                    patient_phrases = []
                    for obj in objects:
                        patient_phrases.append(" ".join([t.text for t in obj.subtree]))
                    patient = "; ".join(patient_phrases) if patient_phrases else None

                    # All tokens in clause (subtree)
                    clause_tokens = [t.text for t in token.subtree]

                    structures.append({
                        "clause_level": clause_type,
                        "predicate": token.lemma_,
                        "agent": agent,
                        "patient": patient,
                        "clause_tokens": clause_tokens
                    })
        return structures
