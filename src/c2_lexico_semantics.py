import spacy
import statistics
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


"""

TO DO:
- corpus freqs???
- x




    "lexico_semantics": {
        "vocabulary": {
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
    def __init__(self, spacy_model='en_core_web_sm', lm_model='gpt2', corpus_freqs=None, device=None):
        """
        spacy_model: spaCy model for parsing
        lm_model: Hugging Face causal LM for computing token log probabilities
        corpus_freqs: dict mapping words -> corpus frequency
        device: 'cuda' or 'cpu' (default: auto)
        """
        self.nlp = spacy.load(spacy_model)
        self.corpus_freqs = corpus_freqs or {}

        # Initialize LLM
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model)
        self.model = AutoModelForCausalLM.from_pretrained(lm_model).to(self.device)
        self.model.eval()

    # -------------------------------
    # 1. Vocabulary metrics
    # -------------------------------
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

    # -------------------------------
    # 2. Compute token log probabilities
    # -------------------------------
    def compute_log_probs(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits
            labels = inputs["input_ids"]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
            return token_log_probs[0].tolist()

    # -------------------------------
    # 3. Information content
    # -------------------------------
    def compute_information_content(self, text):
        log_probs = self.compute_log_probs(text)
        mean_surprisal = round(statistics.mean([-lp for lp in log_probs]), 3)
        surprisal_variance = round(statistics.variance([-lp for lp in log_probs]), 3) if len(log_probs) > 1 else 0
        return {
            "mean_surprisal": mean_surprisal,
            "surprisal_variance": surprisal_variance
        }

    # -------------------------------
    # 4. Semantic structures
    # -------------------------------
    def extract_semantic_structures(self, doc):
        structures = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    # Clause type (simple proxy)
                    clause_type = "main" if token.head == token else "subordinate"

                    # Subject (agent)
                    subject = [child.text for child in token.children if "subj" in child.dep_]
                    agent = " ".join(subject) if subject else None

                    # Object (patient)
                    obj = [child.text for child in token.children if "obj" in child.dep_]
                    patient = " ".join(obj) if obj else None

                    clause_tokens = [t.text for t in token.subtree]

                    structures.append({
                        "clause_level": clause_type,
                        "predicate": token.lemma_,
                        "agent": agent,
                        "patient": patient,
                        "clause_tokens": clause_tokens
                    })
        return structures