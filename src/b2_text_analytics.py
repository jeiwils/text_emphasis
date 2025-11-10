import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter

"""

module for anything text-wide that needs processing before the sentence level analytics are done 


TO DO:
- CHECK IF ANYTHING ELSE NEEDS TO GO HERE




"""

class WholeTextMetrics:
    """
    Module for text preprocessing, including computing LLM token log-probs.
    

    """
    def __init__(self, lm_model='gpt2', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model)
        self.model = AutoModelForCausalLM.from_pretrained(lm_model).to(self.device)
        self.model.eval()



    def compute_log_probs(self, text):
        """
        Computes token-level log-probabilities for a given text.
        Returns a list of log-probs (floats), one per token.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits
            labels = inputs["input_ids"]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
            return token_log_probs[0].tolist()
        


    def compute_corpus_frequencies(self, texts, lowercase=True, min_freq=1):
        """
        Computes corpus-level word frequencies from a list of texts.
        
        Args:
            texts (list of str): List of documents or corpus segments.
            lowercase (bool): Whether to lowercase words.
            min_freq (int): Minimum frequency to include in the output.
        
        Returns:
            dict: {word: frequency} for all words in the corpus.
        """
        word_counter = Counter()
        for text in texts:
            words = [w.lower() if lowercase else w 
                     for w in text.split() if w.isalpha()]
            word_counter.update(words)
        
        # Apply minimum frequency threshold
        corpus_freqs = {w: freq for w, freq in word_counter.items() if freq >= min_freq}
        return corpus_freqs