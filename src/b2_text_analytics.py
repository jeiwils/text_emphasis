import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


"""

module for anything text-wide that needs processing before the sentence level analytics are done 



THIS WOULD OUTPUT TO CORPUS_ANALYTICS

"""

class TextPreprocessor:
    """
    Module for text preprocessing, including computing LLM token log-probs.
    
    Usage:
        preprocessor = TextPreprocessor(lm_model='gpt2', device='cuda')
        log_probs = preprocessor.compute_log_probs(text)
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
