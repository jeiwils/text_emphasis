import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
from .z_utils import processed_text_path
import os 
import json 
from x_configs import model

"""

module for anything text-wide that needs processing before the sentence level analytics are done 


TO DO:
- check how the chunking will work + how it will interface with window-level metrics


{
  "filename": "...",
  "model": "...",
  "avg_log_prob": float,
  "num_tokens": int,
  "top_words": [(word, freq), ...],
  "chunks": [
      {
        "chunk_id": int,
        "start_token": int,
        "end_token": int,
        "avg_log_prob": float,
        "num_tokens": int,
        "text_snippet": str
      }, ...
  ]
}





"""

class WholeTextMetrics:
    """
    Module for text preprocessing, including computing LLM token log-probs.
    

    """
    def __init__(self, lm_model=model, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(lm_model)
        self.model = AutoModelForCausalLM.from_pretrained(lm_model).to(self.device)
        self.model.eval()



    def compute_log_probs_chunked(self, text, chunk_size=2048, stride=0):
        if stride >= chunk_size:
            raise ValueError("stride must be smaller than chunk_size")
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return []
        results = []

        for i in range(0, len(tokens), chunk_size - stride):
            chunk_tokens = tokens[i : i + chunk_size]
            if len(chunk_tokens) < 2:
                continue
            inputs = torch.tensor([chunk_tokens]).to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                logits = outputs.logits[:, :-1]
                target_tokens = inputs[:, 1:]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)
                log_probs_list = [float(x) for x in token_log_probs[0].tolist()]

            if stride > 0 and i > 0:
                scored_log_probs = log_probs_list[stride:]
            else:
                scored_log_probs = log_probs_list
            if not scored_log_probs:
                continue

            avg_log_prob = sum(scored_log_probs) / len(scored_log_probs)
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            text_snippet = self.tokenizer.decode(chunk_tokens[:50], skip_special_tokens=True)  # first 50 tokens as preview

            results.append({
                "chunk_id": len(results),
                "start_token": i,
                "end_token": i + len(chunk_tokens),
                "avg_log_prob": avg_log_prob,
                "num_tokens": len(chunk_tokens),
                "num_scored_tokens": len(scored_log_probs),
                "text": chunk_text,
                "text_snippet": text_snippet,
                "log_probs": scored_log_probs,
            })

        return results


            


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
    






def run_whole_text_metrics(use_existing=True):
    """
    Runs WholeTextMetrics across all cleaned texts in:
    data/processed_texts/cleaned/[novels|novellas|short_stories]
    
    Outputs per-text results to:
    data/processed_texts/corpus/[novels|novellas|short_stories]
    """
    metrics = WholeTextMetrics()

    cleaned_root = processed_text_path("cleaned")
    output_root = processed_text_path("corpus")

    for subdir in cleaned_root.iterdir():
        if not subdir.is_dir():
            continue
        print(f"Processing category: {subdir.name}")

        out_subdir = output_root / subdir.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        for file in subdir.glob("*.txt"):
            output_file = out_subdir / f"{file.stem}_metrics.json"
            if use_existing and output_file.exists():
                print(f"Skipping {file.name} (exists)")
                continue

            text = file.read_text(encoding="utf-8")
            print(f"→ Computing metrics for {file.name}...")

            # Example: token log-probabilities
            log_prob_chunks = metrics.compute_log_probs_chunked(text, chunk_size=2048)

            avg_log_prob = None
            total_scored_tokens = sum(c.get("num_scored_tokens", 0) for c in log_prob_chunks)
            if log_prob_chunks and total_scored_tokens > 0:
                total_log_prob = sum(
                    c["avg_log_prob"] * c.get("num_scored_tokens", 0)
                    for c in log_prob_chunks
                )
                avg_log_prob = total_log_prob / total_scored_tokens

            # Example: corpus-level frequencies
            corpus_freq = metrics.compute_corpus_frequencies([text], min_freq=2)

            result = {
                "filename": file.name,
                "model": model,
                "avg_log_prob": avg_log_prob,
                "num_tokens": total_scored_tokens,
                "top_words": sorted(corpus_freq.items(), key=lambda x: x[1], reverse=True)[:50]
            }

            print(f"✅ Saved metrics to {output_file.name}")

    print("🎉 All done.")