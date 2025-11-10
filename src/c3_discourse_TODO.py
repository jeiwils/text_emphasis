import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import json




"""

LOTS OF WORK NEEDS DOING ON THIS!!!
I THINK I NEED TO TRAIN A LM FOR EACH TEXT???







    "discourse": { # according to PDTB
        "relations": [
            {
            "level1": "<Temporal | Contingency | Comparison | Expansion>",
            "level2": "<Cause | Condition | Contrast | Concession | Conjunction | Instantiation | Restatement | Alternative | Exception | ...>",
            "level3": "<Reason | Result | Purpose | etc_or_null>",

            "arg1_span": "<text_span_or_reference>",
            "arg2_span": "<text_span_or_reference>",
            "connective": "<explicit_marker_or_null>",
            "explicit": "<true | false>",
            "direction": "<forward | backward | bidirectional>"
            }
        ]
    }





"""



class PDTBShallowParser:
    """
    A shallow discourse parser class for PDTB‑style relations: explicit + implicit.
    Pipeline: (1) connective detection, (2) arg1/arg2 extraction, (3) sense classification.
    """
    def __init__(self, backbone_model_name="bert-base-uncased", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_model_name)
        self.backbone = AutoModel.from_pretrained(backbone_model_name)
        self.backbone.to(self.device)
        
        # Example heads (you’ll need to define and fine‑tune in practice)
        hidden_size = self.backbone.config.hidden_size
        # Head for connective detection: binary token‑classification (connective or not)
        self.connective_head = nn.Linear(hidden_size, 2).to(self.device)
        # Head for span tagging: e.g., IOB tagging for arg1/arg2
        self.span_head = nn.Linear(hidden_size, 3).to(self.device)  # tags: O, ARG1, ARG2
        # Head for sense classification: multi‑class classification among (level1 × level2 × level3)
        num_senses = 50  # placeholder: set to size of your sense taxonomy
        self.sense_head = nn.Linear(hidden_size, num_senses).to(self.device)
        
        # Softmax/logits etc
        self.softmax = nn.Softmax(dim=-1)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def _encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)
        # outputs.last_hidden_state shape: (batch_size=1, seq_len, hidden_size)
        return inputs, outputs.last_hidden_state
    
    def detect_connectives(self, inputs, hidden_states):
        # Token‑level logits for connective detection
        logits = self.connective_head(hidden_states)  # (1, seq_len, 2)
        probs = self.softmax(logits)
        # Simple heuristic: select tokens where connective label prob > threshold
        threshold = 0.5
        connective_indices = (probs[...,1] > threshold).nonzero(as_tuple=False)
        # Return list of token indices labelled as connectives
        return connective_indices
    
    def extract_args(self, inputs, hidden_states, connective_indices):
        # Token‐level tagging for ARG1 / ARG2 / O
        logits = self.span_head(hidden_states)  # (1, seq_len, 3)
        tag_ids = torch.argmax(logits, dim=-1).squeeze(0)  # (seq_len,)
        # Map tag_ids to spans: here naive method, you’d want a proper span extractor
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
        arg1_tokens = [tokens[i] for i, tag in enumerate(tag_ids.tolist()) if tag == 1]
        arg2_tokens = [tokens[i] for i, tag in enumerate(tag_ids.tolist()) if tag == 2]
        arg1_span = self.tokenizer.convert_tokens_to_string(arg1_tokens)
        arg2_span = self.tokenizer.convert_tokens_to_string(arg2_tokens)
        return arg1_span, arg2_span
    
    def classify_sense(self, hidden_states, arg1_span, arg2_span):
        # For simplicity: use the [CLS] vector to classify sense
        cls_vector = hidden_states[:,0,:]  # (1, hidden_size)
        logits = self.sense_head(cls_vector)
        sense_id = torch.argmax(logits, dim=-1).item()
        # Map sense_id → (level1, level2, level3) via your taxonomy
        # For now return placeholder
        sense = {
            "level1": "Expansion",
            "level2": "Conjunction",
            "level3": None
        }
        return sense
    
    def parse(self, text):
        inputs, hidden_states = self._encode(text)
        conn_idxs = self.detect_connectives(inputs, hidden_states)
        # Choose first connective (naive)
        connective = None
        if len(conn_idxs) > 0:
            tok_idx = conn_idxs[0].item()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
            connective = tokens[tok_idx]
        arg1_span, arg2_span = self.extract_args(inputs, hidden_states, conn_idxs)
        sense = self.classify_sense(hidden_states, arg1_span, arg2_span)
        
        result = {
            "discourse": {
                "relations": [
                    {
                        "level1": sense["level1"],
                        "level2": sense["level2"],
                        "level3": sense["level3"],
                        "arg1_span": arg1_span,
                        "arg2_span": arg2_span,
                        "connective": connective,
                        "explicit": str(connective is not None).lower(),
                        # direction: heuristic — e.g., if arg1 before arg2 then “forward”
                        "direction": "forward"
                    }
                ]
            }
        }
        return result