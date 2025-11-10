"""

TO DO:
- get these functinons from other scripts - maybe get a standard preprocessing script that I use, upload to github



THIS WOULD OUTPUT TO CLEANED_TEXTS


"""

from typing import List, Optional
import spacy

import re
import pdfplumber

class TextPreprocessor:
    def __init__(self, language: str = "en_core_web_sm"):
        """Initialize the preprocessor with specified language model."""
        self.nlp = spacy.load(language)
    


    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        doc = self.nlp(text)
        return [token.text for token in doc]
    



    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Basic cleaning
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)  # normalize whitespace
        
        # Process with spaCy
        doc = self.nlp(text)
        # Remove punctuation and whitespace tokens
        cleaned = [token.text for token in doc 
                  if not token.is_punct and not token.is_space]
        return ' '.join(cleaned)
    



    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens to their base form."""
        doc = self.nlp(' '.join(tokens))
        return [token.lemma_ for token in doc]
    

    def pdf_to_text(self, pdf_path: str) -> str:
        """Extract text from a text-based PDF."""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
