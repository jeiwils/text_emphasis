


"""

TO DO:
- get these functinons from other scripts - maybe get a standard preprocessing script that I use, upload to github




"""

from pathlib import Path
from .z_utils import processed_text_path, raw_text_path
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
        """Clean text while preserving punctuation and capitalization."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text  # No lowercasing, no punctuation removal

    



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
    





BOOK_CONFIGS = {
    # "siddhartha": {
    #     "pages": list(range(1, 54)),
    #     "start_marker": "In the shade of the house",
    #     "end_marker": "****",  # Anything after this on page 53 will be removed
    #     "patterns": [
    #         r"Part\s+(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)",
    #         r"\n\s*[A-Z][A-Za-z\s]{1,40}\s*\n",  # Likely detects centralized headers/titles
    #     ],
    # },

    "the_dead": {
        "pages": list(range(1, 27)),
        "start_marker": "Lily, the caretaker's daughter",
        "end_marker": None,
        "patterns": [
            r"^\s*\d{1,3}\s*$",  # Remove numeric page numbers
        ],
    },

    "metamorphosis": {
        "pages": list(range(2, 71)),
        "start_marker": "One morning, when Gregor Samsa woke",
        "end_marker": "stretch out her young body.",
        "patterns": [
            r"E-BooksDirectory\.com",
            r"\b[IVXLC]+\b(?!\w)",  # Roman numerals for chapters
        ],
    },

    "the_case_of_charles_dexter_ward": {
        "pages": list(range(3, 97)),
        "start_marker": "From a private hospital for the insane near Providence,",
        "end_marker": "thin coating of fine bluish-grey dust.",
        "patterns": [
            r"chapter\s+\w+",
            r"\bCHAPTER\s+[IVXLC]+\b",
            r"page\s+\d+",
            r"PART\s+[IVXLC]+\s*.*?(?=CHAPTER)",  # PART I ... CHAPTER
        ],
    },

    "a_clockwork_orange": {
        "pages": list(range(10, 178)),
        "start_marker": None,
        "end_marker": None,
        "patterns": [
            r"PART\s+(ONE|TWO|THREE|FOUR)",
            r"chapter\s+\w+",
            r"(?m)^\s*[IVXLC]+\s*$",  # Roman numerals on their own line
        ],
    },

    "coraline": {
        "pages": list(range(11, 200)),
        "start_marker": None,
        "end_marker": None,
        "patterns": [
            r"(?m)^\s*[IVXLC]+\.\s*$",  # Roman numerals with period on their own line
        ],
    },

    "animal_farm": {
        "pages": list(range(5, 108)),
        "start_marker": None,
        "end_marker": "was impossible to say which was which.",
        "patterns": [
            r"\bCHAPTER\s+[IVXLC]+\b",
            r"page\s+\d+",
            r"Animal Farm, by George Orwell",
            r"https://ebooks\.adelaide\.edu\.au/o/orwell/george/o79a/chapter\d+\.html",
            r"Last updated\s+[A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2},\s+\d{4},\s+at\s+\d{1,2}:\d{2}",
        ],
    },

    "american_psycho": {
        "pages": list(range(6, 458)),
        "start_marker": None,
        "end_marker": None,
        "patterns": [
            r"(?m)^[A-Z][a-zA-Z\s']{1,40}$",  # Matches chapter headings like 'Morning'
        ],
    },

    "the_handmaids_tale": {
        "pages": list(range(8, 270)),
        "start_marker": None,
        "end_marker": None,
        "patterns": [
            r"(?m)^\s*[IVXLC]+\s*\n[A-Z\s]{2,50}(?=\n)",  # Roman numeral + caps section name
        ],
    },
}






def extract_pdf_pages(pdf_path: Path, pages: Optional[List[int]] = None) -> str:
    """
    Extract text from specific PDF pages.
    If `pages` is None, extracts all pages.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        if pages is None:
            pages = range(len(pdf.pages))
        for i in pages:
            try:
                page = pdf.pages[i - 1]  # convert to zero-indexed
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except IndexError:
                print(f"[WARN] Page {i} not found in {pdf_path.name}")

    return text



def remove_boilerplate(text: str, patterns: Optional[List[str]] = None,
                       start_marker: Optional[str] = None,
                       end_marker: Optional[str] = None) -> str:
    """Remove boilerplate and trim text to start/end markers."""
    # Trim start
    if start_marker:
        start_idx = text.find(start_marker)
        if start_idx != -1:
            text = text[start_idx:]
    # Trim end
    if end_marker:
        end_idx = text.find(end_marker)
        if end_idx != -1:
            text = text[:end_idx + len(end_marker)]

    # Apply regex patterns
    if patterns:
        for pattern in patterns:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.MULTILINE)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text




def preprocess_pdf(
    pdf_path: Path,
    preproc: "TextPreprocessor",
    config: dict,
):
    """Extract, clean, and save a single PDF with optional page and boilerplate filtering."""
    base_name = pdf_path.stem
    save_dir = processed_text_path("cleaned", base_name)

    if save_dir.exists():
        print(f"[SKIP] {pdf_path.name} already processed.")
        return None

    save_dir.mkdir(parents=True, exist_ok=True)

    # Extract selected pages
    pages = config.get("pages")
    raw_text = extract_pdf_pages(pdf_path, pages)

    # Remove boilerplate, trim start/end markers, apply regex patterns
    cleaned_text = remove_boilerplate(
        raw_text,
        patterns=config.get("patterns"),
        start_marker=config.get("start_marker"),
        end_marker=config.get("end_marker")
    )

    # Normalize whitespace only
    cleaned_text = preproc.clean_text(cleaned_text)

    # Save
    cleaned_path = save_dir / f"{base_name}_cleaned.txt"
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"[INFO] Cleaned text saved to {cleaned_path}")
    return cleaned_path




def preprocess_all_pdfs():
    preproc = TextPreprocessor()
    base_raw_dir = raw_text_path()

    subdirs = ["novels", "novellas", "short_stories"]

    for subdir in subdirs:
        subdir_path = base_raw_dir / subdir
        if not subdir_path.exists():
            print(f"[WARN] Directory not found: {subdir_path}")
            continue

        pdf_files = list(subdir_path.glob("*.pdf"))
        if not pdf_files:
            print(f"[INFO] No PDFs found in {subdir_path}")
            continue

        print(f"[INFO] Processing {len(pdf_files)} PDFs in {subdir}...")

        for pdf_file in pdf_files:
            book_name = pdf_file.stem.lower()
            config = BOOK_CONFIGS.get(book_name)
            if config is None:
                print(f"[WARN] No config found for {book_name}, skipping.")
                continue

            preprocess_pdf(
                pdf_file,
                preproc,
                config=config
            )





if __name__ == "__main__":
    preprocess_all_pdfs()
