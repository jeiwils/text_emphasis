from pathlib import Path
from .z_utils import processed_text_path, raw_text_path
from .a_preprocessing_cleaning import TextPreprocessor


"""

TO DO:
- move this to the a_preprocessing_cleaning module... just put it in the main

siddhartha: 1 - 53 (remove chapters + mid stuff)
the dead: 1 - 26 (remove titles + page numbers)

metamorphosis: 2 - 70 (remove chapters + website boilerplate)
the case of charles dexter ward: 3 - 96 (remove chapters + mid bits)



clockwork: 10 - 177
coraline: 11 - 199 (might need chapter numbers removing... roman numeral followed by full stop)
animal farm: 5 - 107 (remove CHAPTER 1 to CHAPTER 10 - there's some mid bits I need to remove as well)



american psycho: 6 - 457
handmaid's: 8 - 269 (need to remove chapters... CHAPTER + ONE to FORTY-SIX and also sections... roman numberals + word before CHAPTER + roman numerals)


"""



def preprocess_pdf(pdf_path: Path, preproc: TextPreprocessor):
    """
    Extract and clean a single PDF, then save cleaned text to cleaned_texts/{pdf_name}/
    """
    base_name = pdf_path.stem
    save_dir = processed_text_path("cleaned", base_name)
    
    # Skip if already processed
    if save_dir.exists():
        print(f"[SKIP] {pdf_path.name} already processed.")
        return None

    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract text
    raw_text = preproc.pdf_to_text(str(pdf_path))
    
    # Clean text
    cleaned_text = preproc.clean_text(raw_text)

    # Save cleaned text
    cleaned_path = save_dir / f"{base_name}_cleaned.txt"
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"[INFO] Cleaned text saved to {cleaned_path}")
    return cleaned_path


def preprocess_all_pdfs():
    """
    Iterate through all PDFs in raw_texts, process any not yet cleaned.
    """
    preproc = TextPreprocessor()
    raw_dir = raw_text_path()  # Path to TEXT_EMPHASIS/data/raw_texts
    pdf_files = list(raw_dir.glob("*.pdf"))

    if not pdf_files:
        print("[INFO] No PDFs found in raw_texts.")
        return

    for pdf_file in pdf_files:
        preprocess_pdf(pdf_file, preproc)


if __name__ == "__main__":
    preprocess_all_pdfs()
