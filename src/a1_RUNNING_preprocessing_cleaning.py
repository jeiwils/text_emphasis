from pathlib import Path
from .z_utils import processed_text_path, raw_text_path
from .a_preprocessing_cleaning import TextPreprocessor

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
