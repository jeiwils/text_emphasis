"""Preprocessing utilities for cleaning and segmenting source texts."""

from pathlib import Path
from typing import List, Optional, Dict
import re
import json

import pdfplumber
from transformers import pipeline

from x_configs import GENRES, MODEL_CONFIGS, load_spacy_model
from z_utils import text_path


class TextPreprocessor:
    def __init__(self, language: str = "en_core_web_sm"):
        """Initialize the preprocessor with specified language model."""
        self.nlp = load_spacy_model(language)
        self._asr_pipeline = None
        self._asr_model_name = None
        self._asr_chunk_length_s = None
        self._asr_device = None

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words."""
        doc = self.nlp(text)
        return [token.text for token in doc]

    def clean_text(self, text: str) -> str:
        """Clean text while preserving punctuation and capitalization."""
        def fix_mojibake(value: str) -> str:
            """
            Repair common UTF-8/Windows-1252 artefacts that show up after PDF extraction.
            """
            # Known bad sequences
            replacements = {
                "â€™": "’",
                "â€˜": "‘",
                "â€œ": "“",
                "â€": "”",
                "â€“": "–",
                "â€”": "—",
                "Â": "",
                "ƒ?T": "’",
                "ƒ?o": "“",
                "ƒ??": "”",
                "ƒ?": "'",
            }
            fixed = value
            # Try latin-1 -> utf-8 roundtrip when any mojibake markers appear
            if any(marker in value for marker in replacements.keys()):
                try:
                    fixed = value.encode("latin-1").decode("utf-8")
                except UnicodeError:
                    fixed = value
            for bad, good in replacements.items():
                fixed = fixed.replace(bad, good)
            return fixed

        def despace_dropcaps(value: str) -> str:
            """
            Collapse drop-cap artefacts such as 'C ORALINE' or 'T HE' that break
            tokens apart after PDF extraction.
            """
            pattern = re.compile(r'(?:(?<=^)|(?<=[\n\r\.!\?]\s))([A-Z])\s+([A-Z][A-Za-z]+)')

            def _replace(match: re.Match) -> str:
                first, rest = match.group(1), match.group(2)
                merged = (first + rest.lower())
                return merged.capitalize()

            return pattern.sub(_replace, value)

        def fix_letter_spacing_headers(value: str) -> str:
            """
            Collapse spaced-out headings like 'C 1 HAPTER' or 'C HAPTER' and
            abbreviations like 'M r.' -> 'Mr.' that leak into tokens.
            """
            # Remove interspersed numerals (often OCR'd chapter numbers) and glue the word.
            value = re.sub(
                r'\b([A-Z])\s+(?:[0-9IVXLC]+\s+)?([A-Z][A-Za-z]+)\b',
                lambda m: f"{m.group(1)}{m.group(2).lower()}".capitalize(),
                value,
            )
            # Fix split abbreviations such as 'M r.' / 'D r.'.
            value = re.sub(r'\b([A-Z])\s+([a-z]\.)', r'\1\2', value)
            return value

        def normalize_shouting(value: str) -> str:
            """
            Downcase runs of all-caps words that likely came from small-caps PDF styling.
            Example: 'Coraline DISCOVERED THE DOOR' -> 'Coraline discovered the door'.
            """
            def replacer(match: re.Match) -> str:
                lead = match.group(1)
                caps_run = match.group(2)
                lowered = caps_run.lower()
                return f"{lead}{lowered}"

            pattern = re.compile(r'([A-Z][a-z]+)((?:\s+[A-Z]{2,}\b)+)')
            return pattern.sub(replacer, value)

        def fix_split_words(value: str) -> str:
            """
            Join common split-word artefacts like 'dis cover' -> 'discover'.
            Keep this list small to avoid unintended merges.
            """
            value = re.sub(r"\bdis\s+cover(ed|ing|s)?\b", r"discover\1", value, flags=re.IGNORECASE)
            return value

        text = fix_mojibake(text)
        text = despace_dropcaps(text)
        text = fix_letter_spacing_headers(text)
        text = normalize_shouting(text)
        text = fix_split_words(text)
        text = text.replace("\xad", "")
        text = re.sub(r"(?<=\w)-\s+(?=\w)", "-", text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text  # No lowercasing, no punctuation removal

    def segment_sentences_with_offsets(self, text: str) -> List[Dict[str, object]]:
        """
        Segment text into sentences with character offsets to avoid re-segmentation downstream.
        Returns a list of dicts: text, start_char, end_char.
        """
        doc = self.nlp(text)
        sentences = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            sentences.append(
                {
                    "text": sent_text,
                    "start_char": sent.start_char,
                    "end_char": sent.end_char,
                }
            )
        return sentences

    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentence strings (compat wrapper)."""
        return [item["text"] for item in self.segment_sentences_with_offsets(text)]

    def normalize_text(self, text: str) -> str:
        """Normalize text for embedding/topic workflows (lowercase lemmas, no punctuation)."""
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            lemma = (token.lemma_ or token.text).lower().strip()
            if not any(ch.isalnum() for ch in lemma):
                continue
            tokens.append(lemma)
        return " ".join(tokens)

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens to their base form."""
        doc = self.nlp(' '.join(tokens))
        return [token.lemma_ for token in doc]
    

    def transcribe_audio(
        self,
        audio_path: str,
        model_name: str = MODEL_CONFIGS["asr"],
        chunk_length_s: int = 30,
        device: Optional[int] = None,
    ) -> str:
        """
        Transcribe spoken audio to text using a Whisper ASR model.

        Audio can be any format supported by ffmpeg. Requires the model to be
        available locally (or network access for first-time download).
        """
        needs_new_pipeline = (
            self._asr_pipeline is None
            or self._asr_model_name != model_name
            or self._asr_chunk_length_s != chunk_length_s
            or self._asr_device != device
        )

        if needs_new_pipeline:
            try:
                self._asr_pipeline = pipeline(
                    task="automatic-speech-recognition",
                    model=model_name,
                    chunk_length_s=chunk_length_s,
                    device=device,
                )
                self._asr_model_name = model_name
                self._asr_chunk_length_s = chunk_length_s
                self._asr_device = device
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Failed to load ASR model '{model_name}'. Ensure it is installed locally "
                    "or downloadable in your environment."
                ) from exc

        result = self._asr_pipeline(audio_path)
        transcript = result["text"]
        return self.clean_text(transcript)


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
    # --- Joyce (Dubliners PDFs) ---

    "araby": {
        "pages": list(range(1, 6)),  # 1–5
        "use_text_flow": True,
        "start_marker": "North Richmond Street, being blind,",
        "end_marker": "my eyes burned with anguish and anger.",
        "patterns": [
            r"^\s*\d{1,3}\s*$",  # page numbers
        ],
    },

    "eveline": {
        "pages": list(range(1, 4)),  # 1–3
        "use_text_flow": True,       # REQUIRED for this PDF to extract in the right order
        "start_marker": "She sat at the window watching the evening invade the avenue.",
        # The sentence is line-broken in the PDF text layer, so use a marker that doesn't cross a newline:
        "end_marker": "or farewell or recognition.",
        "patterns": [
            r"^\s*\d{1,3}\s*$",  # defensive: page numbers / stray numeric lines
        ],
    },

    "the_dead": {
        "pages": list(range(1, 27)),  # 1–26
        "use_text_flow": True,
        "start_marker": "Lily, the caretaker's daughter",
        "end_marker": "the living and the dead.",
        "patterns": [
            r"^\s*\d{1,3}\s*$",  # page numbers
        ],
    },

    # --- Hawthorne ---

    "young_goodman_brown": {
        "pages": list(range(1, 11)),  # 1–10
        "use_text_flow": True,
        "start_marker": "YOUNG GOODMAN BROWN came forth at sunset",
        "end_marker": "hour was gloom.",
        "patterns": None,
    },

    "the_ministers_black_veil": {
        "pages": list(range(1, 8)),  # 1–7
        "use_text_flow": True,       # REQUIRED for this PDF (default extraction is badly ordered)
        "start_marker": "THE SEXTON stood in the porch of Milford meeting-house,",
        "end_marker": "he hid his face from men.",
        "patterns": [
            # Running headers like "3 NATHANIEL HAWTHORNE"
            r"^\s*\d+\s+NATHANIEL\s+HAWTHORNE\s*$",
        ],
    },

    "rappaccinis_daughter": {
        "pages": list(range(1, 21)),  # 1–20
        "use_text_flow": True,
        "start_marker": "A YOUNG man, named Giovanni Guasconti,",
        "end_marker": "upshot of your experiment?\"",
        "patterns": None,
    },

    # --- Maupassant ---

    "boule_de_suif": {
        "pages": list(range(1, 36)),  # 1–35
        "use_text_flow": True,
        "start_marker": "For several days in succession",
        "end_marker": "between two verses of the song.",
        "patterns": [
            # (mostly redundant because end_marker trims before these, but safe)
            r"Downloaded from\s+www\.libraryofshortstories\.com",
            r"This work is in the public domain.*",
        ],
    },

    "a_piece_of_string": {
        "pages": list(range(1, 8)),  # 1–7
        "use_text_flow": True,
        "start_marker": "ALONG ALL THE ROADS around Goderville",
        "end_marker": "M'sieu the Mayor.\"",
        "patterns": None,
    },

    "the_necklace": {
        "pages": list(range(1, 7)),  # 1–6
        "use_text_flow": True,
        "start_marker": "She was one of those pretty and charming girls",
        "end_marker": "francs!",
        "patterns": [
            r"^\s*\d{1,3}\s*$",  # defensive: standalone page numbers if present
        ],
    },

    # --- Borges ---

    "the_garden_of_forking_paths": {
        "pages": list(range(1, 7)),  # 1–6
        "use_text_flow": True,
        "start_marker": "On page 22 of Liddell Hart’s History of World War I",
        "end_marker": "contrition and weariness.",
        "patterns": [
            # Defensive cleanup in case end_marker fails:
            r"^\s*\[\d+\]\s*$",     # footnote marker lines like "[1]"
            r"\(Editor.?s note\.\)", # "(Editor’s note.)" / "(Editor's note.)"
            r"^\s*Response\s*$",     # trailing "Response" line
        ],
    },

    # NOTE: filename is Borges-The-Library-of-Babel.pdf => key borges_the_library_of_babel
    "borges_the_library_of_babel": {
        # Exclude page 8 (publisher / collection page)
        "pages": list(range(1, 8)),  # 1–7
        "use_text_flow": True,
        "start_marker": "The universe (which others call the Library)",
        "end_marker": "have no \"back.\"",
        "patterns": [
            # Running headers / page artifacts
            r"^\s*THE\s+LIBRARY\s+OF\s+BABEL\s+\d+\s*$",
            r"^\s*[0-9A-Za-z]*\s*JORGE\s+LUIS\s+BORGES\s*$",
            r"^\s*\d{1,3}\s*$",
            # Editorial note inside the text layer
            r"\[Ed\. note\.\]",
        ],
    },

    "the_aleph": {
        "pages": list(range(1, 12)),  # 1–11
        "use_text_flow": True,
        "start_marker": "On the burning February morning Beatriz Viterbo died,",
        "end_marker": "the face of Beatriz.",
        "patterns": None,
    },

    # --- Poe (updated PDFs) ---

    "the_telltale_heart": {
        "pages": list(range(1, 5)),  # 1–4
        "use_text_flow": True,
        "start_marker": "iT’s TRue! yes, i have been ill,",
        "end_marker": "Why does it not stop!?”",
        "patterns": [
            r"^\s*\d{1,3}\s*$",                  # page numbers 64–67
            r"^\s*E\s+d\s+g\s+a\s+r.*P\s+o\s+e.*$",  # spaced author header
            r"^\s*p\s*$",                        # the stray 'p' line
        ],
    },

    "the_cask_of_amontillado": {
        "pages": list(range(1, 6)),  # 1–5
        "use_text_flow": True,
        "start_marker": "foRTunaTo had huRT me a",
        "end_marker": "May he rest in peace!",
        "patterns": [
            r"^\s*\d{1,3}\s*$",                  # page numbers 68–72
            r"^\s*E\s+d\s+g\s+a\s+r.*P\s+o\s+e.*$",  # spaced author header
            r"^\s*p\s*$",                        # the stray 'p' line
        ],
    },

    "the_fall_of_the_house_of_usher": {
        # Skip the cover + copyright pages; story starts on PDF page 3
        "pages": list(range(3, 26)),  # 3–25
        "use_text_flow": True,
        "start_marker": "During the whole of a dull, dark, and soundless day in the",
        "end_marker": "the “House of Usher.”",
        "patterns": [
            # Running headers
            r"^\s*THE\s+FALL\s+OF\s+THE\s+HOUSE\s+OF\s+USHER\s+\d+\s*$",
            r"^\s*EDGAR\s+ALLAN\s+POE\s+\d+\s*$",
            r"^\s*\d{1,3}\s*$",  # stray numeric-only lines
            # Footnote block (cross-line match)
            r"\*\s*Watson[\s\S]{0,200}?vol\.\s*v\.",
        ],
    },
}



DEFAULT_BOOK_CONFIG = {
    "pages": None,
    "start_marker": None,
    "end_marker": None,
    "patterns": None,
}



def _normalize_book_key(name: str) -> str:
    """
    Normalize a filename stem to a config key: lowercase and collapse non-alnum to underscores.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def extract_pdf_pages(
    pdf_path: Path,
    pages: Optional[List[int]] = None,
    use_text_flow: bool = False,
) -> str:
    """
    Extract text from specific PDF pages.
    If `pages` is None, extracts all pages.

    Note: page indices provided via `pages` are expected to be 1-based and are
    normalized to zero-based before iteration to align with pdfplumber's
    indexing.
    """
    text = ""
    processed_pages = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        if pages is not None:
            invalid_pages = [p for p in pages if not isinstance(p, int) or p < 1]
            if invalid_pages:
                raise ValueError(
                    f"Page numbers must be positive integers (1-based). Invalid: {invalid_pages}"
                )
        normalized_indices = (
            range(total_pages) if pages is None else [page - 1 for page in pages]
        )

        for idx in normalized_indices:
            try:
                page = pdf.pages[idx]
                page_text = page.extract_text(use_text_flow=use_text_flow)
                if page_text:
                    text += page_text + "\n"
                processed_pages += 1
            except IndexError:
                human_page = idx + 1
                print(f"[WARN] Page {human_page} not found in {pdf_path.name}")

    if pages is None and processed_pages != total_pages:
        raise AssertionError(
            f"Expected to process {total_pages} pages from {pdf_path.name}, "
            f"but processed {processed_pages}."
        )

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
    config: Optional[dict] = None,
    book_name: Optional[str] = None,
    allow_default_config: bool = True,
    use_existing: bool = True,
    category_override: Optional[str] = None,
):
    """Extract, clean, and save a single PDF with optional page and boilerplate filtering."""
    base_name = pdf_path.stem
    book_label = book_name or base_name
    category = category_override or pdf_path.parent.name
    if config is None:
        active_config = DEFAULT_BOOK_CONFIG if allow_default_config else None
    else:
        active_config = config

    if active_config is None:
        print(f"[WARN] No config found for {book_label}, skipping because default processing is disabled.")
        return None

    if config is None:
        print(
            f"[WARN] No config found for '{book_label}'. Using default processing (all pages, no boilerplate removal). "
            "Add an entry to BOOK_CONFIGS in src/a_preprocessing_cleaning.py to customize page ranges or patterns."
        )

    cleaned_dir = text_path("processed", "cleaned_texts", category)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = cleaned_dir / f"{base_name}_cleaned.json"

    cleaned_segmented_dir = text_path("processed", "cleaned_segmented_texts", category)
    cleaned_segmented_dir.mkdir(parents=True, exist_ok=True)
    cleaned_segmented_path = cleaned_segmented_dir / f"{base_name}_cleaned_segmented.jsonl"

    normalised_dir = text_path("processed", "normalised_texts", category)
    normalised_dir.mkdir(parents=True, exist_ok=True)
    normalised_path = normalised_dir / f"{base_name}_normalised.json"

    normalised_segmented_dir = text_path("processed", "normalised_segmented_texts", category)
    normalised_segmented_dir.mkdir(parents=True, exist_ok=True)
    normalised_segmented_path = normalised_segmented_dir / f"{base_name}_normalised_segmented.jsonl"

    if (
        use_existing
        and cleaned_path.exists()
        and cleaned_segmented_path.exists()
        and normalised_path.exists()
        and normalised_segmented_path.exists()
    ):
        print(f"[INFO] Skipping {base_name} (outputs exist)")
        return cleaned_path

    # Extract selected pages
    pages = active_config.get("pages")
    use_text_flow = bool(active_config.get("use_text_flow", False))
    raw_text = extract_pdf_pages(pdf_path, pages, use_text_flow=use_text_flow)

    # Remove boilerplate, trim start/end markers, apply regex patterns
    cleaned_text = remove_boilerplate(
        raw_text,
        patterns=active_config.get("patterns"),
        start_marker=active_config.get("start_marker"),
        end_marker=active_config.get("end_marker")
    )

    # Normalize whitespace only
    cleaned_text = preproc.clean_text(cleaned_text)

    cleaned_path.write_text(json.dumps({"text": cleaned_text}, ensure_ascii=False, indent=2), encoding="utf-8")

    cleaned_sentences = preproc.segment_sentences_with_offsets(cleaned_text)
    cleaned_segmented_entries: List[Dict[str, object]] = []
    for idx, sentence in enumerate(cleaned_sentences):
        cleaned_segmented_entries.append(
            {
                "sentence_id": idx,
                "text": sentence["text"],
                "start_char": sentence["start_char"],
                "end_char": sentence["end_char"],
            }
        )
    with open(cleaned_segmented_path, "w", encoding="utf-8") as f:
        for entry in cleaned_segmented_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    normalised_text = preproc.normalize_text(cleaned_text)
    normalised_path.write_text(
        json.dumps({"text": normalised_text}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    normalised_sentences = [
        {
            "sentence_id": idx,
            "text": preproc.normalize_text(sentence["text"]),
            "start_char": sentence["start_char"],
            "end_char": sentence["end_char"],
        }
        for idx, sentence in enumerate(cleaned_sentences)
    ]
    normalised_segmented_entries: List[Dict[str, object]] = []
    for sentence in normalised_sentences:
        if not sentence["text"]:
            continue
        normalised_segmented_entries.append(
            {
                "sentence_id": sentence["sentence_id"],
                "text": sentence["text"],
                "start_char": sentence["start_char"],
                "end_char": sentence["end_char"],
            }
        )
    with open(normalised_segmented_path, "w", encoding="utf-8") as f:
        for entry in normalised_segmented_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[INFO] Cleaned text saved to {cleaned_path}")
    print(f"[INFO] Cleaned segmented text saved to {cleaned_segmented_path}")
    print(f"[INFO] Normalised text saved to {normalised_path}")
    print(f"[INFO] Normalised segmented text saved to {normalised_segmented_path}")
    return cleaned_path



def preprocess_audio_file(
    audio_path: Path,
    preproc: "TextPreprocessor",
    model_name: str = "openai/whisper-small", ### CONFIGS SHOULD BE INTEGRATED HERE
    chunk_length_s: int = 30,
    device: Optional[int] = None,
    save: bool = True,
    category: Optional[str] = None,
):
    """
    Transcribe and clean an audio file; optionally save the transcript.

    Returns the cleaned transcript path when saving, otherwise the text.
    """
    transcript = preproc.transcribe_audio(
        str(audio_path),
        model_name=model_name,
        chunk_length_s=chunk_length_s,
        device=device,
    )

    if not save:
        return transcript

    category_name = category or audio_path.parent.name
    save_dir = text_path("processed", "audio_transcripts", category_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = save_dir / f"{audio_path.stem}_transcript.txt"
    transcript_path.write_text(transcript, encoding="utf-8")
    print(f"[INFO] Transcript saved to {transcript_path}")
    return transcript_path



def preprocess_all_pdfs(
    process_unknown: bool = True,
    use_existing: bool = True,
    authors: Optional[List[str]] = None,
):
    preproc = TextPreprocessor()
    base_raw_dir = text_path("raw")
    for genre in GENRES:
        genre_dir = base_raw_dir / genre
        if not genre_dir.exists():
            print(f"[WARN] Directory not found: {genre_dir}")
            continue
        if authors:
            author_dirs = [genre_dir / author for author in authors]
        else:
            author_dirs = [path for path in genre_dir.iterdir() if path.is_dir()]

        for author_dir in author_dirs:
            if not author_dir.exists():
                print(f"[WARN] Directory not found: {author_dir}")
                continue
            pdf_files = list(author_dir.glob("*.pdf"))
            if not pdf_files:
                print(f"[INFO] No PDFs found in {author_dir}")
                continue

            category_key = f"{genre}/{author_dir.name}"
            print(f"[INFO] Processing {len(pdf_files)} PDFs in {category_key}...")

            for pdf_file in pdf_files:
                normalized_name = _normalize_book_key(pdf_file.stem)
                config = BOOK_CONFIGS.get(normalized_name)
                preprocess_pdf(
                    pdf_file,
                    preproc,
                    config=config,
                    book_name=normalized_name,
                    allow_default_config=process_unknown,
                    use_existing=use_existing,
                    category_override=category_key,
                )





if __name__ == "__main__":
    preprocess_all_pdfs()
