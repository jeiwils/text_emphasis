"""Preprocessing utilities for cleaning and segmenting source texts."""

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

try:
    import pdfplumber
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pdfplumber = None
from bs4 import BeautifulSoup

from .x_configs import (
    BOOK_CONFIGS,
    DEFAULT_BOOK_CONFIG,
    DEFAULT_SPACY_MODEL,
    DEFAULT_USE_EXISTING,
    GENRES,
    load_spacy_model,
)
from .z_utils import text_path

def _html_to_text(html: str, selector: Optional[str] = None) -> str:
    """
    Convert HTML to plain text. Optional CSS selector narrows to main content.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Drop non-content
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    root = soup.select_one(selector) if selector else (soup.body or soup)
    if root is None:
        root = soup

    text = root.get_text("\n")

    # Normalize common web whitespace artefacts BEFORE marker matching
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text


def scrape_web_text(
    url: str,
    start_marker: Optional[str] = None,
    end_marker: Optional[str] = None,
    patterns: Optional[List[str]] = None,
    selector: Optional[str] = None,
) -> str:
    """
    Fetch HTML, extract readable text, then slice/clean using markers + regex patterns.
    """
    request = Request(url, headers={"User-Agent": "text_emphasis/1.0 (local)"})
    with urlopen(request) as response:  # noqa: S310
        # Try to respect page encoding if provided
        charset = response.headers.get_content_charset() or "utf-8"
        html_content = response.read().decode(charset, errors="replace")

    raw_text = _html_to_text(html_content, selector=selector)

    return remove_boilerplate(
        raw_text,
        patterns=patterns,
        start_marker=start_marker,
        end_marker=end_marker,
    )

def _resolve_web_category(config: dict) -> str:
    """
    Map a web config to the same <genre>/<author> category layout used by PDFs.
    Falls back to the provided category or a "web/<author>" bucket.
    """
    author = config.get("author")
    genre = config.get("genre")
    if author and genre:
        return f"{genre}/{author}"
    if author:
        raw_root = text_path("raw")
        for candidate_genre in GENRES:
            candidate = raw_root / candidate_genre / author
            if candidate.exists():
                return f"{candidate_genre}/{author}"
    return config.get("category") or (f"web/{author}" if author else "web")


def preprocess_web_story(
    story_key: str,
    preproc: "TextPreprocessor",
    config: dict,
    use_existing: bool = DEFAULT_USE_EXISTING,
    category_override: Optional[str] = None,
):
    """
    Scrape -> clean -> save JSON + JSONL (cleaned + normalised), aligned with preprocess_pdf().
    """
    url = config["url"]
    category = category_override or _resolve_web_category(config)

    cleaned_dir = text_path("processed", "cleaned_texts", category)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = cleaned_dir / f"{story_key}_cleaned.json"

    cleaned_segmented_dir = text_path("processed", "cleaned_segmented_texts", category)
    cleaned_segmented_dir.mkdir(parents=True, exist_ok=True)
    cleaned_segmented_path = cleaned_segmented_dir / f"{story_key}_cleaned_segmented.jsonl"

    normalised_dir = text_path("processed", "normalised_texts", category)
    normalised_dir.mkdir(parents=True, exist_ok=True)
    normalised_path = normalised_dir / f"{story_key}_normalised.json"

    normalised_segmented_dir = text_path("processed", "normalised_segmented_texts", category)
    normalised_segmented_dir.mkdir(parents=True, exist_ok=True)
    normalised_segmented_path = normalised_segmented_dir / f"{story_key}_normalised_segmented.jsonl"

    if (
        use_existing
        and cleaned_path.exists()
        and cleaned_segmented_path.exists()
        and normalised_path.exists()
        and normalised_segmented_path.exists()
    ):
        print(f"[INFO] Skipping {story_key} (outputs exist)")
        return cleaned_path

    raw_text = scrape_web_text(
        url,
        start_marker=config.get("start_marker"),
        end_marker=config.get("end_marker"),
        patterns=config.get("patterns"),
        selector=config.get("selector"),
    )

    cleaned_text = preproc.clean_text(raw_text)
    cleaned_path.write_text(json.dumps({"text": cleaned_text}, ensure_ascii=False, indent=2), encoding="utf-8")

    cleaned_sentences = preproc.segment_sentences_with_offsets(cleaned_text)
    with open(cleaned_segmented_path, "w", encoding="utf-8") as f:
        for idx, sent in enumerate(cleaned_sentences):
            f.write(json.dumps({
                "sentence_id": idx,
                "text": sent["text"],
                "start_char": sent["start_char"],
                "end_char": sent["end_char"],
            }, ensure_ascii=False) + "\n")

    normalised_text, normalised_segmented_entries = preproc.normalize_sentences_with_offsets(
        cleaned_sentences
    )
    normalised_path.write_text(
        json.dumps({"text": normalised_text}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with open(normalised_segmented_path, "w", encoding="utf-8") as f:
        for entry in normalised_segmented_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[INFO] Web cleaned text saved to {cleaned_path}")
    return cleaned_path









class TextPreprocessor:
    def __init__(self, language: str = DEFAULT_SPACY_MODEL):
        """Initialize the preprocessor with specified language model."""
        self.nlp = load_spacy_model(language)

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
            # Remove interspersed numerals (often chapter numbers) and glue the word.
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

        def remove_inline_headers(value: str) -> str:
            """
            Strip embedded page headers like "254] THE MARQUISE OF O".
            """
            value = re.sub(r"\b\d{2,4}\]\s+[A-Z][A-Z\s]{2,}", " ", value)
            value = re.sub(r"\b\d{2,4}\]\b", " ", value)
            return value

        def dehyphenate_linebreaks(value: str) -> str:
            """
            Join words split across line breaks, e.g., "con-\nvent" -> "convent".
            """
            return re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", value)

        def dehyphenate_common_splits(value: str) -> str:
            """
            Merge obvious OCR line-break hyphenations while leaving real compounds.
            """
            prefixes = {
                "al", "ar", "be", "com", "con", "de", "dis", "en", "em", "ex",
                "in", "im", "inter", "mis", "non", "pre", "pro", "re", "sub",
                "trans", "un", "under", "over",
            }
            lowered = value.lower()
            candidates = set(re.findall(r"\b[a-z]{2,}-[a-z]{2,}\b", value))
            replacements = {}
            for token in candidates:
                left, right = token.split("-")
                merged = left + right
                if merged in lowered:
                    replacements[token] = merged
                elif left in prefixes and len(right) > 2:
                    replacements[token] = merged
            for token in sorted(replacements, key=len, reverse=True):
                value = re.sub(rf"\b{re.escape(token)}\b", replacements[token], value)
            return value

        def fix_digit_glue(value: str) -> str:
            """
            Remove OCR line-number glue like "Belfast1" or "4Letizia".
            Keep ordinals and common currency suffixes (e.g., 30th, 11s, 6d).
            """
            allowed_suffixes = {"st", "nd", "rd", "th", "s", "d"}

            def strip_trailing_digits(match: re.Match) -> str:
                word, digits = match.group(1), match.group(2)
                if len(digits) <= 2 and len(word) >= 2:
                    return word
                return match.group(0)

            def strip_leading_digits(match: re.Match) -> str:
                digits, word = match.group(1), match.group(2)
                if word.lower() in allowed_suffixes:
                    return match.group(0)
                if len(digits) <= 3 and len(word) >= 2:
                    return word
                return match.group(0)

            value = re.sub(r"\b([A-Za-z]{2,})(\d{1,3})\b", strip_trailing_digits, value)
            value = re.sub(r"\b(\d{1,3})([A-Za-z]{2,})\b", strip_leading_digits, value)
            return value

        text = fix_mojibake(text)
        text = despace_dropcaps(text)
        text = fix_letter_spacing_headers(text)
        text = normalize_shouting(text)
        text = fix_split_words(text)
        text = re.sub(r"\bL/n\b", "In", text)
        text = remove_inline_headers(text)
        text = dehyphenate_linebreaks(text)
        text = fix_digit_glue(text)
        text = text.replace("\xad", "")
        text = re.sub(r"(?<=\w)-\s+(?=\w)", "-", text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = dehyphenate_common_splits(text)
        # Common PDF artefacts
        text = re.sub(r"\(cid:\d+\)", "", text)   # removes (cid:20) etc
        text = text.replace("−", "-")             # normalize U+2212 to hyphen
        text = text.replace("\xa0", " ")          # NBSP -> space

        return text  # No lowercasing, no punctuation removal

    def segment_sentences_with_offsets(self, text: str) -> List[Dict[str, object]]:
        """
        Segment text into sentences with character offsets to avoid re-segmentation downstream.
        Returns a list of dicts: text, start_char, end_char.
        """
        doc = self.nlp(text)
        sentences = []
        for sent in doc.sents:
            sent_text = text[sent.start_char:sent.end_char] # or sent_text = sent.text.strip()???
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
        # Replace dash-like separators before ASCII folding to prevent word glue.
        text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", " ", text)
        # ASCII-fold to keep downstream normalization stable (e.g., "æ" -> "ae").
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
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

    def normalize_sentences_with_offsets(
        self,
        sentences: List[Dict[str, object]],
    ) -> Tuple[str, List[Dict[str, object]]]:
        """
        Normalize pre-segmented sentences while preserving sentence IDs and offsets.
        Returns the normalized text plus JSONL-ready entries aligned to the input order.
        """
        if not sentences:
            return "", []

        normalized_entries: List[Dict[str, object]] = []
        parts: List[str] = []
        cursor = 0
        last_idx = len(sentences) - 1

        for idx, sentence in enumerate(sentences):
            sentence_id = sentence.get("sentence_id", idx)
            raw_text = sentence.get("text", "") if isinstance(sentence, dict) else ""
            normalized_sentence = self.normalize_text(str(raw_text))

            start_char = cursor
            end_char = start_char + len(normalized_sentence)
            normalized_entries.append(
                {
                    "sentence_id": int(sentence_id),
                    "text": normalized_sentence,
                    "start_char": start_char,
                    "end_char": end_char,
                }
            )

            parts.append(normalized_sentence)
            cursor = end_char
            if idx < last_idx:
                parts.append(" ")
                cursor += 1

        normalized_text = "".join(parts)
        return normalized_text, normalized_entries

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens to their base form."""
        doc = self.nlp(' '.join(tokens))
        return [token.lemma_ for token in doc]
    

    def pdf_to_text(self, pdf_path: str) -> str:
        """Extract text from a text-based PDF."""
        if pdfplumber is None:
            raise RuntimeError("pdfplumber is required to extract PDF text.")
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    

def _normalize_book_key(name: str) -> str:
    """
    Normalize a filename stem to a config key: lowercase and collapse non-alnum to underscores.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def extract_pdf_pages(
    pdf_path: Path,
    pages: Optional[List[int]] = None,
    use_text_flow: bool = False,
    extract_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Extract text from specific PDF pages.
    If `pages` is None, extracts all pages.

    Note: page indices provided via `pages` are expected to be 1-based and are
    normalized to zero-based before iteration to align with pdfplumber's
    indexing.

    """
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required to extract PDF text.")
    text = ""
    processed_pages = 0
    extract_kwargs = extract_kwargs or {}

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
                page_text = page.extract_text(
                    use_text_flow=use_text_flow,
                    **extract_kwargs,
                )
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
    use_existing: bool = DEFAULT_USE_EXISTING,
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
            "Add an entry to BOOK_CONFIGS in src/x_configs.py to customize page ranges or patterns."
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
    extract_kwargs = active_config.get("extract_kwargs")
    raw_text = extract_pdf_pages(
        pdf_path,
        pages,
        use_text_flow=use_text_flow,
        extract_kwargs=extract_kwargs,
    )

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

    normalised_text, normalised_segmented_entries = preproc.normalize_sentences_with_offsets(
        cleaned_segmented_entries
    )
    normalised_path.write_text(
        json.dumps({"text": normalised_text}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with open(normalised_segmented_path, "w", encoding="utf-8") as f:
        for entry in normalised_segmented_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"[INFO] Cleaned text saved to {cleaned_path}")
    print(f"[INFO] Cleaned segmented text saved to {cleaned_segmented_path}")
    print(f"[INFO] Normalised text saved to {normalised_path}")
    print(f"[INFO] Normalised segmented text saved to {normalised_segmented_path}")
    return cleaned_path



def preprocess_all_pdfs(
    process_unknown: bool = True,
    use_existing: bool = DEFAULT_USE_EXISTING,
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
