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

from x_configs import GENRES, load_spacy_model
from z_utils import text_path

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



WEB_CONFIGS = {
    "indian_uprising": {
        "author": "barthelme",
        "url": "https://xpressenglish.com/our-stories/indian-uprising/",
        # Short + robust markers (don’t cross newlines)
        "start_marker": "WE DEFENDED the city as best we could.",
        "end_marker": "paint, feathers, beads.",
        # Optional cleanup; markers already cut most boilerplate
        "patterns": [
            r"^\s*©\s*\d{4}.*xpressenglish\.com.*$",
        ],
        # Optional: narrow extraction (keeps nav junk down)
        "selector": "article",
        "category": "web/barthelme",
    },

    "all_at_one_point": {
        "author": "calvino",
        "url": "https://www.ruanyifeng.com/calvino/2007/07/ch_4_all_at_one_point.html",
        # Includes the “science epigraph” line (your choice); move to “Naturally, we were all there,” if you want epigraph removed
        "start_marker": "Through the calculations begun by Edwin P. Hubble",
        "end_marker": "mourning her loss.",
        "patterns": [
            r"^Posted on.*$",
        ],
        "selector": None,
        "category": "web/calvino",
    },

    "the_spiral": {
        "author": "calvino",
        "url": "https://www.ruanyifeng.com/calvino/2007/06/ch_12_the_spiral.html",
        "start_marker": "For the majority of mollusks, the visible organic form",
        "end_marker": "without shores, without boundaries.",
        "patterns": [
            r"^Posted on.*$",
        ],
        "selector": None,
        "category": "web/calvino",
    },

    "kleist_marquise_of_o": {
        "author": "kleist",
        "url": "https://archive.org/stream/in.ernet.dli.2015.225965/2015.225965.The-Marquise_djvu.txt",

        # Start at the first story sentence (skips the title/TOC/intro junk)
        "start_marker": "Liy  n M , a large  town  in",

        # End at the story’s final sentence (right before Michael Kohlhaas starts)
        "end_marker": "if  he  had  not  seemed  like  an  angel  to  her  at  his  first  appearance.",

        # Kill running headers + page-number-only lines that get injected mid-story
        "patterns": [
            # "THE MARQUISE OF O" / zero variant "0"
            r"^\s*THE\s+MARQUISE\s+OF\s+[O0]\s*$",

            # Running header with optional page marker:
            # e.g. "The  Marquise  of  O [61" or "The  Marquise  of  O-"
            r"^\s*The\s+Marquise\s+of\s+O-?(?:\s*\[\s*\d+(?:\s+\d+)*\s*\]?)?\s*$",

            # Page numbers that show up alone: "41", "[59", "60]", "25 1", etc.
            r"^\s*\[?\s*\d+(?:\s+\d+)*\s*\]?\s*$",
        ],

        "selector": None,
        "category": "web/kleist",
    },

    "kleist_earthquake_in_chile": {
        "author": "kleist",
        "url": "https://archive.org/stream/in.ernet.dli.2015.225965/2015.225965.The-Marquise_djvu.txt",

        # First narrative line of the Earthquake story
        "start_marker": "L/n  Santiago,  the  capital  of  the",

        # Last line before St. Cecilia begins
        "end_marker": "it  almost  seemed  to  him  that  he  had  reason  to  feel  glad.",

        "patterns": [
            # Archive scans inject the book header even inside other stories
            r"^\s*THE\s+MARQUISE\s+OF\s+[O0]\s*$",

            # Running header for this story, sometimes with page numbers attached:
            # e.g. "The  Earthquake  in  Chile  [267"
            r"^\s*The\s+Earthquake\s+in\s+Chile(?:\s*\[\s*\d+(?:\s+\d+)*\s*\]?)?\s*$",

            # Page numbers alone
            r"^\s*\[?\s*\d+(?:\s+\d+)*\s*\]?\s*$",
        ],

        "selector": None,
        "category": "web/kleist",
    },
}


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
    use_existing: bool = True,
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
    def __init__(self, language: str = "en_core_web_sm"):
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
        # Skip cover + copyright pages; story starts on PDF page 3
        "pages": list(range(3, 9)),  # 3-8
        "use_text_flow": True,
        "start_marker": "TRUE!",
        "end_marker": "hideous heart!",
        "patterns": [
            r"^\s*\d{1,3}\s*$",  # page numbers
            r"^\s*EDGAR\s+ALLAN\s+POE\s+\d+\s*$",
            r"^\s*\d+\s+THE\s+TELL-TALE\s+HEART\s*$",
            r"^\s*THE\s+TELL-TALE\s+HEART\s+\d+\s*$",
            r"^\s*E\s+d\s+g\s+a\s+r.*P\s+o\s+e.*$",
            r"^\s*p\s*$",
        ],
    },

    "the_cask_of_amontillado": {
        # Skip cover + copyright pages; story starts on PDF page 3
        "pages": list(range(3, 11)),  # 3-10
        "use_text_flow": True,
        "start_marker": "THE thousand injuries of Fortunato I had borne as I best",
        "end_marker": "In pace requiescat!",
        "patterns": [
            r"^\s*\d{1,3}\s*$",  # page numbers
            r"^\s*EDGAR\s+ALLAN\s+POE\s+\d+\s*$",
            r"^\s*\d+\s+THE\s+CASK\s+OF\s+AMONTILLADO\s*$",
            r"^\s*THE\s+CASK\s+OF\s+AMONTILLADO\s+\d+\s*$",
            r"^\s*E\s+d\s+g\s+a\s+r.*P\s+o\s+e.*$",
            r"^\s*p\s*$",
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
    # --- Kleist ---
    "saint_cecilia": {
        "pages": list(range(1, 9)),  # 1–8
        "use_text_flow": True,
        "start_marker": "At the end of the sixteenth century",
        "end_marker": "Gloria in excelsis yet again.",
        "patterns": [
            r"^\s*\d+\s*$",  # page numbers
        ],
    },

    # --- Hoffmann (Blackmask) ---
    # NOTE: filename is "counsillor_krespel.pdf" (typo), so the normalized key is "counsillor_krespel"
    "counsillor_krespel": {
        "pages": list(range(5, 15)),  # 5–14 (skip cover + TOC)
        "use_text_flow": False,
        "start_marker": "The man whom I am going to tell you about was Krespel",
        "end_marker": "But she was dead!",
        "patterns": [
            r"This page copyright.*",
            r"http://www\.blackmask\.com.*",
            r"^Councillor Krespel\s*$",
            r"^Councillor Krespel\s+\d+\s*$",
            r"^E\.T\.A\. Hoffmann\s*\d*\s*$",
            r"^Translation by.*$",
        ],
    },

    "the_sandman": {
        "pages": list(range(1, 18)),  # 1–17
        "use_text_flow": True,  # important for correct reading order
        "start_marker": "Certainly you must all be uneasy",
        "end_marker": "would never have given her.",
        "patterns": [
            r"^\s*Hoffmann\s+The Sandman\s+\d+\s*$",
            r"^Translation by.*$",
            r"^\s*\d+\s*$",
        ],
    },

    "automata": {
        "pages": list(range(3, 22)),  # 3–21 (skip cover + TOC)
        "use_text_flow": False,
        "start_marker": "A considerable time ago I was invited",
        # avoid newline breaks in the last quote
        "end_marker": "told, after all.",
        "patterns": [
            r"This page copyright.*",
            r"http://www\.blackmask\.com.*",
            r"^Automata\s*$",
            r"^Automata\s+\d+\s*$",
            r"^E\.?\s*T\.?\s*A\.?\s*Hoffmann\s*$",
            r"\(cid:\d+\)",
        ],
    },

    # --- Le Fanu (Blackmask) ---
    "mr_justice_harbottle": {
        "pages": list(range(3, 24)),  # 3-23
        "use_text_flow": False,
        "start_marker": "CHAPTER I. THE JUDGE'S HOUSE",
        "end_marker": "the rich man died, and was buried.",
        "patterns": [
            r"This page copyright.*",
            r"http://www\.blackmask\.com.*",
            r"^\s*MR\. JUSTICE HARBOTTLE\s*$",
            r"^\s*MR\. JUSTICE HARBOTTLE\s+\d+\s*$",
            r"^\s*\?\s+.*$",  # TOC bullets
            r"^\s*CHAPTER\s+[IVXLC]+\..*\s+\d+\s*$",
            r"\(cid:\d+\)",
        ],
    },

    "the_familiar": {
        "pages": list(range(4, 27)),  # 4-26 (skip cover + TOC + copyright page)
        "use_text_flow": False,
        "start_marker": "CHAPTER I. FOOTSTEPS",
        "end_marker": "absolute and impenetrable mystery is like to prevail until the day of doom.",
        "patterns": [
            r"This page copyright.*",
            r"http://www\.blackmask\.com.*",
            r"^\s*THE FAMILIAR\s*$",
            r"^\s*THE FAMILIAR\s+\d+\s*$",
            r"^\s*CHAPTER.*\s+\d+\s*$",
            r"^\s*POSTSCRIPT BY THE EDITOR\s+\d+\s*$",
            r"\(cid:\d+\)",
        ],
    },

    "green_tea": {
        "pages": list(range(3, 23)),  # 3-22 (skip cover/TOC/end page)
        "use_text_flow": False,
        "start_marker": "CHAPTER I. Dr. Hesselius Relates How He Met the Rev. Mr. Jennings",
        "end_marker": "and the mortal and immortal prematurely make acquaintance.",
        "patterns": [
            r"This page copyright.*",
            r"http://www\.blackmask\.com.*",
            r"^\s*Green Tea\s*$",
            r"^\s*Green Tea\s+\d+\s*$",
            r"^\s*\?\s+.*$",
            r"^\s*i\s*$",
        ],
    },

    # --- Mary Shelley (UFSC / Blackmask mix) ---
    "the_mortal_immortal": {
        "pages": list(range(2, 12)),  # 2–11 (skip cover)
        "use_text_flow": False,
        "start_marker": "JULY 16, 1833",
        "end_marker": "its immortal essence.",
        "patterns": [],
    },

    "the_transformation": {
        "pages": list(range(2, 15)),  # 2–14 (skip cover)
        "use_text_flow": False,
        "start_marker": "I HAVE heard it said",
        "end_marker": "Guido il Cortese.",
        "patterns": [],
    },

    "the_dream": {
        "pages": list(range(3, 11)),  # 3–10 (skip cover + TOC)
        "use_text_flow": False,
        "start_marker": "THE time of the occurrence of the little legend",
        "end_marker": "bid me be blest for evermore",
        "patterns": [
            r"This page copyright.*",
            r"http://www\.blackmask\.com.*",
            r"^The Dream\s*$",
            r"^The Dream\s+\d+\s*$",
            r"^Mary Shelley\s*$",
            r"^by The Author of Frankenstein\s*$",
        ],
    },

    # --- Kate Chopin ---
    "the_story_of_an_hour": {
        "pages": [1, 2],
        "use_text_flow": True,
        "start_marker": "Knowing that Mrs. Mallard was afflicted with a heart trouble,",
        "end_marker": "When the doctors came they said she had died of heart disease—of joy that kills.",
        "patterns": None,
    },

    "a_pair_of_silk_stockings": {
        "pages": [1, 2, 3],  # page 4 is notes only
        "use_text_flow": True,
        "start_marker": "Little Mrs. Sommers one day found herself the unexpected possessor of fifteen",
        "end_marker": "go on and on with her forever.",
        "patterns": [
            r"^\s*\d+\s*$",  # page numbers like "2", "3"
        ],
    },

    "desirees_baby": {
        "pages": [1, 2, 3, 4],
        "use_text_flow": True,
        "start_marker": "As the day was pleasant, Madame Valmondé drove over to L’Abri to see Désirée",
        "end_marker": "the brand of slavery.”",
        "patterns": None,
    },

    # --- Henry James (Blackmask-style PDFs: TOC/headers/footers) ---

    # NOTE: file name is the_real_thing.pdf but the actual text is "The Real Right Thing"
    "the_real_thing": {
        # Skip cover/TOC/copyright pages; story starts on PDF page 6
        "pages": list(range(6, 14)),  # 6–13
        "use_text_flow": True,
        "start_marker": "When, after the death of Ashton Doyne",
        "end_marker": "\"I give up.\"",
        "patterns": [
            r"^\s*The Real Right Thing\s*$",  # footer/header
            r"^\s*\d+\s+\d+\s*$",             # footer like "1 3", "2 6", etc.
            r"^\s*\d+\s*$",                   # section markers like "1", "2", "3" on their own line
        ],
    },

    "the_author_of_beltraffio": {
        # Skip cover + TOC; story begins on PDF page 3
        "pages": list(range(3, 29)),  # 3–28
        "use_text_flow": True,
        "start_marker": "Much as I wished to see him I had kept my letter of introduction",
        "end_marker": 'she even dipped into the black "Beltraffio."',
        "patterns": [
            r"^\s*This page copyright.*$",
            r"^\s*http://www\.blackmask\.com\s*$",
            r"^\s*The Author of Beltraffio\s*$",
            r"^\s*The Author of Beltraffio\s*\d+\s*$",  # footer like "The Author of Beltraffio 12"
            r"^\s*CHAPTER\s+[IVXLC]+\.?\s+\d+\s*$",     # footer like "CHAPTER II 9"
            r"^\s*•\s*$",
        ],
    },

    "the_figure_in_the_carpet": {
        # Skip cover + TOC; story content begins on PDF page 3 (after the copyright/transcription lines)
        "pages": list(range(3, 24)),  # 3–23
        "use_text_flow": True,
        "start_marker": "I had done a few things and earned a few pence",
        "end_marker": "quite my revenge.",
        "patterns": [
            r"^\s*This page copyright.*$",
            r"^\s*http://www\.blackmask\.com\s*$",
            r"^\s*Transcribed from.*$",
            r"^\s*The Figure in the Carpet\s*$",               # per-page footer/header
            r"^\s*CHAPTER\s+[IVXLC0-9]+\.?\s+\d+\s*$",         # footer like "CHAPTER XI. 21"
            r"^\s*CHAPTER\s+[IVXLC0-9]+\.?\s*•\s*$",           # "CHAPTER I •" lists
            r"^\s*•\s*$",
        ],
    },

    # --- Kafka ---
    "in_the_penal_colony": {
        "pages": list(range(1, 17)),  # 1–16
        "use_text_flow": True,
        "start_marker": "‘It’s a remarkable piece of apparatus,’ said the officer to the explorer",
        "end_marker": "kept them from attempting the leap.",
        "patterns": [
            r"^\s*©\s*\d{4}\s+by\s+http://www\.HorrorMasters\.com\s*$",
            r"^\s*\(c\)\s*\d{4}\s+by\s+Horror\s+Masters\s*$",
            r"^\s*Blah blah blah.*$",
            r"^\s*To the reader:.*stolen this story.*$",
            r"^~\^\^.*$",
            r"^@#\$.*$",
        ],
    },

    "the_judgement": {
        "pages": list(range(1, 8)),  # 1–7
        "use_text_flow": True,
        # Keeps "For F." and captures the split drop-cap ("I" + "t was...")
        "start_marker": "For F.\nI\nt was on a Sunday morning",
        "end_marker": "At this moment an almost endless traffic rolled across the bridge.",
        "patterns": [
            r'^\s*Franz\s+Kafka\s+"The\s+Judgement"\s+\d+\s*$',  # header on every page
            r"^_+\s*$",                                         # trailing separator line
        ],
    },

    "a_hunger_artist": {
        "pages": [1, 2, 3, 4, 5],
        "use_text_flow": True,

        # IMPORTANT: This PDF needs x_tolerance lowered to prevent missing spaces
        "extract_kwargs": {"x_tolerance": 1},

        "start_marker": "During these last decades the interest in professional fasting has markedly diminished.",
        "end_marker": "did not ever want to move away.",
        "patterns": [
            r"^\s*Franz\s+Kafka\s+\d+\s+A\s+Hunger\s+Artist\s*$",  # footer on every page
        ],
    },

    "kew_gardens": {
        "pages": list(range(1, 7)),  # 1–6
        "use_text_flow": False,
        "start_marker": "From the oval-shaped flower-bed",
        "end_marker": "voices cried aloud and the petals of myriads of flowers flashed their colours into the air.",
        "patterns": [
            r"Downloaded from\s+www\.libraryofshortstories\.com.*",
            r"This work is in the public domain.*",
            r"Please check your local copyright laws.*",
        ],
    },
    "the_mark_on_the_wall": {
        "pages": list(range(1, 10)),  # 1–9 (story); pages 10–13 are exercises
        "use_text_flow": True,
        "start_marker": "Perhaps it was the middle of January",
        "end_marker": "Ah, the mark on the wall! It was a snail",
        "patterns": [
            # Running headers
            r"^\s*\d{1,3}\s*/\s*(?:KALEIDOSCOPE|THE MARK ON THE WALL)\s*$",
            # Footer
            r"^\s*Reprint\s+\d{4}-\d{2}\s*$",

            # “Stop and Think” block injected mid-story (remove questions, keep story)
            r"Stop and Think(?:\s+Stop and Think)*\s*1\.[\s\S]*?(?=In certain lights)",
            r"Stop and Think(?:\s+Stop and Think)*\s*1\.[\s\S]*?(?=Someone is standing over me)",

            # Whitaker’s Almanack explanatory note (not Woolf’s story text)
            r"\*\s*Whitaker.?s Almanack[\s\S]{0,250}?subjects\.",
        ],
    },

    "an_unwritten_novel": {
        # PDF has 7 pages
        "pages": list(range(1, 8)),  # 1–7

        # IMPORTANT: fixes the initial drop-cap ordering ("S" + "UCH")
        "use_text_flow": True,

        # Start at title (easy to match), and cut before end notes
        "start_marker": "AN UNWRITTEN NOVEL",
        "end_marker": "adorable world!",

        # Optional: remove title line so cleaned text starts immediately with "Such ..."
        # (Your clean_text will repair "S UCH" -> "Such" either way, but this drops the title.)
        "patterns": [
            r"^\s*AN\s+UNWRITTEN\s+NOVEL\s*$",
        ],
    },

    "the_balloon": {
        # PDF has 6 pages
        "pages": list(range(1, 7)),  # 1–6

        # Default extraction order is fine for this PDF
        "use_text_flow": False,

        # Trim out title/header by starting at the first story sentence
        "start_marker": "The balloon, beginning at a point on Fourteenth Street",
        "end_marker": "when we are angry with one another.",

        # Remove repeating footer line
        "patterns": [
            r"^\s*xpressenglish\.com\s*$",
        ],
    },

    "the_school": {
        "pages": list(range(1, 5)),  # 1–4
        "use_text_flow": False,      # IMPORTANT for correct order in this PDF
        "start_marker": "Well, we had all these children out planting trees, see,",
        "end_marker": "The children cheered wildly.",
        "patterns": [
            # Running headers
            r"^\s*\d+\s+amateurs\s*$",
            r"^\s*the\s+school\s+\d+\s*$",

            # LOA subscribe boilerplate
            r"^\s*Are you receiving Story of the Week.*$",
            r"^\s*Sign up now at loa\.org/sotw.*$",

            # Footer garbage from "6966 Book.indb ..."
            r"^\s*\d+\s*BBooookk\.\.iinnddbb.*$",

            # Standalone page-number lines
            r"^\s*\d{1,4}\s*$",
        ],
    },

    "the_distance_of_the_moon": {
        "pages": list(range(1, 7)),  # 1–6 only
        "use_text_flow": True,       # REQUIRED for correct reading order
        "start_marker": "At one time, according to Sir George H. Darwin",
        "end_marker": "and me with them.",
        "patterns": [
            r"^\s*\d+\s*$",  # page numbers 1–6 at the top
        ],
    },
}


DEFAULT_BOOK_CONFIG = {
    "pages": None,
    "use_text_flow": False,
    "extract_kwargs": None,
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
