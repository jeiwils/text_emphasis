from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import spacy

LOOP_BLOCK_SIZES_ENABLED = False
LOOP_BLOCK_SIZES = (3, 5, 7)
DEFAULT_BLOCK_SIZE = 5
DEFAULT_DASHBOARD_PERMUTATIONS: int = 1000
DEFAULT_RNG_SEED: int = 42

USE_EXISTING = True

MODEL_CONFIGS = {
    "causal_lm": "gpt2",
    "sentence_embedding": "all-MiniLM-L6-v2",
}

# Default spaCy pipeline configuration
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_SPACY_DISABLE: Sequence[str] = ()
# Shared window size (in sentences) for sliding window metrics
DEFAULT_WINDOW_SIZE: int = 3
# Default soft topic-score filtering for topic modelling + dashboard
DEFAULT_SOFT_SCORE_THRESHOLD: float = 0.3 #0.5
DEFAULT_SOFT_TOP_K: int = 5 #3
# Default stride (in sentences) for sliding window metrics
DEFAULT_METRIC_WINDOW_STRIDE: int = 1
# Topic model windows use base_window_size * multiple
DEFAULT_TOPIC_WINDOW_MULTIPLE: int = 5
# Topic window stride uses base_window_size * stride_multiple
DEFAULT_TOPIC_WINDOW_STRIDE_MULTIPLE: int = 2
# Pipeline topic stride override uses base_window_size * stride_multiple
DEFAULT_PIPELINE_TOPIC_WINDOW_STRIDE_MULTIPLE: int = 1
# Genre layout for raw/processed text folders
GENRES = [
    "gothic",
    "romanticism",
    "realism",
    "modernism",
    "postmodernism",
]

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

    # --- Fanu (Blackmask) ---
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

DEFAULT_BOOK_CONFIG = { #### I think I can remove this? or do i need to update to some default if I want to run with another book without making configs??
    "pages": None,
    "use_text_flow": False,
    "extract_kwargs": None,
    "start_marker": None,
    "end_marker": None,
    "patterns": None,
}

DASHBOARD_WINDOW_CONFIG = {
    "discourse": {
        "keep_keys": {
            "explicit_connectives_per_token",
            "modality_per_token",
            "connective_counts_per_token",
            "tense_shift",
            "entity_overlap_ratio",
            "entity_overlap_per_token",
            "content_overlap_ratio",
            "content_overlap_per_token",
            "pronoun_ratio",
        },
        "nested_keys": {"connective_counts_per_token"},
    },
    "lexico_semantics": {
        "keep_keys": {
            "lexical_density_per_token",
            "lexical_diversity_mattr",
            "avg_word_freq",
            "normalized_freq",
            "information_content",
        },
        "nested_keys": {"lexical_diversity_mattr"},
        "nested_subkeys": {"lexical_diversity_mattr": {"mattr_score"}},
    },
    "syntax": {
        "keep_keys": {
            "clause_counts_per_token",
            "clause_ratios",
            "avg_dependents_per_head",
            "avg_mean_dependency_distance",
            "avg_tokens_per_sentence",
            "median_depth",
            "max_depth",
            "depth_skew",
            "punctuation_per_token",
        },
        "nested_keys": {
            "clause_counts_per_token",
            "clause_ratios",
            "avg_dependents_per_head",
        },
    },
    "log_prob": {
        "keep_keys": {
            "token_weighted_mean_surprisal",
            "token_weighted_surprisal_variance",
            "max_token_surprisal",
        },
    },
}

DEFAULT_CENTRAL_PRESENCE_P = 2.0
DEFAULT_CENTRAL_PRESENCE_NORMALIZE = True
DEFAULT_CENTRALITY_TOP_SCORE_FRACTION = 0.1
DEFAULT_CENTRALITY_COHERENCE_FLOOR = 0.3
DEFAULT_CENTRALITY_EXCLUSIVITY_FLOOR = 0.3

@dataclass(frozen=True)
class DashboardCorrelationConfig:
    block_size: int = DEFAULT_BLOCK_SIZE
    permutations: int = DEFAULT_DASHBOARD_PERMUTATIONS
    loop_enabled: bool = LOOP_BLOCK_SIZES_ENABLED
    loop_block_sizes: Sequence[int] = LOOP_BLOCK_SIZES
    loop_output_template: str = "dashboard_L{block_size}"

@dataclass(frozen=True)
class CentralTopicSelectionConfig:
    near_top_alpha: float = 0.85
    max_topics: Optional[int] = None
    top_score_fraction: float = DEFAULT_CENTRALITY_TOP_SCORE_FRACTION
    coherence_floor: float = DEFAULT_CENTRALITY_COHERENCE_FLOOR
    exclusivity_floor: float = DEFAULT_CENTRALITY_EXCLUSIVITY_FLOOR


@dataclass(frozen=True)
class CentralTopicXBarConfig:
    top_n: int = 10
    p_threshold: float = 0.05
    fig_width: float = 9.0
    min_height: float = 4.5
    row_height: float = 0.45
    annotation_max_len: int = 40
    positive_color: str = "#2c7fb8"
    negative_color: str = "#d95f0e"
    alpha_significant: float = 0.9
    alpha_nonsignificant: float = 0.4


@dataclass(frozen=True)
class ExemplarScatterConfig:
    top_per_genre: int = 1
    min_points: int = 3
    fig_width: float = 7.5
    fig_height: float = 5.5
    point_size: float = 18.0
    point_alpha: float = 0.7
    cmap_name: str = "viridis"
    ci_z: float = 1.96


@dataclass(frozen=True)
class PresenceSlopegraphConfig:
    p_threshold: float = 0.01
    fig_width: float = 9.0
    min_height: float = 3.5
    row_height: float = 0.4
    positive_color: str = "#2c7fb8"
    negative_color: str = "#d95f0e"


DEFAULT_EXCLUDE_METRICS = (
    "syntax.clause_ratios.subordination_ratio",
    "discourse.connective_counts_per_token.Comparison",
    "discourse.explicit_connectives_per_token",
    "discourse.tense_shift",
    "lexico_semantics.content_function_ratio",
    "lexico_semantics.lexical_density_per_token",
    "lexico_semantics.lexical_diversity_mattr.mattr_score",
    "discourse.modality_per_token",
    "syntax.avg_dependents_per_head.main_clause",
    "syntax.avg_dependents_per_head.subordinate_clause",
)


CORE_SIGNATURE_METRICS = (
    "syntax.median_depth",
    "syntax.clause_ratios.coordination_ratio",
    "syntax.avg_tokens_per_sentence",
    "discourse.content_overlap_ratio",
    "log_prob.token_weighted_mean_surprisal",
)


@dataclass(frozen=True)
class ConvergenceIndexConfig:
    metrics: Sequence[str] = ("significant_count",)
    p_threshold: float = 0.05
    fig_width: float = 8.0
    fig_height: float = 4.0
    line_width: float = 2.0
    marker_size: float = 5.0
    zero_nonsignificant: bool = True
    sign_agreement_min_texts: int = 2
    sign_agreement_use_p_threshold: bool = False


@dataclass(frozen=True)
class AggregatedHeatmapConfig:
    p_threshold: float = 0.05
    fig_width: float = 9.0
    min_height: float = 6.0
    row_height: float = 0.3
    cmap_name: str = "coolwarm"
    mask_color: str = "#d9d9d9"
    exclude_metrics: Sequence[str] = (
        "syntax.avg_dependents_per_head.subordinate_clause",
        "syntax.clause_counts_per_token.subordinate",
        "discourse.tense_shift",
        "discourse.connective_counts_per_token.Temporal",
    )


@dataclass(frozen=True)
class TopicMetricHeatmapConfig:
    value_key: str = "variance_delta"
    min_windows: int = 2
    top_n: Optional[int] = None
    min_width: float = 8.0
    min_height: float = 6.0
    col_width: float = 0.5
    row_height: float = 0.4
    cmap_name: str = "viridis"
    mask_color: str = "lightgrey"


@dataclass(frozen=True)
class ForestPlotConfig:
    metrics: Sequence[str] = CORE_SIGNATURE_METRICS
    p_threshold: float = 0.05
    fig_width: float = 8.0
    min_height: float = 4.5
    row_height: float = 0.35
    point_size: float = 30.0
    aggregate_size: float = 60.0
    line_width: float = 1.6
    positive_color: str = "#2c7fb8"
    negative_color: str = "#d95f0e"
    alpha_significant: float = 0.9
    alpha_nonsignificant: float = 0.35
    ci_z: float = 1.96
    xlim: Optional[Tuple[float, float]] = (-1.0, 1.0)
    label_max_len: int = 45


@dataclass(frozen=True)
class TextMetricHeatmapConfig:
    p_threshold: float = 0.05
    min_width: float = 10.0
    min_height: float = 6.0
    col_width: float = 0.4
    row_height: float = 0.35
    cmap_name: str = "coolwarm"
    mask_color: str = "#d9d9d9"
    exclude_metrics: Sequence[str] = DEFAULT_EXCLUDE_METRICS
    top_n: Optional[int] = None
    metrics: Optional[Sequence[str]] = None
    label_max_len: int = 45


@dataclass(frozen=True)
class CentralTopicWindowHeatmapConfig:
    min_width: float = 10.0
    min_height: float = 4.5
    col_width: float = 0.06
    row_height: float = 0.5
    cmap_name: str = "viridis"
    mask_color: str = "lightgrey"
    label_max_len: int = 45
    show_keywords: bool = True
    vmin: float = 0.0
    vmax: Optional[float] = 1.0
    max_xticks: int = 12


@dataclass(frozen=True)
class StabilityFilterConfig:
    metric_key: str = "sign_agreement_rate"
    threshold: float = 0.8
    direction: str = "gte"
    min_pair_count: Optional[int] = None


@dataclass(frozen=True)
class StabilityStackedBarConfig:
    stability: StabilityFilterConfig = StabilityFilterConfig()
    fig_width: float = 9.0
    fig_height: float = 4.5
    family_order: Sequence[str] = ("syntax", "discourse", "lexico_semantics", "log_prob")
    family_colors: Sequence[str] = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a")
    bar_alpha: float = 0.85


@dataclass(frozen=True)
class TopicMetricLineConfig:
    families: Sequence[str] = ("syntax", "discourse", "lexico_semantics")
    top_n_metrics: int = 3
    p_threshold: Optional[float] = None
    normalize: bool = True
    normalization: str = "zscore"
    fig_width: float = 11.0
    fig_height: float = 4.0
    topic_color: str = "#222222"
    metric_colors: Sequence[str] = (
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
    )
    topic_line_width: float = 2.2
    metric_line_width: float = 1.4
    line_alpha: float = 0.85
    label_max_len: int = 45
    max_xticks: int = 12


@dataclass(frozen=True)
class DataSelectionConfig:
    genres: Optional[Sequence[str]] = None
    authors: Optional[Sequence[str]] = None
    texts: Optional[Sequence[str]] = None
    categories: Optional[Sequence[str]] = None
    exclude_genres: Optional[Sequence[str]] = None
    exclude_authors: Optional[Sequence[str]] = None
    exclude_texts: Optional[Sequence[str]] = None
    exclude_categories: Optional[Sequence[str]] = None

DEFAULT_CENTRAL_TOPIC_X_CONFIG = CentralTopicXBarConfig()
DEFAULT_CENTRAL_TOPIC_SELECTION_CONFIG = CentralTopicSelectionConfig()
DEFAULT_DASHBOARD_CORRELATION_CONFIG = DashboardCorrelationConfig()
DEFAULT_EXEMPLAR_SCATTER_CONFIG = ExemplarScatterConfig()
DEFAULT_PRESENCE_SLOPEGRAPH_CONFIG = PresenceSlopegraphConfig()
DEFAULT_CONVERGENCE_INDEX_CONFIG = ConvergenceIndexConfig()
DEFAULT_AGGREGATED_HEATMAP_CONFIG = AggregatedHeatmapConfig()
DEFAULT_TOPIC_METRIC_HEATMAP_CONFIG = TopicMetricHeatmapConfig()
DEFAULT_FOREST_PLOT_CONFIG = ForestPlotConfig()
DEFAULT_TEXT_METRIC_HEATMAP_CONFIG = TextMetricHeatmapConfig()
DEFAULT_CENTRAL_TOPIC_WINDOW_HEATMAP_CONFIG = CentralTopicWindowHeatmapConfig()
DEFAULT_STABILITY_FILTER_CONFIG = StabilityFilterConfig()
DEFAULT_STABILITY_STACKED_BAR_CONFIG = StabilityStackedBarConfig()
DEFAULT_TOPIC_METRIC_LINE_CONFIG = TopicMetricLineConfig()
DEFAULT_DATA_SELECTION_CONFIG = DataSelectionConfig(genres=tuple(GENRES))

CONVERGENCE_METRIC_LABELS = {
    "significant_count": "Significant metrics (count)",
    "mean_abs_r": "Mean |r|",
    "mean_abs_r_zeroed": "Mean |r| (nonsig=0)",
    "sign_agreement": "Sign agreement (proportion)",
}

@lru_cache(maxsize=None)
def load_spacy_model(
    model_name: str = DEFAULT_SPACY_MODEL,
    disable: Optional[Sequence[str]] = None,
):
    """
    Shared spaCy loader with simple caching driven by config defaults.
    Pass a different model_name/disable list to override per call.
    """
    disable_components = tuple(disable) if disable else DEFAULT_SPACY_DISABLE
    return spacy.load(model_name, disable=list(disable_components))
