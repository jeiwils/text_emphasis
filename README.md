# Textual Emphasis Analysis

PYTHON 3.10


A Python package for analyzing textual emphasis through linguistic and network-based approaches.







all metrics are taken from sentence windows, and these windows are then matched with the top topics 






## Pipeline modules (a–d)

- **a – preprocessing/cleaning** (`src/a_preprocessing_cleaning.py`): spaCy-based tokenization, whitespace cleaning, and lemmatization; Whisper ASR transcription for audio; PDF extraction with per-book configs for page ranges, start/end markers, and boilerplate removal; helpers to preprocess all PDFs under `data/raw_texts` into `data/processed_texts/cleaned`.
- **b – whole-text metrics** (`src/b2_RUNNING_text_analytics.py`): loads the configured Hugging Face causal LM (`src/x_configs.py`) to compute chunked token log-probabilities, per-text averages, and top word frequencies; writes full-text corpus metrics and per-chunk stats to `data/processed_texts/corpus`.
- **c – sentence/window analytics**:
  - `src/c1_syntactics.py`: clause counts, embedding depth, and dependency complexity aggregated over sliding windows, plus plotting via `SyntaxVisualiser`; outputs window JSONs under `data/processed_texts/window` and saves graphs.
  - `src/c2_lexico_semantics_TODO_DEPENDS_ON_CORPUS.py`: lexical density, information content (using corpus frequencies), cohesion overlap, semantic roles, and sliding-window averages for word frequency and surprisal (from chunk log-probs).
  - `src/c3_discourse_TODO.py`: heuristic discourse markers mapped to PDTB-style relations, entity/content overlap, pronoun ratio, tense-shift flags, window aggregates, and summary stats; `run_discourse_analysis` saves JSON next to other window metrics.
  - `src/c4_RUNNING_window_metrics.py`: orchestrates sentence/window metrics by loading corpus outputs from module **b**, rebuilding text if needed, and emitting combined syntax + lexico-semantic metrics to `data/processed_texts/window`.
- **d – sliding-window diversity** (`src/d1_window_TODO.py`): computes moving-average type–token ratio (MATTR) over tokenized text for quick lexical diversity checks.
