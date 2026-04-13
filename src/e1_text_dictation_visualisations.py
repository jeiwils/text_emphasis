"""Dictation-focused visualisations for book-level stylistic drift.

This module builds a compact figure suite around syntax and pacing:
1) Syntax stretch (sentence length vs subordinate-clause depth).
2) Clause-density metrics from saved syntax files.
3) Parataxis / hypotaxis early-vs-late dumbbells.
4) Parataxis / hypotaxis six-text trajectories.
5) Mean dependency distance slopegraph.
6) Lexical-density bubble charts on their own axis.

It generates grouped dictation plots for each cohort and author without relying on
any topic-modelling visualisations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

from spacy.tokens import Doc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .b3_log_prob_metrics import WholeTextMetrics
from .c1_syntactics import SyntaxAnalyzer
from .c2_lexico_semantics import LexicoSemanticsAnalyzer
from .x_configs import DEFAULT_BLOCK_SIZE, DEFAULT_USE_EXISTING
from .z_utils import analytics_path, load_spacy_model, results_path, text_path, window_metrics_path

PHASES: Tuple[str, str, str] = ("early", "middle", "late")
PHASE_LABELS: Dict[str, str] = {"early": "Early", "middle": "Middle", "late": "Late"}

PART_SPECS: Dict[str, Dict[str, object]] = {
    "part_1_physical_cognitive": {
        "title": "Part I: Physical & Cognitive Ailment → Dictation",
        "authors": ("james", "conrad", "scott", "stevenson"),
    },
    "part_2_visual_impairment": {
        "title": "Part II: Visual Impairment → Dictation",
        "authors": ("huxley", "tarkington", "collins", "hearn"),
    },
}


def _compose_figure_title(figure_title: str, plot_title: str) -> str:
    """Drop leading part labels while keeping any useful author line."""
    cleaned_lines: List[str] = []
    for line in str(figure_title or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("part "):
            continue
        cleaned_lines.append(stripped)
    return "\n".join([*cleaned_lines, plot_title]) if cleaned_lines else plot_title


AUTHOR_DISPLAY_NAMES: Dict[str, str] = {
    "james": "Henry James",
    "conrad": "Joseph Conrad",
    "scott": "Walter Scott",
    "stevenson": "Robert Louis Stevenson",
    "huxley": "Aldous Huxley",
    "tarkington": "Booth Tarkington",
    "collins": "Wilkie Collins",
    "hearn": "Lafcadio Hearn",
}

AUTHOR_TO_PART: Dict[str, str] = {
    "james": "part_1_physical_cognitive",
    "conrad": "part_1_physical_cognitive",
    "scott": "part_1_physical_cognitive",
    "stevenson": "part_1_physical_cognitive",
    "huxley": "part_2_visual_impairment",
    "tarkington": "part_2_visual_impairment",
    "collins": "part_2_visual_impairment",
    "hearn": "part_2_visual_impairment",
}

AUTHOR_PHASE_TEXTS: Dict[str, Dict[str, str]] = {
    "james": {
        "henry_james_the_american_1877": "early",
        "henry_james_the_portrait_of_a_lady_1881": "early",
        "henry_james_the_spoils_of_poynton_1897": "middle",
        "henry_james_the_awkward_age_1899": "middle",
        "henry_james_the_ambassadors_1903": "late",
        "henry_james_the_golden_bowl_1904": "late",
    },
    "conrad": {
        "joseph_conrad_almayers_folly_1895": "early",
        "joseph_conrad_lord_jim_1900": "early",
        "joseph_conrad_nostromo_1904": "middle",
        "joseph_conrad_the_secret_agent_1907": "middle",
        "joseph_conrad_the_arrow_of_gold_1919": "late",
        "joseph_conrad_the_rover_1923": "late",
    },
    "scott": {
        "walter_scott_waverley_1814": "early",
        "walter_scott_the_antiquary_1816": "early",
        "walter_scott_the_bride_of_lammermoor_1819": "middle",
        "walter_scott_a_legend_of_montrose_1819": "middle",
        "walter_scott_count_robert_of_paris_1831": "late",
        "walter_scott_castle_dangerous_1831": "late",
    },
    "stevenson": {
        "robert_louis_stevenson_treasure_island_1883": "early",
        "robert_louis_stevenson_kidnapped_1886": "early",
        "robert_louis_stevenson_the_master_of_ballantrae_1889": "middle",
        "robert_louis_stevenson_the_wrecker_1892": "middle",
        "robert_louis_stevenson_weir_of_hermiston_1896": "late",
        "robert_louis_stevenson_st_ives_1897": "late",
    },
    "huxley": {
        "aldous_huxley_crome_yellow_1921": "early",
        "aldous_huxley_antic_hay_1923": "early",
        "aldous_huxley_point_counter_point_1928": "middle",
        "aldous_huxley_brave_new_world_1932": "middle",
        "aldous_huxley_the_genius_and_the_goddess_1955": "late",
        "aldous_huxley_island_1962": "late",
    },
    "tarkington": {
        "booth_tarkington_the_gentleman_from_indiana_1899": "early",
        "booth_tarkington_the_magnificent_ambersons_1918": "early",
        "booth_tarkington_alice_adams_1921": "middle",
        "booth_tarkington_the_plutocrat_1927": "middle",
        "booth_tarkington_the_heritage_of_hatcher_ide_1941": "late",
        "booth_tarkington_kate_fennigate_1943": "late",
    },
    "collins": {
        "wilkie_collins_the_antonina_1850": "early",
        "wilkie_collins_basil_1852": "early",
        "wilkie_collins_the_moonstone_1868": "middle",
        "wilkie_collins_man_and_wife_1870": "middle",
        "wilkie_collins_heart_and_science_1883": "late",
        "wilkie_collins_the_legacy_of_cain_1889": "late",
    },
    "hearn": {
        "lafciadio_hearn_chita_a_memory_of_last_island_1889": "early",
        "lafciadio_hearn_youma_the_story_of_a_west_indian_slave_1890": "early",
        "lafciadio_hearn_kokoro_hints_and_echoes_of_japanese_inner_life_1896": "middle",
        "lafciadio_hearn_in_ghostly_japan_1899": "middle",
        "lafciadio_hearn_kwaidan_stories_and_studies_of_strange_things_1904": "late",
        "lafciadio_hearn_the_romance_of_the_milky_way_1905": "late",
    },
}

DISCOURSE_RELATIONS: Tuple[str, ...] = ("Expansion", "Contingency", "Comparison", "Temporal")
AUTHOR_ORDER: Tuple[str, ...] = (
    "james",
    "conrad",
    "scott",
    "stevenson",
    "huxley",
    "tarkington",
    "collins",
    "hearn",
)
AUTHOR_COLORS: Dict[str, str] = {
    "james": "#1b9e77",
    "conrad": "#d95f02",
    "scott": "#7570b3",
    "stevenson": "#e7298a",
    "huxley": "#66a61e",
    "tarkington": "#e6ab02",
    "collins": "#a6761d",
    "hearn": "#1f78b4",
}
AUTHOR_LINESTYLES: Dict[str, str] = {
    "james": "-",
    "conrad": "--",
    "scott": "-.",
    "stevenson": ":",
    "huxley": "-",
    "tarkington": "--",
    "collins": "-.",
    "hearn": ":",
}
AUTHOR_MARKERS: Dict[str, str] = {
    "james": "o",
    "conrad": "s",
    "scott": "^",
    "stevenson": "D",
    "huxley": "o",
    "tarkington": "s",
    "collins": "^",
    "hearn": "D",
}
GROUP_COLORS: Dict[str, str] = {
    "part_1_physical_cognitive": "#c44e52",
    "part_2_visual_impairment": "#4c72b0",
}
PHASE_COLORS: Dict[str, str] = {
    "early": "#bdbdbd",
    "middle": "#9ecae1",
    "late": "#08519c",
}


def _load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values or not weights:
        return 0.0
    denom = float(sum(weights))
    if denom <= 0:
        return float(mean(values)) if values else 0.0
    return float(sum(v * w for v, w in zip(values, weights)) / denom)


def _author_display_name(author: str) -> str:
    return AUTHOR_DISPLAY_NAMES.get(author, author.replace("_", " ").title())


def _mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    cleaned = [float(value) for value in values if isinstance(value, (int, float))]
    return float(mean(cleaned)) if cleaned else None


def _sentence_numeric_values(sentences: Sequence[object], key: str) -> List[float]:
    values: List[float] = []
    for sent in sentences:
        if not isinstance(sent, dict):
            continue
        value = sent.get(key)
        if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
            values.append(float(value))
    return values


def _sentences_need_recomputed_syntax_metrics(sentences: Sequence[object]) -> bool:
    """Return True when legacy syntax rows are missing the newer entropy/rhythm fields."""
    required_numeric_keys = (
        "normalized_structural_entropy",
        "normalized_pos_ngram_entropy",
        "sentence_length_approx_entropy",
    )
    for key in required_numeric_keys:
        values = _sentence_numeric_values(sentences, key)
        if not values:
            return True
        if key in {"normalized_pos_ngram_entropy", "sentence_length_approx_entropy"} and all(
            abs(value) <= 1e-12 for value in values
        ):
            return True

    has_parataxis = any(
        isinstance(sent, dict) and isinstance(sent.get("parataxis_hypotaxis"), dict) and bool(sent.get("parataxis_hypotaxis"))
        for sent in sentences
    )
    has_breath_units = any(
        isinstance(sent, dict) and isinstance(sent.get("breath_unit_counts"), dict) and bool(sent.get("breath_unit_counts"))
        for sent in sentences
    )
    return not (has_parataxis and has_breath_units)


def _build_doc_from_segmented_sentences(segmented_sentences: Sequence[str]) -> Optional[Doc]:
    if not segmented_sentences:
        return None

    nlp = load_spacy_model()
    tokenized_docs = [nlp.make_doc(text) for text in segmented_sentences]
    for sent_doc in tokenized_docs:
        for idx, token in enumerate(sent_doc):
            token.is_sent_start = idx == 0

    doc = Doc.from_docs(tokenized_docs)
    for name, proc in nlp.pipeline:
        if name == "senter":
            continue
        doc = proc(doc)
    return doc


def _recompute_syntax_sentences(author: str, text_name: str) -> List[Dict[str, object]]:
    """Fallback parse for syntax metrics when saved analytics predate the new fields."""
    segmented_path = text_path(
        "processed",
        "cleaned_segmented_texts",
        [author, author],
        f"{text_name}_cleaned_segmented.jsonl",
    )
    segmented_sentences = _read_segmented_sentences(segmented_path)
    doc = _build_doc_from_segmented_sentences(segmented_sentences)
    if doc is None:
        return []

    nlp = load_spacy_model()
    metrics = SyntaxAnalyzer(nlp).analyze_document(doc, window_size=3)
    sentences = metrics.get("sentences", []) if isinstance(metrics, dict) else []
    return sentences if isinstance(sentences, list) else []


def _recompute_lexico_sentences(author: str, text_name: str) -> List[Dict[str, object]]:
    """Fallback parse for lexical metrics when saved analytics predate the new entropy fields."""
    segmented_path = text_path(
        "processed",
        "cleaned_segmented_texts",
        [author, author],
        f"{text_name}_cleaned_segmented.jsonl",
    )
    segmented_sentences = _read_segmented_sentences(segmented_path)
    doc = _build_doc_from_segmented_sentences(segmented_sentences)
    if doc is None:
        return []

    nlp = load_spacy_model()
    metrics = LexicoSemanticsAnalyzer(nlp, corpus_freqs={}).analyze_document(doc, window_size=3)
    sentences = metrics.get("sentences", []) if isinstance(metrics, dict) else []
    return sentences if isinstance(sentences, list) else []


def _weighted_row_metric(rows: Sequence[Dict[str, object]], value_key: str, weight_key: str) -> float:
    values: List[float] = []
    weights: List[float] = []
    for row in rows:
        values.append(float(row.get(value_key) or 0.0))
        weight = float(row.get(weight_key) or 0.0)
        weights.append(weight if weight > 0 else 1.0)
    return _weighted_mean(values, weights)


def _aggregate_phase_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {}

    sample = rows[0]
    discourse_weights = [float(row.get("discourse_sentence_count") or 0.0) for row in rows]
    surprisal_weights = [float(row.get("surprisal_weight_total") or 0.0) for row in rows]
    text_names = [str(row.get("text_name") or "") for row in rows if row.get("text_name")]

    relation_map: Dict[str, float] = {}
    for relation in DISCOURSE_RELATIONS:
        relation_values = [
            float(((row.get("discourse_density_by_relation") or {}).get(relation) or 0.0))
            for row in rows
        ]
        relation_map[relation] = _weighted_mean(
            relation_values,
            [weight if weight > 0 else 1.0 for weight in discourse_weights],
        )

    max_depth_series = [
        float(depth)
        for row in rows
        for depth in (row.get("max_depth_series") or [])
        if isinstance(depth, (int, float))
    ]
    entropy_series = [
        float(value)
        for row in rows
        for value in (row.get("normalized_structural_entropy_series") or [])
        if isinstance(value, (int, float))
    ]
    mean_dependency_distance_series = [
        float(value)
        for row in rows
        for value in (row.get("mean_dependency_distance_series") or [])
        if isinstance(value, (int, float))
    ]
    lexical_density_series = [
        float(value)
        for row in rows
        for value in (row.get("lexical_density_series") or [])
        if isinstance(value, (int, float))
    ]
    lexical_entropy_series = [
        float(value)
        for row in rows
        for value in (row.get("lexical_entropy_series") or [])
        if isinstance(value, (int, float))
    ]
    pos_ngram_entropy_series = [
        float(value)
        for row in rows
        for value in (row.get("normalized_pos_ngram_entropy_series") or [])
        if isinstance(value, (int, float))
    ]
    rhythm_apen_series = [
        float(value)
        for row in rows
        for value in (row.get("sentence_length_approx_entropy_series") or [])
        if isinstance(value, (int, float))
    ]
    breath_unit_length_series = [
        float(value)
        for row in rows
        for value in (row.get("breath_unit_length_series") or [])
        if isinstance(value, (int, float))
    ]
    total_parataxis = float(sum(float(row.get("parataxis_count") or 0.0) for row in rows))
    total_hypotaxis = float(sum(float(row.get("hypotaxis_count") or 0.0) for row in rows))

    return {
        "genre": sample.get("genre"),
        "author": sample.get("author"),
        "group": sample.get("group"),
        "phase": sample.get("phase"),
        "text_name": ", ".join(text_names),
        "text_names": text_names,
        "avg_sentence_length": _weighted_row_metric(rows, "avg_sentence_length", "sentence_count"),
        "avg_subordinate_per_sentence": _weighted_row_metric(rows, "avg_subordinate_per_sentence", "sentence_count"),
        "avg_max_depth": _weighted_row_metric(rows, "avg_max_depth", "sentence_count"),
        "avg_main_dependents_per_head": _weighted_row_metric(rows, "avg_main_dependents_per_head", "sentence_count"),
        "avg_subordinate_dependents_per_head": _weighted_row_metric(rows, "avg_subordinate_dependents_per_head", "sentence_count"),
        "avg_coordinate_dependents_per_head": _weighted_row_metric(rows, "avg_coordinate_dependents_per_head", "sentence_count"),
        "avg_discourse_density": _weighted_row_metric(rows, "avg_discourse_density", "discourse_sentence_count"),
        "discourse_density_by_relation": relation_map,
        "mean_surprisal": _weighted_mean(
            [float(row.get("mean_surprisal") or 0.0) for row in rows],
            [weight if weight > 0 else 1.0 for weight in surprisal_weights],
        ),
        "surprisal_variance": _weighted_mean(
            [float(row.get("surprisal_variance") or 0.0) for row in rows],
            [weight if weight > 0 else 1.0 for weight in surprisal_weights],
        ),
        "coordination_ratio": _weighted_row_metric(rows, "coordination_ratio", "sentence_count"),
        "subordination_ratio": _weighted_row_metric(rows, "subordination_ratio", "sentence_count"),
        "normalized_structural_entropy": float(mean(entropy_series)) if entropy_series else 0.0,
        "normalized_structural_entropy_std": float(np.std(np.asarray(entropy_series, dtype=float))) if entropy_series else 0.0,
        "normalized_structural_entropy_series": entropy_series,
        "mean_dependency_distance": _weighted_row_metric(rows, "mean_dependency_distance", "sentence_count"),
        "mean_dependency_distance_series": mean_dependency_distance_series,
        "lexical_density": _weighted_row_metric(rows, "lexical_density", "sentence_count"),
        "lexical_density_series": lexical_density_series,
        "lexical_entropy": float(mean(lexical_entropy_series)) if lexical_entropy_series else 0.0,
        "lexical_entropy_series": lexical_entropy_series,
        "normalized_pos_ngram_entropy": float(mean(pos_ngram_entropy_series)) if pos_ngram_entropy_series else 0.0,
        "normalized_pos_ngram_entropy_series": pos_ngram_entropy_series,
        "sentence_length_approx_entropy": float(mean(rhythm_apen_series)) if rhythm_apen_series else 0.0,
        "sentence_length_approx_entropy_series": rhythm_apen_series,
        "cross_entropy_to_early": _weighted_row_metric(rows, "cross_entropy_to_early", "sentence_count"),
        "kl_divergence_to_early": _weighted_row_metric(rows, "kl_divergence_to_early", "sentence_count"),
        "parataxis_count": total_parataxis,
        "hypotaxis_count": total_hypotaxis,
        "parataxis_to_hypotaxis_ratio": _safe_ratio(total_parataxis, total_hypotaxis),
        "breath_unit_length_mean": float(mean(breath_unit_length_series)) if breath_unit_length_series else 0.0,
        "breath_unit_length_series": breath_unit_length_series,
        "max_depth_series": max_depth_series,
        "sentence_count": float(sum(float(row.get("sentence_count") or 0.0) for row in rows)),
        "discourse_sentence_count": float(sum(discourse_weights)),
        "surprisal_weight_total": float(sum(surprisal_weights)),
    }


def _author_phase_rows(
    rows: Sequence[Dict[str, object]],
    *,
    authors: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    selected = set(authors) if authors is not None else None
    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = {}

    for row in rows:
        author = str(row.get("author") or "")
        if selected is not None and author not in selected:
            continue
        phase = str(row.get("phase") or "")
        grouped.setdefault(author, {}).setdefault(phase, []).append(dict(row))

    return {
        author: {phase: _aggregate_phase_rows(phase_rows) for phase, phase_rows in phase_map.items()}
        for author, phase_map in grouped.items()
    }


def _ordered_rows_for_author(rows: Sequence[Dict[str, object]], author: str) -> List[Dict[str, object]]:
    author_rows = [dict(row) for row in rows if str(row.get("author") or "") == author]
    explicit_order = list(AUTHOR_PHASE_TEXTS.get(author, {}).keys())
    if explicit_order:
        order_map = {name: idx for idx, name in enumerate(explicit_order)}
        return sorted(author_rows, key=lambda row: order_map.get(str(row.get("text_name") or ""), 10_000))
    return sorted(author_rows, key=lambda row: str(row.get("text_name") or ""))


def _part_specs_for_authors(authors: Sequence[str]) -> List[Tuple[str, Dict[str, object]]]:
    author_set = {str(author) for author in authors}
    part_items = [
        (part_key, part_spec)
        for part_key, part_spec in PART_SPECS.items()
        if not author_set or any(author in author_set for author in tuple(part_spec.get("authors", ())))
    ]
    return part_items or list(PART_SPECS.items())


def _text_sequence_label(author: str, text_name: str) -> str:
    ordered = list(AUTHOR_PHASE_TEXTS.get(author, {}).keys())
    year = text_name.rsplit("_", 1)[-1] if "_" in text_name else ""
    if text_name in ordered:
        idx = ordered.index(text_name)
        phase_code = ("E", "M", "L")[min(idx // 2, 2)]
        within_phase = (idx % 2) + 1
        return f"{phase_code}{within_phase}\n{year}" if year.isdigit() else f"{phase_code}{within_phase}"
    return year if year.isdigit() else text_name


def _author_text_tick_label(author: str, text_name: str) -> str:
    surname = _author_display_name(author).split()[-1]
    phase_label = _text_sequence_label(author, text_name).splitlines()[0]
    return f"{surname}\n{phase_label}"


def _phase_mapping_for_author(author: str, text_names: Sequence[str]) -> Dict[str, str]:
    explicit = AUTHOR_PHASE_TEXTS.get(author, {})
    if author in AUTHOR_PHASE_TEXTS:
        return {name: phase for name, phase in explicit.items() if name in text_names}

    ordered = sorted(text_names)
    if not ordered:
        return {}
    if len(ordered) == 1:
        return {ordered[0]: "middle"}
    if len(ordered) == 2:
        return {ordered[0]: "early", ordered[1]: "late"}

    mapping: Dict[str, str] = {}
    for idx, text_name in enumerate(ordered):
        if idx == 0:
            mapping[text_name] = "early"
        elif idx == len(ordered) - 1:
            mapping[text_name] = "late"
        else:
            mapping[text_name] = "middle"
    return mapping


def _iter_author_roots(window_root: Path) -> List[Tuple[str, Path]]:
    candidates: List[Tuple[str, Path]] = []
    seen: set[str] = set()

    for author in AUTHOR_ORDER:
        author_root = window_root / author / author
        if author_root.exists():
            candidates.append((author, author_root))
            seen.add(author)

    if window_root.exists():
        for outer_dir in sorted(window_root.iterdir()):
            if not outer_dir.is_dir():
                continue
            author = outer_dir.name
            author_root = outer_dir / author
            if author in seen or not author_root.exists() or not author_root.is_dir():
                continue
            candidates.append((author, author_root))
            seen.add(author)

    return candidates


def _collect_text_rows(
    window_root: Path,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for author, author_root in _iter_author_roots(window_root):
        text_dirs = sorted([entry for entry in author_root.iterdir() if entry.is_dir()])
        if not text_dirs:
            continue

        phase_map = _phase_mapping_for_author(author, [entry.name for entry in text_dirs])
        group = AUTHOR_TO_PART.get(author, "part_unknown")

        for text_dir in text_dirs:
            phase = phase_map.get(text_dir.name)
            if phase is None:
                continue

            syntax_file = window_metrics_path(text_dir=text_dir, domain="syntax")
            lexico_file = window_metrics_path(text_dir=text_dir, domain="lexico_semantics")
            discourse_file = window_metrics_path(text_dir=text_dir, domain="discourse")
            log_prob_file = window_metrics_path(text_dir=text_dir, domain="log_prob")

            syntax_data = _load_json(syntax_file)
            lexico_data = _load_json(lexico_file)
            discourse_data = _load_json(discourse_file)
            log_prob_data = _load_json(log_prob_file)
            if not syntax_data or not lexico_data or not discourse_data or not log_prob_data:
                continue

            syntax_sentences = syntax_data.get("sentences") or []
            lexico_sentences = lexico_data.get("sentences") or []
            discourse_sentences = discourse_data.get("sentences") or []
            log_prob_sentences = log_prob_data.get("sentences") or []
            if not (
                isinstance(syntax_sentences, list)
                and isinstance(lexico_sentences, list)
                and isinstance(discourse_sentences, list)
                and isinstance(log_prob_sentences, list)
            ):
                continue

            needs_enhanced_syntax = _sentences_need_recomputed_syntax_metrics(syntax_sentences)
            enhanced_syntax_sentences = _recompute_syntax_sentences(author, text_dir.name) if needs_enhanced_syntax else []
            needs_enhanced_lexico = not any(
                isinstance(sent, dict) and "normalized_lexical_entropy" in sent
                for sent in lexico_sentences
            )
            enhanced_lexico_sentences = _recompute_lexico_sentences(author, text_dir.name) if needs_enhanced_lexico else []

            token_counts: List[float] = []
            subordinate_counts: List[float] = []
            max_depth_series: List[float] = []
            main_dependents_per_head: List[float] = []
            subordinate_dependents_per_head: List[float] = []
            coordinate_dependents_per_head: List[float] = []
            coordination_ratios: List[float] = []
            subordination_ratios: List[float] = []
            normalized_structural_entropy_series: List[float] = []
            pos_ngram_entropy_series: List[float] = []
            sentence_length_apen_series: List[float] = []
            mean_dependency_distance_series: List[float] = []
            lexical_density_series: List[float] = []
            lexical_entropy_series: List[float] = []
            breath_unit_length_series: List[float] = []
            parataxis_total = 0.0
            hypotaxis_total = 0.0

            for idx, sent in enumerate(syntax_sentences):
                if not isinstance(sent, dict):
                    continue
                fallback_sent = enhanced_syntax_sentences[idx] if idx < len(enhanced_syntax_sentences) else {}
                token_count = float(sent.get("token_count") or fallback_sent.get("token_count") or 0.0)
                clause_counts = sent.get("clause_counts") or fallback_sent.get("clause_counts") or {}
                clause_ratios = sent.get("clause_ratios") or fallback_sent.get("clause_ratios") or {}
                avg_dependents = sent.get("avg_dependents_per_head") or fallback_sent.get("avg_dependents_per_head") or {}
                token_counts.append(token_count)
                subordinate_counts.append(float((clause_counts or {}).get("subordinate") or 0.0))
                max_depth_series.append(float(sent.get("max_depth") or fallback_sent.get("max_depth") or 0.0))
                main_dependents_per_head.append(float((avg_dependents or {}).get("main_clause") or 0.0))
                subordinate_dependents_per_head.append(float((avg_dependents or {}).get("subordinate_clause") or 0.0))
                coordinate_dependents_per_head.append(float((avg_dependents or {}).get("coordinate_clause") or 0.0))
                coordination_ratios.append(float((clause_ratios or {}).get("coordination_ratio") or 0.0))
                subordination_ratios.append(float((clause_ratios or {}).get("subordination_ratio") or 0.0))
                normalized_structural_entropy_series.append(
                    float(sent.get("normalized_structural_entropy") or fallback_sent.get("normalized_structural_entropy") or 0.0)
                )
                pos_ngram_entropy_series.append(
                    float(sent.get("normalized_pos_ngram_entropy") or fallback_sent.get("normalized_pos_ngram_entropy") or 0.0)
                )
                sentence_length_apen_series.append(
                    float(sent.get("sentence_length_approx_entropy") or fallback_sent.get("sentence_length_approx_entropy") or 0.0)
                )
                mean_dependency_distance_series.append(
                    float(sent.get("avg_mean_dependency_distance") or fallback_sent.get("avg_mean_dependency_distance") or 0.0)
                )
                parataxis_payload = sent.get("parataxis_hypotaxis") or fallback_sent.get("parataxis_hypotaxis") or {}
                if isinstance(parataxis_payload, dict) and parataxis_payload:
                    parataxis_total += float(parataxis_payload.get("parataxis_count") or 0.0)
                    hypotaxis_total += float(parataxis_payload.get("hypotaxis_count") or 0.0)
                else:
                    parataxis_total += float((clause_ratios or {}).get("coordination_ratio") or 0.0)
                    hypotaxis_total += float((clause_ratios or {}).get("subordination_ratio") or 0.0)
                breath_unit_counts = sent.get("breath_unit_counts") or fallback_sent.get("breath_unit_counts") or {}
                if isinstance(breath_unit_counts, dict):
                    pause_total = float(
                        sum(
                            float(value)
                            for value in breath_unit_counts.values()
                            if isinstance(value, (int, float))
                        )
                    )
                else:
                    pause_total = float(sent.get("punctuation_count") or fallback_sent.get("punctuation_count") or 0.0)
                breath_unit_length_series.append(
                    float(token_count / (pause_total + 1.0)) if token_count > 0 else 0.0
                )

            for idx, sent in enumerate(lexico_sentences):
                if not isinstance(sent, dict):
                    continue
                fallback_sent = enhanced_lexico_sentences[idx] if idx < len(enhanced_lexico_sentences) else {}
                lexical_density_series.append(float(sent.get("lexical_density") or fallback_sent.get("lexical_density") or 0.0))
                lexical_entropy_series.append(
                    float(sent.get("normalized_lexical_entropy") or fallback_sent.get("normalized_lexical_entropy") or 0.0)
                )

            discourse_density: List[float] = []
            relation_density: Dict[str, List[float]] = {key: [] for key in DISCOURSE_RELATIONS}
            for sent in discourse_sentences:
                if not isinstance(sent, dict):
                    continue
                discourse_density.append(float(sent.get("explicit_connectives_per_token") or 0.0))
                rel_counts = sent.get("connective_counts_per_token") or {}
                for relation in DISCOURSE_RELATIONS:
                    relation_density[relation].append(float((rel_counts or {}).get(relation) or 0.0))

            surprisal_means: List[float] = []
            surprisal_vars: List[float] = []
            surprisal_weights: List[float] = []
            for sent in log_prob_sentences:
                if not isinstance(sent, dict):
                    continue
                surprisal = sent.get("sentence_surprisal_metrics") or {}
                if not isinstance(surprisal, dict):
                    continue
                num_tokens = float(surprisal.get("num_tokens") or 0.0)
                surprisal_means.append(float(surprisal.get("mean_surprisal") or 0.0))
                surprisal_vars.append(float(surprisal.get("surprisal_variance") or 0.0))
                surprisal_weights.append(max(1.0, num_tokens))

            avg_sentence_length = float(mean(token_counts)) if token_counts else 0.0
            avg_subordinate_per_sentence = float(mean(subordinate_counts)) if subordinate_counts else 0.0

            row = {
                "genre": author,
                "author": author,
                "group": group,
                "phase": phase,
                "text_name": text_dir.name,
                "avg_sentence_length": avg_sentence_length,
                "avg_subordinate_per_sentence": avg_subordinate_per_sentence,
                "avg_max_depth": float(mean(max_depth_series)) if max_depth_series else 0.0,
                "avg_main_dependents_per_head": float(mean(main_dependents_per_head)) if main_dependents_per_head else 0.0,
                "avg_subordinate_dependents_per_head": float(mean(subordinate_dependents_per_head)) if subordinate_dependents_per_head else 0.0,
                "avg_coordinate_dependents_per_head": float(mean(coordinate_dependents_per_head)) if coordinate_dependents_per_head else 0.0,
                "avg_discourse_density": float(mean(discourse_density)) if discourse_density else 0.0,
                "discourse_density_by_relation": {
                    relation: float(mean(values)) if values else 0.0
                    for relation, values in relation_density.items()
                },
                "mean_surprisal": _weighted_mean(surprisal_means, surprisal_weights),
                "surprisal_variance": _weighted_mean(surprisal_vars, surprisal_weights),
                "coordination_ratio": float(mean(coordination_ratios)) if coordination_ratios else 0.0,
                "subordination_ratio": float(mean(subordination_ratios)) if subordination_ratios else 0.0,
                "normalized_structural_entropy": float(mean(normalized_structural_entropy_series)) if normalized_structural_entropy_series else 0.0,
                "normalized_structural_entropy_std": float(np.std(np.asarray(normalized_structural_entropy_series, dtype=float))) if normalized_structural_entropy_series else 0.0,
                "normalized_structural_entropy_series": normalized_structural_entropy_series,
                "mean_dependency_distance": float(mean(mean_dependency_distance_series)) if mean_dependency_distance_series else 0.0,
                "mean_dependency_distance_series": mean_dependency_distance_series,
                "lexical_density": float(mean(lexical_density_series)) if lexical_density_series else 0.0,
                "lexical_density_series": lexical_density_series,
                "lexical_entropy": float(mean(lexical_entropy_series)) if lexical_entropy_series else 0.0,
                "lexical_entropy_std": float(np.std(np.asarray(lexical_entropy_series, dtype=float))) if lexical_entropy_series else 0.0,
                "lexical_entropy_series": lexical_entropy_series,
                "normalized_pos_ngram_entropy": float(mean(pos_ngram_entropy_series)) if pos_ngram_entropy_series else 0.0,
                "normalized_pos_ngram_entropy_series": pos_ngram_entropy_series,
                "sentence_length_approx_entropy": float(mean(sentence_length_apen_series)) if sentence_length_apen_series else 0.0,
                "sentence_length_approx_entropy_series": sentence_length_apen_series,
                "parataxis_count": parataxis_total,
                "hypotaxis_count": hypotaxis_total,
                "parataxis_to_hypotaxis_ratio": _safe_ratio(parataxis_total, hypotaxis_total),
                "breath_unit_length_mean": float(mean(breath_unit_length_series)) if breath_unit_length_series else 0.0,
                "breath_unit_length_series": breath_unit_length_series,
                "max_depth_series": max_depth_series,
                "sentence_count": float(len(token_counts)),
                "discourse_sentence_count": float(len(discourse_density)),
                "surprisal_weight_total": float(sum(surprisal_weights)) if surprisal_weights else 0.0,
            }
            rows.append(row)

    text_cache: Dict[Tuple[str, str], str] = {}

    def _distribution_text(author_name: str, text_name: str) -> str:
        cache_key = (author_name, text_name)
        if cache_key not in text_cache:
            segmented_path = text_path(
                "processed",
                "cleaned_segmented_texts",
                [author_name, author_name],
                f"{text_name}_cleaned_segmented.jsonl",
            )
            text_cache[cache_key] = " ".join(
                sentence.strip()
                for sentence in _read_segmented_sentences(segmented_path)
                if isinstance(sentence, str) and sentence.strip()
            )
        return text_cache[cache_key]

    nlp = load_spacy_model()
    author_groups: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        author_groups.setdefault(str(row.get("author") or ""), []).append(row)

    for author_name, author_rows in author_groups.items():
        early_rows = [row for row in author_rows if str(row.get("phase") or "") == "early"]
        reference_text = " ".join(
            _distribution_text(author_name, str(row.get("text_name") or ""))
            for row in early_rows
        ).strip()

        for row in author_rows:
            comparison_text = _distribution_text(author_name, str(row.get("text_name") or ""))
            if not reference_text or not comparison_text:
                row["cross_entropy_to_early"] = 0.0
                row["kl_divergence_to_early"] = 0.0
                continue
            divergence = WholeTextMetrics.compare_text_distributions(
                reference_text,
                comparison_text,
                nlp=nlp,
            )
            row["cross_entropy_to_early"] = float(divergence.get("cross_entropy") or 0.0)
            row["kl_divergence_to_early"] = float(divergence.get("kl_divergence") or 0.0)

    return rows


def _maybe_skip(path: Path, use_existing: bool) -> bool:
    return bool(use_existing and path.exists())


def _phase_label_offset(index: int, x_values: Sequence[float], y_values: Sequence[float]) -> Tuple[float, float]:
    x_span = max(x_values) - min(x_values) if x_values else 0.0
    y_span = max(y_values) - min(y_values) if y_values else 0.0
    base_x = max(0.6, x_span * 0.035)
    base_y = max(0.02, y_span * 0.08 if y_span else 0.04)
    offsets = [
        (-base_x, base_y),
        (base_x, base_y * 1.2),
        (base_x, -base_y),
        (-base_x, -base_y),
        (0.0, base_y * 1.45),
        (0.0, -base_y * 1.45),
    ]
    if index < len(offsets):
        return offsets[index]
    return (base_x, base_y)


def _has_numeric_metrics(row: Dict[str, object], *keys: str) -> bool:
    for key in keys:
        value = row.get(key)
        if not isinstance(value, (int, float)):
            return False
        if isinstance(value, float) and math.isnan(value):
            return False
    return True


def _metric_or_zero(row: Dict[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return float(value)
    return 0.0


def _clean_numeric_sequence(values: object) -> List[float]:
    if not isinstance(values, (list, tuple, np.ndarray)):
        return []
    cleaned: List[float] = []
    for value in values:
        if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
            cleaned.append(float(value))
    return cleaned


def _safe_ratio(parataxis_count: float, hypotaxis_count: float) -> float:
    return float((max(0.0, parataxis_count) + 1.0) / (max(0.0, hypotaxis_count) + 1.0))


def _rolling_mean(values: Sequence[float], window_size: int = 50) -> np.ndarray:
    arr = np.asarray(_clean_numeric_sequence(values), dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    window = max(1, min(int(window_size), int(arr.size)))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(arr, kernel, mode="same")


def _resample_series(values: Sequence[float], points: int = 100) -> np.ndarray:
    arr = np.asarray(_clean_numeric_sequence(values), dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    if arr.size == 1:
        return np.repeat(arr[0], points)
    base_x = np.linspace(0.0, 1.0, arr.size)
    target_x = np.linspace(0.0, 1.0, points)
    return np.interp(target_x, base_x, arr)


def _phase_series(rows: Sequence[Dict[str, object]], author: str, phase: str, key: str) -> List[float]:
    collected: List[float] = []
    for row in rows:
        if str(row.get("author") or "") != author or str(row.get("phase") or "") != phase:
            continue
        collected.extend(_clean_numeric_sequence(row.get(key)))
    return collected


def plot_syntax_stretch(
    rows: Sequence[Dict[str, object]],
    output_path: Path,
    *,
    authors: Sequence[str],
    figure_title: str,
    use_existing: bool,
) -> Optional[Path]:
    """Connected scatter: sentence length vs subordinate-clause depth."""
    if _maybe_skip(output_path, use_existing):
        return output_path

    fig, ax = plt.subplots(figsize=(11.8, 7.4))
    plotted_any = False

    for author in authors:
        ordered = _ordered_rows_for_author(rows, author)
        if len(ordered) < 2:
            continue
        plotted_any = True
        x = [float(item.get("avg_sentence_length") or 0.0) for item in ordered]
        y = [float(item.get("avg_subordinate_per_sentence") or 0.0) for item in ordered]
        color = AUTHOR_COLORS.get(author, "#7570b3")
        linestyle = AUTHOR_LINESTYLES.get(author, "-")
        marker = AUTHOR_MARKERS.get(author, "o")

        ax.plot(
            x,
            y,
            marker=marker,
            linewidth=2.0,
            color=color,
            linestyle=linestyle,
            alpha=0.92,
            label=_author_display_name(author),
        )
        for idx, (item, xv, yv) in enumerate(zip(ordered, x, y)):
            point_label = _text_sequence_label(author, str(item.get("text_name") or ""))
            dx, dy = _phase_label_offset(idx, x, y)
            ax.annotate(
                point_label,
                xy=(xv, yv),
                xytext=(xv + dx, yv + dy),
                textcoords="data",
                fontsize=7.5,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.82},
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7, "alpha": 0.55},
            )

    if not plotted_any:
        plt.close(fig)
        return None

    ax.set_xlabel("Sentence Length (tokens per sentence)")
    ax.set_ylabel("Syntactic Complexity (subordinate clauses per sentence)")
    ax.set_title(_compose_figure_title(figure_title, "sentence length to syntactic complexity across 6 texts"))
    ax.grid(alpha=0.2, linestyle="--")

    handles, labels = ax.get_legend_handles_labels()
    dedup: Dict[str, object] = {}
    for handle, label in zip(handles, labels):
        dedup[label] = handle
    if dedup:
        ax.legend(dedup.values(), dedup.keys(), loc="best", frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_clause_density_metrics(
    rows: Sequence[Dict[str, object]],
    output_path: Path,
    *,
    authors: Sequence[str],
    figure_title: str,
    use_existing: bool,
) -> Optional[Path]:
    """Plot saved syntax-density metrics across the six chronological texts."""
    if _maybe_skip(output_path, use_existing):
        return output_path

    metric_specs = [
        ("avg_max_depth", "Avg max dependency depth"),
        ("avg_main_dependents_per_head", "Avg dependents/head (main)"),
        ("avg_subordinate_dependents_per_head", "Avg dependents/head (subordinate)"),
    ]
    fig, axes = plt.subplots(len(metric_specs), 1, figsize=(12.6, 10.2), sharex=True)
    if len(metric_specs) == 1:
        axes = [axes]

    plotted_any = False
    x_base = np.arange(1, 7, dtype=float)
    tick_labels = ["E1", "E2", "M1", "M2", "L1", "L2"]

    for metric_idx, (ax, (metric_key, ylabel)) in enumerate(zip(axes, metric_specs)):
        pooled_x: List[float] = []
        pooled_y: List[float] = []
        for author in authors:
            ordered = _ordered_rows_for_author(rows, author)
            if len(ordered) < 2:
                continue
            plotted_any = True
            x = np.arange(1, len(ordered) + 1, dtype=float)
            y = [float(item.get(metric_key) or 0.0) for item in ordered]
            pooled_x.extend(float(v) for v in x)
            pooled_y.extend(float(v) for v in y)
            ax.plot(
                x,
                y,
                marker=AUTHOR_MARKERS.get(author, "o"),
                linewidth=2.0,
                color=AUTHOR_COLORS.get(author, "#7570b3"),
                linestyle=AUTHOR_LINESTYLES.get(author, "-"),
                alpha=0.92,
                label=_author_display_name(author),
            )

        # Pooled linear trend across all authors for this metric panel.
        if len(pooled_x) >= 2 and len(set(pooled_x)) >= 2:
            x_arr = np.array(pooled_x, dtype=float)
            y_arr = np.array(pooled_y, dtype=float)
            slope, intercept = np.polyfit(x_arr, y_arr, 1)
            trend = slope * x_base + intercept
            trend_label = "Overall linear trend" if metric_idx == 0 else None
            ax.plot(
                x_base,
                trend,
                color="#111111",
                linestyle="--",
                linewidth=1.8,
                alpha=0.9,
                label=trend_label,
            )

        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2, linestyle="--")
        ax.margins(x=0.04)

    if not plotted_any:
        plt.close(fig)
        return None

    axes[-1].set_xticks(x_base, tick_labels)
    axes[-1].set_xlabel("Chronological text order")
    axes[0].set_title(_compose_figure_title(figure_title, "clause tree density across 6 texts"))

    handles, labels = axes[0].get_legend_handles_labels()
    dedup: Dict[str, object] = {}
    for handle, label in zip(handles, labels):
        dedup[label] = handle
    if dedup:
        fig.legend(dedup.values(), dedup.keys(), loc="upper center", ncol=4, frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_parataxis_hypotaxis_divergence(
    rows: Sequence[Dict[str, object]],
    output_path: Path,
    *,
    authors: Sequence[str],
    figure_title: str,
    use_existing: bool,
) -> Optional[Path]:
    """Horizontal early-vs-late dumbbell comparison of parataxis/hypotaxis balance per author."""
    if _maybe_skip(output_path, use_existing):
        return output_path

    phase_rows = _author_phase_rows(rows, authors=authors)
    labels: List[str] = []
    early_values: List[float] = []
    late_values: List[float] = []

    for author in authors:
        phase_map = phase_rows.get(author, {})
        early = phase_map.get("early", {})
        late = phase_map.get("late", {})
        if not early and not late:
            continue
        early_ratio = _safe_ratio(float(early.get("parataxis_count") or 0.0), float(early.get("hypotaxis_count") or 0.0))
        late_ratio = _safe_ratio(float(late.get("parataxis_count") or 0.0), float(late.get("hypotaxis_count") or 0.0))
        labels.append(_author_display_name(author))
        early_values.append(math.log2(max(early_ratio, 1e-6)))
        late_values.append(math.log2(max(late_ratio, 1e-6)))

    if not labels:
        return None

    y_positions = np.arange(len(labels), dtype=float)
    all_values = early_values + late_values
    x_padding = max(0.08, (max(all_values) - min(all_values)) * 0.12 if len(all_values) > 1 else 0.1)
    fig_height = max(4.8, 0.85 * len(labels) + 2.0)
    fig, ax = plt.subplots(figsize=(9.6, fig_height))

    for idx, (y_pos, early_value, late_value) in enumerate(zip(y_positions, early_values, late_values)):
        ax.plot(
            [early_value, late_value],
            [y_pos, y_pos],
            color="#9a9a9a",
            linewidth=2.2,
            alpha=0.9,
            zorder=1,
        )
        ax.scatter(
            early_value,
            y_pos,
            s=84,
            color=PHASE_COLORS["early"],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
            label="Early" if idx == 0 else None,
        )
        ax.scatter(
            late_value,
            y_pos,
            s=84,
            color=PHASE_COLORS["late"],
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
            label="Late" if idx == 0 else None,
        )

    ax.axvline(0.0, color="#111111", linewidth=1.2, linestyle="--", alpha=0.85)
    ax.set_xlim(min(all_values) - x_padding, max(all_values) + x_padding)
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("log₂(parataxis : hypotaxis)")
    ax.set_ylabel("Author")
    ax.set_title(_compose_figure_title(figure_title, "parataxis vs hypotaxis shift (Early → Late)"))
    ax.grid(axis="x", linestyle="--", alpha=0.24)
    ax.legend(frameon=False, loc="best")

    fig.text(
        0.5,
        0.012,
        "Negative values indicate hypotactic dominance; positive values indicate paratactic dominance. The connector shows the early-to-late shift.",
        ha="center",
        fontsize=8.7,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.03, 0.04, 1, 1))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_parataxis_hypotaxis_per_book(
    rows: Sequence[Dict[str, object]],
    output_path: Path,
    *,
    authors: Sequence[str],
    figure_title: str,
    use_existing: bool,
) -> Optional[Path]:
    """Six-text line trajectories of parataxis/hypotaxis balance for each author."""
    if _maybe_skip(output_path, use_existing):
        return output_path

    selected_authors = [author for author in authors if _ordered_rows_for_author(rows, author)]
    if not selected_authors:
        return None

    ncols = min(2, len(selected_authors)) if len(selected_authors) <= 4 else 4
    nrows = int(math.ceil(len(selected_authors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.9 * nrows), sharey=False)
    axes_list = list(np.atleast_1d(axes).flat)
    plotted_any = False

    for ax, author in zip(axes_list, selected_authors):
        ordered = _ordered_rows_for_author(rows, author)
        x_vals: List[float] = []
        y_vals: List[float] = []
        tick_labels: List[str] = []
        phases_for_points: List[str] = []

        for idx, row in enumerate(ordered, start=1):
            ratio = row.get("parataxis_to_hypotaxis_ratio")
            if not isinstance(ratio, (int, float)) or (isinstance(ratio, float) and math.isnan(ratio)) or float(ratio) <= 0.0:
                ratio = _safe_ratio(
                    float(row.get("parataxis_count") or 0.0),
                    float(row.get("hypotaxis_count") or 0.0),
                )
            x_vals.append(float(idx))
            y_vals.append(math.log2(max(float(ratio), 1e-6)))
            tick_labels.append(_author_text_tick_label(author, str(row.get("text_name") or "")))
            phases_for_points.append(str(row.get("phase") or ""))

        if not x_vals:
            ax.text(0.5, 0.5, "No ratio data", ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title(_author_display_name(author))
            ax.set_xticks([])
            ax.grid(axis="y", linestyle="--", alpha=0.22)
            continue

        plotted_any = True
        line_color = AUTHOR_COLORS.get(author, "#4e79a7")
        ax.plot(
            x_vals,
            y_vals,
            color=line_color,
            linewidth=1.9,
            linestyle=AUTHOR_LINESTYLES.get(author, "-"),
            alpha=0.92,
            zorder=2,
        )
        for x_val, y_val, phase in zip(x_vals, y_vals, phases_for_points):
            ax.scatter(
                x_val,
                y_val,
                s=62,
                color=PHASE_COLORS.get(phase, line_color),
                edgecolor="white",
                linewidth=0.85,
                marker=AUTHOR_MARKERS.get(author, "o"),
                zorder=3,
            )

        ax.axhline(0.0, color="#111111", linewidth=1.1, linestyle="--", alpha=0.85)
        ax.set_xticks(x_vals, tick_labels)
        ax.set_xlim(0.6, max(x_vals) + 0.4)
        ax.set_xlabel("Chronological text order")
        ax.set_title(_author_display_name(author))
        ax.grid(axis="y", linestyle="--", alpha=0.24)

    for idx, ax in enumerate(axes_list):
        if idx >= len(selected_authors):
            ax.axis("off")
        elif idx % ncols == 0:
            ax.set_ylabel("log₂(parataxis : hypotaxis)")

    if not plotted_any:
        plt.close(fig)
        return None

    phase_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=PHASE_COLORS[phase], markersize=8, label=PHASE_LABELS[phase])
        for phase in PHASES
    ]
    fig.legend(phase_handles, [handle.get_label() for handle in phase_handles], loc="upper center", bbox_to_anchor=(0.5, 0.98), frameon=False, ncol=3)
    fig.suptitle(_compose_figure_title(figure_title, "parataxis vs hypotaxis across the six texts"), y=0.995)
    fig.text(
        0.5,
        0.01,
        "Negative values indicate hypotactic dominance; positive values indicate paratactic dominance.",
        ha="center",
        fontsize=8.7,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_mean_dependency_distance_slopegraph(
    rows: Sequence[Dict[str, object]],
    output_path: Path,
    *,
    authors: Sequence[str],
    figure_title: str,
    use_existing: bool,
) -> Optional[Path]:
    """Trajectory plot of mean dependency distance across all six chronological texts."""
    if _maybe_skip(output_path, use_existing):
        return output_path

    fig, ax = plt.subplots(figsize=(10.6, 7.0))
    plotted = False
    max_text_count = 0
    all_values: List[float] = []

    for author in authors:
        ordered = _ordered_rows_for_author(rows, author)
        x_vals: List[float] = []
        y_vals: List[float] = []

        for idx, row in enumerate(ordered, start=1):
            if not _has_numeric_metrics(row, "mean_dependency_distance"):
                continue
            x_vals.append(float(idx))
            y_vals.append(float(row.get("mean_dependency_distance") or 0.0))

        if not x_vals:
            continue

        plotted = True
        max_text_count = max(max_text_count, len(x_vals))
        all_values.extend(y_vals)
        color = AUTHOR_COLORS.get(author, "#4e79a7")
        ax.plot(
            x_vals,
            y_vals,
            color=color,
            linewidth=2.0,
            marker=AUTHOR_MARKERS.get(author, "o"),
            linestyle=AUTHOR_LINESTYLES.get(author, "-"),
            alpha=0.88,
            label=_author_display_name(author),
        )
        ax.annotate(
            _author_display_name(author),
            xy=(x_vals[-1], y_vals[-1]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8.0,
            color=color,
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": color, "alpha": 0.82},
        )

    if not plotted:
        plt.close(fig)
        return None

    tick_positions = list(range(1, max_text_count + 1)) or [1]
    tick_labels = [f"T{idx}" for idx in tick_positions]
    phase_labels = ["E1", "E2", "M1", "M2", "L1", "L2"]
    for idx, label in enumerate(phase_labels[: len(tick_labels)]):
        tick_labels[idx] = label

    y_min = min(all_values) if all_values else 0.0
    y_max = max(all_values) if all_values else 1.0
    y_pad = max(0.05, (y_max - y_min) * 0.16 if y_max > y_min else 0.15)

    ax.set_xlim(0.6, max_text_count + 0.4)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xticks(tick_positions, tick_labels)
    ax.set_xlabel("Chronological text order")
    ax.set_ylabel("Mean dependency distance")
    ax.set_title(_compose_figure_title(figure_title, "mean dependency distance across the six texts"))
    ax.grid(axis="y", linestyle="--", alpha=0.24)

    handles, labels = ax.get_legend_handles_labels()
    dedup: Dict[str, object] = {}
    for handle, label in zip(handles, labels):
        dedup[label] = handle
    if dedup:
        ax.legend(dedup.values(), dedup.keys(), frameon=False, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_lexical_density_bubbles(
    rows: Sequence[Dict[str, object]],
    output_path: Path,
    *,
    authors: Sequence[str],
    figure_title: str,
    use_existing: bool,
) -> Optional[Path]:
    """Standalone lexical-density plot across text order, with bubble size reflecting mean sentence length."""
    if _maybe_skip(output_path, use_existing):
        return output_path

    fig, ax = plt.subplots(figsize=(11.8, 7.4))
    plotted_any = False
    max_text_count = 0

    for author in authors:
        ordered = _ordered_rows_for_author(rows, author)
        if not ordered:
            continue
        plotted_any = True
        x = list(range(1, len(ordered) + 1))
        y = [float(item.get("lexical_density") or 0.0) for item in ordered]
        sizes = [max(90.0, 11.0 * float(item.get("avg_sentence_length") or 0.0) ** 1.8) for item in ordered]
        color = AUTHOR_COLORS.get(author, "#7570b3")
        max_text_count = max(max_text_count, len(ordered))

        ax.plot(
            x,
            y,
            linewidth=1.7,
            color=color,
            linestyle=AUTHOR_LINESTYLES.get(author, "-"),
            alpha=0.58,
            label=_author_display_name(author),
        )
        ax.scatter(x, y, s=sizes, color=color, alpha=0.28, edgecolor=color, linewidth=1.2)
        for idx, (item, xv, yv) in enumerate(zip(ordered, x, y)):
            point_label = _text_sequence_label(author, str(item.get("text_name") or "")).splitlines()[0]
            ax.annotate(
                point_label,
                xy=(xv, yv),
                xytext=(0, 8 if idx % 2 == 0 else -10),
                textcoords="offset points",
                fontsize=7.2,
                ha="center",
                va="bottom" if idx % 2 == 0 else "top",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.82},
            )

    if not plotted_any:
        plt.close(fig)
        return None

    tick_positions = list(range(1, max_text_count + 1)) or [1]
    tick_labels = [f"T{idx}" for idx in tick_positions]
    phase_labels = ["E1", "E2", "M1", "M2", "L1", "L2"]
    for idx, label in enumerate(phase_labels[: len(tick_labels)]):
        tick_labels[idx] = label

    ax.scatter([], [], s=max(90.0, 11.0 * 15.0 ** 1.8), color="#666666", alpha=0.22, label="Smaller bubble = shorter mean sentence")
    ax.scatter([], [], s=max(90.0, 11.0 * 25.0 ** 1.8), color="#666666", alpha=0.22, label="Larger bubble = longer mean sentence")
    ax.set_xticks(tick_positions, tick_labels)
    ax.set_xlim(0.6, max_text_count + 0.4)
    ax.set_xlabel("Chronological text order")
    ax.set_ylabel("Lexical density")
    ax.set_title(_compose_figure_title(figure_title, "lexical density across the six texts"))
    ax.grid(alpha=0.2, linestyle="--")

    handles, labels = ax.get_legend_handles_labels()
    dedup: Dict[str, object] = {}
    for handle, label in zip(handles, labels):
        dedup[label] = handle
    if dedup:
        ax.legend(dedup.values(), dedup.keys(), loc="best", frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _plot_scalar_trajectory(
    rows: Sequence[Dict[str, object]],
    output_path: Path,
    *,
    authors: Sequence[str],
    figure_title: str,
    use_existing: bool,
    value_key: str,
    series_key: Optional[str],
    ylabel: str,
    plot_title: str,
    line_color: str,
    empty_message: str,
) -> Optional[Path]:
    if _maybe_skip(output_path, use_existing):
        return output_path

    selected_authors = [author for author in authors if _ordered_rows_for_author(rows, author)]
    if not selected_authors:
        return None

    ncols = min(2, len(selected_authors)) if len(selected_authors) <= 4 else 4
    nrows = int(math.ceil(len(selected_authors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.9 * nrows), sharey=False)
    axes_list = list(np.atleast_1d(axes).flat)
    plotted_any = False

    for ax, author in zip(axes_list, selected_authors):
        ordered = _ordered_rows_for_author(rows, author)
        x_vals: List[float] = []
        y_vals: List[float] = []
        y_errs: List[float] = []
        tick_labels: List[str] = []

        for idx, row in enumerate(ordered, start=1):
            if not _has_numeric_metrics(row, value_key):
                continue
            y_value = _metric_or_zero(row, value_key)
            x_vals.append(float(idx))
            y_vals.append(y_value)
            tick_labels.append(_text_sequence_label(author, str(row.get("text_name") or "")).splitlines()[0])
            series = _clean_numeric_sequence(row.get(series_key)) if series_key else []
            y_errs.append(float(np.std(np.asarray(series, dtype=float))) if len(series) > 1 else 0.0)

        if not x_vals:
            ax.text(0.5, 0.5, empty_message, ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title(_author_display_name(author))
            ax.set_xticks([])
            ax.grid(axis="y", linestyle="--", alpha=0.2)
            continue

        plotted_any = True
        color = AUTHOR_COLORS.get(author, line_color)
        ax.plot(
            x_vals,
            y_vals,
            color=color,
            linewidth=1.9,
            marker=AUTHOR_MARKERS.get(author, "o"),
            linestyle=AUTHOR_LINESTYLES.get(author, "-"),
            alpha=0.9,
        )
        if any(err > 0 for err in y_errs):
            lower = np.asarray(y_vals, dtype=float) - np.asarray(y_errs, dtype=float)
            upper = np.asarray(y_vals, dtype=float) + np.asarray(y_errs, dtype=float)
            ax.fill_between(x_vals, lower, upper, color=color, alpha=0.14)

        ax.set_xticks(x_vals, tick_labels)
        ax.set_xlim(0.6, max(x_vals) + 0.4)
        ax.set_xlabel("Chronological text order")
        ax.set_title(_author_display_name(author))
        ax.grid(axis="y", linestyle="--", alpha=0.22)

    for idx, ax in enumerate(axes_list):
        if idx >= len(selected_authors):
            ax.axis("off")
        elif idx % ncols == 0:
            ax.set_ylabel(ylabel)

    if not plotted_any:
        plt.close(fig)
        return None

    fig.suptitle(_compose_figure_title(figure_title, plot_title), y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_distribution_divergence_bars(
    rows: Sequence[Dict[str, object]],
    output_path: Path,
    *,
    authors: Sequence[str],
    figure_title: str,
    use_existing: bool,
) -> Optional[Path]:
    """Grouped bars for cross-entropy and KL divergence against each author's early baseline."""
    if _maybe_skip(output_path, use_existing):
        return output_path

    selected_authors = [author for author in authors if _ordered_rows_for_author(rows, author)]
    if not selected_authors:
        return None

    ncols = min(2, len(selected_authors)) if len(selected_authors) <= 4 else 4
    nrows = int(math.ceil(len(selected_authors) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 5.1 * nrows), sharey=False)
    axes_list = list(np.atleast_1d(axes).flat)
    plotted_any = False

    for ax, author in zip(axes_list, selected_authors):
        ordered = _ordered_rows_for_author(rows, author)
        if not ordered:
            ax.axis("off")
            continue

        x_vals = np.arange(len(ordered), dtype=float)
        cross_entropy = np.asarray([
            _metric_or_zero(row, "cross_entropy_to_early") for row in ordered
        ], dtype=float)
        kl_divergence = np.asarray([
            _metric_or_zero(row, "kl_divergence_to_early") for row in ordered
        ], dtype=float)
        tick_labels = [
            _text_sequence_label(author, str(row.get("text_name") or "")).splitlines()[0]
            for row in ordered
        ]
        width = 0.38
        plotted_any = True

        ax.bar(x_vals - (width / 2.0), cross_entropy, width=width, color="#f28e2b", alpha=0.84, label="Cross-entropy")
        ax.bar(x_vals + (width / 2.0), kl_divergence, width=width, color="#e15759", alpha=0.78, label="KL divergence")
        ax.set_xticks(x_vals, tick_labels)
        ax.set_xlabel("Chronological text order")
        ax.set_title(_author_display_name(author))
        ax.grid(axis="y", linestyle="--", alpha=0.22)

    for idx, ax in enumerate(axes_list):
        if idx >= len(selected_authors):
            ax.axis("off")
        elif idx % ncols == 0:
            ax.set_ylabel("Divergence from early baseline")

    if not plotted_any:
        plt.close(fig)
        return None

    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(_compose_figure_title(figure_title, "cross entropy and kl divergencevs early baseline"), y=0.995)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _smoothed_density(
    values: Sequence[float],
    *,
    bins: int = 160,
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    clean = np.asarray([float(value) for value in values if isinstance(value, (int, float))], dtype=float)
    if clean.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    hist, edges = np.histogram(clean, bins=bins, range=value_range, density=True)
    kernel_x = np.linspace(-3.0, 3.0, 21)
    kernel = np.exp(-0.5 * kernel_x**2)
    kernel /= kernel.sum()
    smooth = np.convolve(hist, kernel, mode="same")
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, smooth


# Topic-modelling visualisations were moved out of this dictation-only module.
# `src/e1_text_dictation_visualisations.py` now focuses on syntax, rhythm, and lexical form.


def _read_segmented_sentences(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            text = str(obj.get("text") or "").strip()
            if text:
                out.append(text)
    return out


def _dependency_tree_for_sentence(sentence: str) -> Tuple[nx.DiGraph, Dict[int, str], Optional[int]]:
    nlp = load_spacy_model()
    doc = nlp(sentence)

    graph = nx.DiGraph()
    labels: Dict[int, str] = {}
    root_idx: Optional[int] = None

    for token in doc:
        if token.is_space:
            continue
        labels[token.i] = token.text
        graph.add_node(token.i)

    for token in doc:
        if token.is_space:
            continue
        if token.head.i == token.i:
            root_idx = token.i
            continue
        if token.head.i in graph and token.i in graph:
            graph.add_edge(token.head.i, token.i)

    if root_idx is None and len(graph.nodes) > 0:
        # Fallback is deterministic: pick a source node if available, else first token index.
        source_nodes = [node for node in graph.nodes if graph.in_degree(node) == 0]
        root_idx = min(source_nodes) if source_nodes else min(graph.nodes)
    return graph, labels, root_idx


def _hierarchical_layout_left_to_right(
    graph: nx.DiGraph,
    root: int,
    depth_step: float = 1.8,
    leaf_step: float = 1.15,
) -> Dict[int, Tuple[float, float]]:
    """Lay out the dependency tree from left (root) to right (dependents)."""
    pos: Dict[int, Tuple[float, float]] = {}
    visited: set[int] = set()
    next_leaf_y = 0.0

    def place(node: int, depth: int) -> float:
        nonlocal next_leaf_y
        if node in visited:
            return pos[node][1]

        visited.add(node)
        children = [child for child in sorted(graph.successors(node)) if child not in visited]

        if not children:
            y = next_leaf_y
            next_leaf_y -= leaf_step
        else:
            child_ys = [place(child, depth + 1) for child in children]
            y = float(sum(child_ys) / len(child_ys))

        pos[node] = (depth * depth_step, y)
        return y

    if root in graph:
        place(root, 0)

    extra_depth = int(max((x for x, _ in pos.values()), default=0.0) / depth_step) + 1
    for node in sorted(graph.nodes):
        if node in visited:
            continue
        place(node, extra_depth)
        extra_depth += 1

    if pos:
        ys = [y for _, y in pos.values()]
        y_mid = (max(ys) + min(ys)) / 2.0
        pos = {node: (x, y - y_mid) for node, (x, y) in pos.items()}

    return pos


def _max_tree_depth(graph: nx.DiGraph, root: int) -> int:
    """Return max hop-depth from root in the dependency tree."""
    if root not in graph:
        return 0
    depths = nx.single_source_shortest_path_length(graph, root)
    if not depths:
        return 0
    return int(max(depths.values()))


def plot_clausal_nesting_overlay(
    rows: Sequence[Dict[str, object]],
    output_path: Path,
    *,
    authors: Sequence[str],
    figure_title: str,
    use_existing: bool,
) -> Optional[Path]:
    """Color-coded max-depth overlays in an author-by-phase facet grid."""
    if _maybe_skip(output_path, use_existing):
        return output_path

    if not authors:
        return None

    by_author = {author: _ordered_rows_for_author(rows, author) for author in authors}
    max_texts = max((len(author_rows) for author_rows in by_author.values()), default=0)
    if max_texts == 0:
        return None

    fig, axes = plt.subplots(len(authors), max_texts, figsize=(3.0 * max_texts, 3.1 * len(authors)), sharex=True, sharey=True)
    if len(authors) == 1 and max_texts == 1:
        axes = np.array([[axes]])
    elif len(authors) == 1:
        axes = np.array([axes])
    elif max_texts == 1:
        axes = np.array([[ax] for ax in axes])
    cmap = plt.get_cmap("magma")
    all_depths = [
        float(depth)
        for author in authors
        for row in by_author.get(author, [])
        for depth in (row.get("max_depth_series") or [])
        if isinstance(depth, (int, float))
    ]
    vmin = min(all_depths) if all_depths else 0.0
    vmax = max(all_depths) if all_depths else 1.0
    scatter_handle = None

    for row_idx, author in enumerate(authors):
        author_rows = by_author.get(author, [])
        for col_idx in range(max_texts):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(author_rows):
                ax.axis("off")
                continue

            row = author_rows[col_idx]
            series = [float(v) for v in (row.get("max_depth_series") or [])]
            if not series:
                ax.text(0.5, 0.5, "No depth series", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                ax.set_xlim(0.0, 100.0)
                ax.grid(alpha=0.2, linestyle="--")
                continue
            x = np.linspace(0.0, 100.0, num=len(series), endpoint=True)
            colors = np.array(series)
            scatter_handle = ax.scatter(
                x,
                series,
                c=colors,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=3,
                alpha=0.36,
                marker="o",
                linewidths=0,
            )

            if row_idx == 0:
                ax.set_title(
                    _text_sequence_label(author, str(row.get("text_name") or "")),
                    fontsize=10,
                )
            if col_idx == 0:
                ax.set_ylabel(f"{_author_display_name(author)}\nMax dependency depth")
            if row_idx == len(authors) - 1:
                ax.set_xlabel("Text progress (%)")
            ax.set_xlim(0.0, 100.0)
            ax.grid(alpha=0.2, linestyle="--")

    if scatter_handle is not None:
        cax = fig.add_axes([0.91, 0.12, 0.015, 0.76])
        colorbar = fig.colorbar(scatter_handle, cax=cax)
        colorbar.set_label("Max dependency depth")

    fig.suptitle(_compose_figure_title(figure_title, "Clausal Nesting Overlay Across 6 Texts"), y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(top=0.92, right=0.89, wspace=0.16, hspace=0.28)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def generate_all_dictation_visualisations(
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    use_existing: bool = DEFAULT_USE_EXISTING,
    output_root: Optional[Path] = None,
) -> Dict[str, object]:
    """Generate the dictation-emphasis visualisation suite."""
    window_root = analytics_path("window")
    if output_root is None:
        output_root = results_path("figures", subfolder="dictation_emphasis", block_size=block_size)

    rows = _collect_text_rows(window_root, block_size=block_size)
    if not rows:
        empty_outputs = {
            part_key: {
                "syntax_stretch": None,
                "syntax_stretch_by_author": {},
                "clause_density_metrics": None,
                "parataxis_hypotaxis_divergence": None,
                "parataxis_hypotaxis_per_book": None,
                "mean_dependency_distance_slopegraph": None,
                "lexical_density_bubbles": None,
                "distribution_divergence_bars": None,
            }
            for part_key in PART_SPECS
        }
        return empty_outputs

    outputs: Dict[str, object] = {}
    for part_key, part_spec in PART_SPECS.items():
        authors = tuple(str(author) for author in part_spec.get("authors", ()))
        figure_title = str(part_spec.get("title") or part_key)
        part_root = output_root / part_key

        syntax_stretch_by_author: Dict[str, Optional[Path]] = {}
        for author in authors:
            syntax_stretch_by_author[author] = plot_syntax_stretch(
                rows,
                part_root / "syntax_stretch_by_author" / f"dictation_sentence_length_to_syntactic_complexity_{author}.png",
                authors=(author,),
                figure_title=f"{figure_title}\n{_author_display_name(author)}",
                use_existing=use_existing,
            )

        outputs[part_key] = {
            "syntax_stretch": plot_syntax_stretch(
                rows,
                part_root / "dictation_sentence_length_to_syntactic_complexity_across_6_texts.png",
                authors=authors,
                figure_title=figure_title,
                use_existing=use_existing,
            ),
            "syntax_stretch_by_author": syntax_stretch_by_author,
            "clause_density_metrics": plot_clause_density_metrics(
                rows,
                part_root / "dictation_clause_tree_density_across_6_texts.png",
                authors=authors,
                figure_title=figure_title,
                use_existing=use_existing,
            ),
            "parataxis_hypotaxis_divergence": plot_parataxis_hypotaxis_divergence(
                rows,
                part_root / "dictation_parataxis_hypotaxis_diverging_bars.png",
                authors=authors,
                figure_title=figure_title,
                use_existing=use_existing,
            ),
            "parataxis_hypotaxis_per_book": plot_parataxis_hypotaxis_per_book(
                rows,
                part_root / "dictation_parataxis_hypotaxis_across_6_texts.png",
                authors=authors,
                figure_title=figure_title,
                use_existing=use_existing,
            ),
            "mean_dependency_distance_slopegraph": plot_mean_dependency_distance_slopegraph(
                rows,
                part_root / "dictation_mean_dependency_distance_slopegraph.png",
                authors=authors,
                figure_title=figure_title,
                use_existing=use_existing,
            ),
            "lexical_density_bubbles": plot_lexical_density_bubbles(
                rows,
                part_root / "dictation_lexical_density_bubble_chart.png",
                authors=authors,
                figure_title=figure_title,
                use_existing=use_existing,
            ),
            "distribution_divergence_bars": plot_distribution_divergence_bars(
                rows,
                part_root / "dictation_cross_entropy_kl_divergence_across_6_texts.png",
                authors=authors,
                figure_title=figure_title,
                use_existing=use_existing,
            ),
        }

    return outputs


if __name__ == "__main__":
    generate_all_dictation_visualisations()
