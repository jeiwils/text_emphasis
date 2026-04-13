"""Central configuration defaults for text_emphasis.

Defaults defined here are the single source of truth for runtime arguments and docs.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

DEFAULT_USE_EXISTING = True
DEFAULT_RNG_SEED: int = 42

DEFAULT_BLOCK_SIZE: int = 5
DEFAULT_DASHBOARD_PERMUTATIONS: int = 1000
LOOP_BLOCK_SIZES_ENABLED = False
LOOP_BLOCK_SIZES = (3, 5, 7)

MODEL_CONFIGS = {
    "causal_lm": "gpt2",
    "sentence_embedding": "all-MiniLM-L6-v2",
}

# Default spaCy pipeline configuration
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_SPACY_DISABLE: Sequence[str] = ()
DEFAULT_SPACY_MAX_LENGTH: int = 5_000_000
# Shared window size (in sentences) for sliding window metrics
DEFAULT_WINDOW_SIZE: int = 3
# Default stride (in sentences) for sliding window metrics
DEFAULT_METRIC_WINDOW_STRIDE: int = 1
# Topic model windows use base_window_size * multiple
DEFAULT_TOPIC_WINDOW_MULTIPLE: int = 5
# Topic window stride uses base_window_size * stride_multiple
DEFAULT_TOPIC_WINDOW_STRIDE_MULTIPLE: int = 2
# Shared HDBSCAN defaults for corpus-wide novel topic modelling.
DEFAULT_TOPIC_MIN_CLUSTER_SIZE: Optional[int] = 12
DEFAULT_TOPIC_MIN_SAMPLES: Optional[int] = 3
# Default soft topic-score filtering for topic modelling + dashboard
DEFAULT_SOFT_SCORE_THRESHOLD: float = 0.08
DEFAULT_SOFT_TOP_K: int = 2
# Short-text fallback (low window count) to avoid empty topic scores.
DEFAULT_SHORT_TEXT_WINDOW_COUNT: int = 10
DEFAULT_SHORT_TEXT_SOFT_SCORE_THRESHOLD: float = 0.25
DEFAULT_SHORT_TEXT_SOFT_TOP_K: int = 3
DEFAULT_SHORT_TEXT_MIN_CLUSTER_SIZE: int = 2
DEFAULT_SHORT_TEXT_MIN_SAMPLES: int = 1
DEFAULT_SHORT_TEXT_WINDOW_MULTIPLE: int = 4
DEFAULT_USE_PCA: bool = False
DEFAULT_PCA_COMPONENTS: int = 50
DEFAULT_TOPIC_KEYWORD_TOP_N: int = 8
DEFAULT_TOPIC_KEYWORD_NGRAM_RANGE: Tuple[int, int] = (1, 3)

DEFAULT_MATTR_WINDOW_SIZE: int = 50
DEFAULT_CONCEPT_TOP_N: int = 100

# Ordered author directories for the 6x8 TXT corpus.
TXT_AUTHOR_DIRS: Tuple[str, ...] = (
    "james",
    "conrad",
    "scott",
    "stevenson",
    "huxley",
    "tarkington",
    "collins",
    "hearn",
)


TXT_BOOK_CONFIGS = {
    # ==========================================
    # WILKIE COLLINS
    # ==========================================
    "wilkie_collins_the_antonina_1850": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "wilkie_collins_the_legacy_of_cain_1889": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "wilkie_collins_the_moonstone_1868": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "wilkie_collins_basil_1852": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "wilkie_collins_heart_and_science_1883": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "wilkie_collins_man_and_wife_1870": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },

    # ==========================================
    # JOSEPH CONRAD
    # ==========================================
    "joseph_conrad_almayers_folly_1895": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "joseph_conrad_lord_jim_1900": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "joseph_conrad_nostromo_1904": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "joseph_conrad_the_arrow_of_gold_1919": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "joseph_conrad_the_rover_1923": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "joseph_conrad_the_secret_agent_1907": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },

    # ==========================================
    # ALDOUS HUXLEY
    # ==========================================
    "aldous_huxley_antic_hay_1923": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "aldous_huxley_brave_new_world_1932": {
        "start_marker": "Chapter One \n\n\n\nA SQUAT grey building",
        "patterns": [
            r"^\s*\d+\s*$",
        ],
    },
    "aldous_huxley_crome_yellow_1921": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "aldous_huxley_island_1962": {
        "start_marker": "Chapter One ",
        "patterns": [
            r"^\s*\d+\s*$",
        ],
    },
    "aldous_huxley_point_counter_point_1928": {
        "start_marker": "POINT COUNTER POINT\n\n                                   By\n                             ALDOUS HUXLEY",
        "end_marker": "[The end of _Point Counter Point",
    },
    "aldous_huxley_the_genius_and_the_goddess_1955": {
        "start_marker": "“The trouble with fiction,” said John Rivers",
        "end_marker": "[The end of _The Genius and the Goddess",
    },

    # ==========================================
    # HENRY JAMES
    # ==========================================
    "henry_james_the_ambassadors_1903": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "henry_james_the_awkward_age_1899": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "henry_james_the_portrait_of_a_lady_1881": {
        "start_marker": "PREFACE \n\n• \n\nThe  Portrait  of  a  Lady",
        "patterns": [
            r"^\s*\d+\s*$",
        ],
    },
    "henry_james_the_american_1877": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "henry_james_the_golden_bowl_1904": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "henry_james_the_spoils_of_poynton_1897": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },

    # ==========================================
    # LAFCADIO HEARN
    # ==========================================
    "lafciadio_hearn_chita_a_memory_of_last_island_1889": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "lafciadio_hearn_in_ghostly_japan_1899": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
        "patterns": [
            r"\[Illustration.*?\]",
        ],
    },
    "lafciadio_hearn_kokoro_hints_and_echoes_of_japanese_inner_life_1896": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "lafciadio_hearn_kwaidan_stories_and_studies_of_strange_things_1904": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "lafciadio_hearn_the_romance_of_the_milky_way_1905": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "lafciadio_hearn_youma_the_story_of_a_west_indian_slave_1890": {
        "start_marker": "THE da, during old colonial days, of-",
        "end_marker": "suffrage to the slaves of \nMartinique.",
        "patterns": [
            r"^\s*\d+\s*$",
            r"Youma\.\s*\d+",
        ],
    },

    # ==========================================
    # WALTER SCOTT
    # ==========================================
    "walter_scott_a_legend_of_montrose_1819": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "walter_scott_castle_dangerous_1831": {
        "start_marker": "CASTLE DANGEROUS.\r\n\r\nCHAPTER THE FIRST.",
        "end_marker": "END OF CASTLE DANGEROUS.",
    },
    "walter_scott_count_robert_of_paris_1831": {
        "start_marker": "COUNT ROBERT OF PARIS.",
        "end_marker": None,
    },
    "walter_scott_the_antiquary_1816": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "walter_scott_the_bride_of_lammermoor_1819": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "walter_scott_waverley_1814": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
        "patterns": [
            r"\[Illustration.*?\]",
        ],
    },

    # ==========================================
    # ROBERT LOUIS STEVENSON
    # ==========================================
    "robert_louis_stevenson_st_ives_1897": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "robert_louis_stevenson_the_master_of_ballantrae_1889": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
        "patterns": [
            r"\[Illustration.*?\]",
        ],
    },
    "robert_louis_stevenson_the_wrecker_1892": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "robert_louis_stevenson_treasure_island_1883": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "robert_louis_stevenson_weir_of_hermiston_1896": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
        "patterns": [
            r"\[Picture.*?\]",
            r"\[Illustration.*?\]",
        ],
    },
    "robert_louis_stevenson_kidnapped_1886": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },

    # ==========================================
    # BOOTH TARKINGTON
    # ==========================================
    "booth_tarkington_kate_fennigate_1943": {
        "start_marker": None,
        "end_marker": "[The end of _Kate Fennigate_",
    },
    "booth_tarkington_the_gentleman_from_indiana_1899": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
    "booth_tarkington_the_heritage_of_hatcher_ide_1941": {
        "start_marker": None,
        "end_marker": "[The end of _The Heritage of Hatcher Ide_",
    },
    "booth_tarkington_the_magnificent_ambersons_1918": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
        "patterns": [
            r"\[Illustration.*?\]",
        ],
    },
    "booth_tarkington_the_plutocrat_1927": {
        "start_marker": None,
        "end_marker": "[The end of _The Plutocrat_",
    },
    "booth_tarkington_alice_adams_1921": {
        "start_marker": "*** START OF THE PROJECT GUTENBERG EBOOK",
        "end_marker": "*** END OF THE PROJECT GUTENBERG EBOOK",
    },
}

# Fallback TXT config when no book-specific settings are provided.
DEFAULT_TEXT_BOOK_CONFIG = {
    "start_marker": None,
    "end_marker": None,
    "patterns": None,
}

# Fallback PDF config when no book-specific settings are provided.
DEFAULT_BOOK_CONFIG = {
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
            "parataxis_hypotaxis",
            "avg_dependents_per_head",
            "avg_mean_dependency_distance",
            "avg_tokens_per_sentence",
            "median_depth",
            "max_depth",
            "depth_skew",
            "structural_entropy",
            "normalized_structural_entropy",
            "punctuation_per_token",
            "breath_unit_counts",
            "breath_unit_per_1000_words",
        },
        "nested_keys": {
            "clause_counts_per_token",
            "clause_ratios",
            "parataxis_hypotaxis",
            "avg_dependents_per_head",
            "breath_unit_counts",
            "breath_unit_per_1000_words",
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

WINDOW_METRIC_DOMAINS = ("syntax", "lexico_semantics", "discourse", "log_prob")
CENTRALITY_METRICS = ("coherence", "exclusivity", "prevalence", "persistence", "top10_mean")

TOPIC_CORRELATIONS_SUFFIX = "_topic_correlations.json"
CENTRAL_TOPIC_CORRELATIONS_SUFFIX = "_central_topic_correlations.json"
TOPIC_PRESENCE_CORRELATIONS_SUFFIX = "_topic_presence_correlations.json"
CENTRAL_TOPIC_PRESENCE_CORRELATIONS_SUFFIX = "_central_topic_presence_correlations.json"
NON_CENTRAL_TOPIC_PRESENCE_CORRELATIONS_SUFFIX = "_non_central_topic_presence_correlations.json"
WINDOW_VARIANCE_SUFFIX = "_window_variances.json"

GENRE_CENTRAL_PRESENCE_FILENAME = "00_genre_central_topic_presence_correlations.json"
TOPIC_SPLIT_HALF_FILENAME = "00_topic_split_half_stability.json"
CENTRAL_TOPIC_SPLIT_HALF_FILENAME = "00_central_topic_split_half_stability.json"
GENRE_TOPIC_SPLIT_HALF_FILENAME = "00_genre_topic_split_half_stability.json"
GENRE_CENTRAL_TOPIC_SPLIT_HALF_FILENAME = "00_genre_central_topic_split_half_stability.json"
AUTHOR_TOPIC_SPLIT_HALF_TEMPLATE = "{author}_topic_split_half_stability.json"
AUTHOR_CENTRAL_TOPIC_SPLIT_HALF_TEMPLATE = "{author}_central_topic_split_half_stability.json"
CENTRAL_PRESENCE_SPLIT_HALF_FILENAME = "00_central_topic_presence_split_half_stability.json"
GENRE_CENTRAL_PRESENCE_SPLIT_HALF_FILENAME = "00_genre_central_topic_presence_split_half_stability.json"
AUTHOR_CENTRAL_PRESENCE_SPLIT_HALF_TEMPLATE = "{author}_central_topic_presence_split_half_stability.json"
CROSS_BLOCK_CONSISTENCY_FILENAME = "00_cross_block_consistency.json"

TABLE_COUNT_COLUMNS = {"topic_id", "n", "text_count", "total_windows", "n_windows"}
TABLE_P_VALUE_COLUMNS = {"p_value"}
TABLE_DEFAULT_FLOAT_DECIMALS = 3
TABLE_P_VALUE_MIN_DISPLAY = 0.001

DEFAULT_CENTRAL_PRESENCE_P = 2.0
DEFAULT_CENTRAL_PRESENCE_NORMALIZE = True
DEFAULT_PRESENCE_K_REFERENCE = "central"
DEFAULT_CENTRAL_TOPIC_NEAR_TOP_ALPHA = 0.8
DEFAULT_CENTRALITY_TOP_SCORE_FRACTION = 0.1
DEFAULT_CENTRALITY_COHERENCE_FLOOR = 0.25
DEFAULT_CENTRALITY_EXCLUSIVITY_FLOOR = 0.25

@dataclass(frozen=True)
class DashboardCorrelationConfig:
    block_size: int = DEFAULT_BLOCK_SIZE
    permutations: int = DEFAULT_DASHBOARD_PERMUTATIONS
    loop_enabled: bool = LOOP_BLOCK_SIZES_ENABLED
    loop_block_sizes: Sequence[int] = LOOP_BLOCK_SIZES
    loop_output_template: str = "dashboard_L{block_size}"

@dataclass(frozen=True)
class CentralTopicSelectionConfig:
    near_top_alpha: float = DEFAULT_CENTRAL_TOPIC_NEAR_TOP_ALPHA
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
    fixed_exemplars: Sequence[Dict[str, object]] = ()


@dataclass(frozen=True)
class PresenceSlopegraphConfig:
    p_threshold: float = 0.01
    fig_width: float = 9.0
    min_height: float = 3.5
    row_height: float = 0.4
    positive_color: str = "#2c7fb8"
    negative_color: str = "#d95f0e"


@dataclass(frozen=True)
class PresenceComparisonScatterConfig:
    metrics: Sequence[str] = (
        "variance.syntax_rms_z",
        "variance.lexico_semantics_rms_z",
        "variance.discourse_rms_z",
        "variance.log_prob_rms_z",
        "variance.overall_rms_z",
    )
    cols: int = 3
    panel_width: float = 4.2
    panel_height: float = 3.4
    point_size: float = 20.0
    point_alpha: float = 0.7
    cmap_name: str = "tab10"
    axis_limits: Optional[Tuple[float, float]] = (-1.0, 1.0)
    show_identity_line: bool = True
    show_zero_lines: bool = True
    identity_line_color: str = "#9e9e9e"
    zero_line_color: str = "#d0d0d0"
    label_max_len: int = 36
    legend_max_cols: int = 3
    legend_fontsize: float = 8.0
    title: Optional[str] = "All-topic vs central-topic presence (RMZ correlations)"
    equal_aspect: bool = True


@dataclass(frozen=True)
class PresenceComparisonPanelConfig:
    metric: str = "variance.overall_rms_z"
    normalize: bool = True
    normalization: str = "zscore"
    fig_width: float = 11.0
    fig_height: float = 6.2
    line_alpha: float = 0.85
    central_color: str = "#222222"
    all_color: str = "#4c78a8"
    variance_color: str = "#d95f02"
    delta_color: str = "#1b9e77"
    central_label: str = "central topic presence"
    all_label: str = "all topic presence"
    variance_label: str = "variance (overall rms z)"
    delta_label: str = "central - all"
    show_zero_line: bool = True
    title: Optional[str] = None


@dataclass(frozen=True)
class PresenceTimelinePanelConfig:
    metric: str = "variance.overall_rms_z"
    normalize: bool = True
    normalization: str = "zscore"
    fig_width: float = 12.0
    fig_height: float = 5.5
    central_color: str = "#222222"
    variance_color: str = "#d95f02"
    topic_colors: Sequence[str] = ("#1b9e77", "#7570b3", "#66a61e")
    line_alpha: float = 0.85
    marker_alpha: float = 0.85
    marker_size: float = 28.0
    highlight_percentile: float = 0.9
    highlight_min_count: int = 3
    label_max_len: int = 48
    show_keywords: bool = True
    title: Optional[str] = None


@dataclass(frozen=True)
class TopicCentralityQuadrantConfig:
    metric: str = "variance.overall_rms_z"
    use_abs: bool = True
    centrality_threshold: Optional[float] = None
    variance_threshold: Optional[float] = None
    fig_width: float = 7.8
    fig_height: float = 6.2
    point_size: float = 55.0
    point_alpha: float = 0.85
    central_color: str = "#d95f02"
    non_central_color: str = "#1b9e77"
    line_color: str = "#8c8c8c"
    line_style: str = "--"
    line_width: float = 1.0
    regression_line: bool = True
    regression_color: str = "#4c4c4c"
    regression_style: str = "-"
    regression_width: float = 1.6
    label_max_len: int = 36
    show_labels: bool = True
    show_keywords: bool = True
    title: Optional[str] = None


@dataclass(frozen=True)
class QualitativePassageTimelineConfig:
    metric: str = "variance.overall_rms_z"
    normalize: bool = True
    normalization: str = "zscore"
    top_n_per_topic: int = 3
    excerpt_sentences: int = 2
    excerpt_max_len: int = 140
    fig_width: float = 12.5
    fig_height: float = 6.5
    marker_size: float = 40.0
    marker_alpha: float = 0.9
    line_alpha: float = 0.85
    topic_colors: Sequence[str] = ("#1b9e77", "#7570b3", "#66a61e")
    variance_color: str = "#d95f02"
    annotation_offset: float = 0.35
    title: Optional[str] = None


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
    "discourse.explicit_connectives_per_token",
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
class AuthorDispersionConfig:
    metrics: Sequence[str] = CORE_SIGNATURE_METRICS
    author_targets: Sequence[Dict[str, object]] = ()
    sd_fig_width: float = 8.5
    sd_min_height: float = 3.5
    sd_row_height: float = 0.4
    dist_fig_width: float = 9.0
    dist_fig_height: float = 4.5
    bar_color: str = "#2c7fb8"
    box_color: str = "#9ecae1"
    point_size: float = 0.0
    point_alpha: float = 0.7
    label_max_len: int = 45
    distribution: str = "box"


@dataclass(frozen=True)
class InternalStabilityRankConfig:
    fig_width: float = 11.0
    min_height: float = 4.0
    row_height: float = 0.4
    rate_color: str = "#2c7fb8"
    delta_color: str = "#9ecae1"
    label_max_len: int = 45
    annotate: bool = True
    label_font_size: int = 8
    value_font_size: int = 7


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
    normalize: bool = True
    normalize_mode: str = "total"
    as_percent: bool = True


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
    include_all_presence: bool = False
    all_presence_color: str = "#4c78a8"
    all_presence_style: str = "--"
    label_max_len: int = 45
    max_xticks: int = 12
    presence_metric_targets: Sequence[Dict[str, object]] = ()


@dataclass(frozen=True)
class TopicGraphConfig:
    text_targets: Sequence[str] = ()
    top_k_edges: int = 3
    min_similarity: float = 0.05
    fig_width: float = 9.5
    fig_height: float = 7.0
    layout_seed: int = 42
    node_size_base: float = 220.0
    node_size_scale: float = 900.0
    central_color: str = "#d95f02"
    non_central_color: str = "#1b9e77"
    node_border_color: str = "#1a1a1a"
    node_border_width: float = 0.8
    node_border_metric: Optional[str] = "variance.overall_rms_z"
    node_border_metric_abs: bool = True
    node_border_width_min: float = 0.8
    node_border_width_max: float = 2.4
    label_font_size: float = 7.5
    label_color: str = "#111111"
    label_max_len: int = 36
    edge_color: str = "#bdbdbd"
    edge_alpha: float = 0.6
    edge_width_min: float = 0.35
    edge_width_scale: float = 2.4


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
DEFAULT_PRESENCE_COMPARISON_SCATTER_CONFIG = PresenceComparisonScatterConfig()
DEFAULT_PRESENCE_COMPARISON_PANEL_CONFIG = PresenceComparisonPanelConfig()
DEFAULT_PRESENCE_TIMELINE_PANEL_CONFIG = PresenceTimelinePanelConfig()
DEFAULT_TOPIC_CENTRALITY_QUADRANT_CONFIG = TopicCentralityQuadrantConfig()
DEFAULT_CONVERGENCE_INDEX_CONFIG = ConvergenceIndexConfig()
DEFAULT_AGGREGATED_HEATMAP_CONFIG = AggregatedHeatmapConfig()
DEFAULT_TOPIC_METRIC_HEATMAP_CONFIG = TopicMetricHeatmapConfig()
DEFAULT_FOREST_PLOT_CONFIG = ForestPlotConfig()
DEFAULT_AUTHOR_DISPERSION_CONFIG = AuthorDispersionConfig()

DEFAULT_INTERNAL_STABILITY_RANK_CONFIG = InternalStabilityRankConfig()
DEFAULT_TEXT_METRIC_HEATMAP_CONFIG = TextMetricHeatmapConfig()
DEFAULT_CENTRAL_TOPIC_WINDOW_HEATMAP_CONFIG = CentralTopicWindowHeatmapConfig()
DEFAULT_STABILITY_FILTER_CONFIG = StabilityFilterConfig()
DEFAULT_STABILITY_STACKED_BAR_CONFIG = StabilityStackedBarConfig()
DEFAULT_TOPIC_METRIC_LINE_CONFIG = TopicMetricLineConfig()

DEFAULT_TOPIC_GRAPH_CONFIG = TopicGraphConfig()
DEFAULT_DATA_SELECTION_CONFIG = DataSelectionConfig(authors=tuple(TXT_AUTHOR_DIRS))

CONVERGENCE_METRIC_LABELS = {
    "significant_count": "Significant metrics (proportion)",
    "mean_abs_r": "Mean |r|",
    "mean_abs_r_zeroed": "Mean |r| (nonsig=0)",
    "sign_agreement": "Sign agreement (proportion)",
}


