
"""



########### SENTENCE BY SENTENCE 

{ 
  "text": "<original_sentence>",                       
  "source": "<source_text_or_corpus>",                  
  "sentence_id": "<unique_sentence_id>",               

  

    "syntax": { 
        "embedding": {                
            "max_depth": "<max_syntactic_depth>",
            "mean_depth": "<mean_syntactic_depth>", 
            "median_depth": "<median_syntactic_depth>"
            "depth_skew": "<mean_minus_median_depth>"
        },
        "clause_metrics": {
            "counts": {
                "main": "<count>",
                "subordinate": "<count>",
                "coordinate": "<count>"
            },
            "ratios": {
                "subordination_ratio": "<subordinate_to_main_ratio>"
                "coordinoation_ratio": "<coordinate_to_main_ratio>"
            }
        "dependency_complexity": { 
            "avg_dependents_per_head": {
                "main_clause": "<mean_dependents_main>",
                "subordinate_clause": "<mean_dependents_subordinate>",
                "coordinate_clause": "<mean_dependents_coordinate>"
            },
            "max_dependents_per_head": "<max_dependents_any_head>",
            "mean_dependency_distance": "<avg_linear_head_dependent_distance>"
        }
    },


  
    


    "lexico_semantics": {
        "vocabulary": { # I NEED TO PASS ANALYTICS ABOUT THE WHOLE CORPUS TO CALCULATE THESE
            "mattr_score": "<moving_average_type_token_ratio>", # proxy for lexical diversity 
            "avg_word_freq": "<mean_corpus_frequency>", # proxy for vocabulary rareness - in relation to whole corpus (i.e individual story)
            "content_function_ratio": "<content_words / total>" # proxy for density of informational content vs descriptive
        },
        "information_content": {
            "mean_surprisal": "<avg_llm_logprob_or_entropy>",
            "surprisal_variance": "<variance_llm_logprob>"
        },
        "semantic_structures": [ 
            {
            "clause_level": "<main/subordinate/coordinate>",
            "predicate": "<main_verb_lemma>",
            "agent": "<subject_phrase_or_token>",
            "patient": "<object_phrase_or_token>",
            "clause_tokens": ["<list_of_tokens_in_clause>"]
            }
        ]
        }




    "discourse": { # according to PDTB
        "relations": [
            {
            "level1": "<Temporal | Contingency | Comparison | Expansion>",
            "level2": "<Cause | Condition | Contrast | Concession | Conjunction | Instantiation | Restatement | Alternative | Exception | ...>",
            "level3": "<Reason | Result | Purpose | etc_or_null>",

            "arg1_span": "<text_span_or_reference>",
            "arg2_span": "<text_span_or_reference>",
            "connective": "<explicit_marker_or_null>",
            "explicit": "<true | false>",
            "direction": "<forward | backward | bidirectional>"
            }
        ]
    }
}











#### SLIDING WINDOW

clause density per 100 tokens 







"""