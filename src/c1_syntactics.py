import spacy
import statistics





"""



check echkec cehkce 


    "syntax": { 
        "embedding": {                
            "max_depth": "<max_syntactic_depth>",
            "avg_depth": "<average_syntactic_depth>",
            "median_depth": "<median_syntactic_depth>"
        },
        "clause_metrics": {
            "counts": {
                "main": "<count>",
                "subordinate": "<count>",
                "coordinate": "<count>"
            },
            "ratios": {
                "clause_density": "<clauses_per_100_tokens>",
                "subordination_ratio": "<subordinate_to_main_ratio>"
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




"""

class SyntaxAnalyzer:
    def __init__(self, model='en_core_web_sm'):
        self.nlp = spacy.load(model)

    # -------------------------------
    # 1. Embedding metrics
    # -------------------------------
    def compute_embedding(self, doc):
        def token_depth(token):
            depth = 0
            while token.head != token:
                depth += 1
                token = token.head
            return depth

        all_depths = [token_depth(token) for token in doc]

        return {
            "max_depth": max(all_depths) if all_depths else 0,
            "avg_depth": round(statistics.mean(all_depths), 2) if all_depths else 0,
            "median_depth": statistics.median(all_depths) if all_depths else 0
        }

    # -------------------------------
    # 2. Clause metrics
    # -------------------------------
    def compute_clause_metrics(self, doc):
        main_counts = 0
        sub_counts = 0
        coord_counts = 0

        for sent in doc.sents:
            for token in sent:
                if token.dep_ == 'ROOT':
                    main_counts += 1
                elif 'advcl' in token.dep_ or 'ccomp' in token.dep_ or 'xcomp' in token.dep_:
                    sub_counts += 1
                elif 'conj' in token.dep_:
                    coord_counts += 1

        total_tokens = len(doc)
        clause_density = (main_counts + sub_counts + coord_counts) / total_tokens * 100 if total_tokens else 0
        sub_to_main_ratio = sub_counts / main_counts if main_counts else 0

        return {
            "counts": {
                "main": main_counts,
                "subordinate": sub_counts,
                "coordinate": coord_counts
            },
            "ratios": {
                "clause_density": round(clause_density, 2),
                "subordination_ratio": round(sub_to_main_ratio, 2)
            }
        }

    # -------------------------------
    # 3. Dependency complexity
    # -------------------------------
    def compute_dependency_complexity(self, doc):
        dependents_per_head = {
            "main_clause": [],
            "subordinate_clause": [],
            "coordinate_clause": []
        }
        dependency_distances = []

        for sent in doc.sents:
            for token in sent:
                num_dependents = len(list(token.children))
                dependency_distances.extend([abs(token.i - child.i) for child in token.children])

                if token.dep_ == 'ROOT':
                    dependents_per_head['main_clause'].append(num_dependents)
                elif 'advcl' in token.dep_ or 'ccomp' in token.dep_ or 'xcomp' in token.dep_:
                    dependents_per_head['subordinate_clause'].append(num_dependents)
                elif 'conj' in token.dep_:
                    dependents_per_head['coordinate_clause'].append(num_dependents)

        all_dependents = dependents_per_head['main_clause'] + dependents_per_head['subordinate_clause'] + dependents_per_head['coordinate_clause']

        return {
            "avg_dependents_per_head": {
                "main_clause": round(statistics.mean(dependents_per_head['main_clause']), 2) if dependents_per_head['main_clause'] else 0,
                "subordinate_clause": round(statistics.mean(dependents_per_head['subordinate_clause']), 2) if dependents_per_head['subordinate_clause'] else 0,
                "coordinate_clause": round(statistics.mean(dependents_per_head['coordinate_clause']), 2) if dependents_per_head['coordinate_clause'] else 0
            },
            "max_dependents_per_head": max(all_dependents, default=0),
            "mean_dependency_distance": round(statistics.mean(dependency_distances), 2) if dependency_distances else 0
        }
