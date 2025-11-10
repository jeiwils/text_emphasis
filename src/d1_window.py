"""


THIS IS TOKEN LEVEL - MAYBE I MAKE ANOTHER MODULE FOR THIS EVENTUALLY!!




 "mattr_score": "<moving_average_type_token_ratio>", # proxy for lexical diversity 



        # Moving Average Type-Token Ratio (mattr) # THIS IS WINDOW, NOT SENTENCE-BASED
        window_size = 50
        if total_tokens < window_size:
            mattr = len(types) / total_tokens if total_tokens else 0
        else:
            ttr_values = []
            for i in range(total_tokens - window_size + 1):
                window = words[i:i+window_size]
                ttr_values.append(len(set(window)) / window_size)
            mattr = round(statistics.mean(ttr_values), 3)

        return {
            "mattr_score": mattr,

"""

