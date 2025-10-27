"""Module for integrating metrics and analysis."""

from typing import List, Dict
import pandas as pd
import numpy as np
from scipy import stats

class IntegrationAnalyzer:
    def merge_metrics(self,
                     node_metrics: pd.DataFrame,
                     network_metrics: pd.DataFrame) -> pd.DataFrame:
        """Merge node-level linguistic and network metrics."""
        return pd.merge(node_metrics, network_metrics, on='text', how='inner')
    
    def compute_composite_emphasis(self,
                                 node_metrics: pd.DataFrame,
                                 weights: Dict[str, float]) -> pd.Series:
        """Compute weighted emphasis score."""
        # Normalize each metric column
        normalized = node_metrics.copy()
        for col in weights.keys():
            if col in normalized.columns:
                normalized[col] = stats.zscore(normalized[col])
        
        # Compute weighted sum
        emphasis = pd.Series(0.0, index=normalized.index)
        for col, weight in weights.items():
            if col in normalized.columns:
                emphasis += normalized[col] * weight
        
        return emphasis
    
    def community_level_analysis(self,
                               node_metrics: pd.DataFrame,
                               communities: Dict[int, List[str]]) -> pd.DataFrame:
        """Aggregate metrics at community level."""
        community_metrics = []
        
        for comm_id, nodes in communities.items():
            # Get metrics for nodes in this community
            comm_data = node_metrics[node_metrics['text'].isin(nodes)]
            
            # Compute summary statistics
            summary = comm_data.describe()
            metrics = {
                'community_id': comm_id,
                'size': len(nodes),
                'mean_emphasis': comm_data['emphasis'].mean(),
                'std_emphasis': comm_data['emphasis'].std(),
                'max_emphasis': comm_data['emphasis'].max(),
                'mean_centrality': comm_data['eigenvector'].mean()
            }
            community_metrics.append(metrics)
            
        return pd.DataFrame(community_metrics)