"""
TO DO:
- add clusters = {k: v for k, v in clusters.items() if k != -1} somewhere??

"""

from typing import List, Dict, Tuple
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class NetworkAnalyzer:
    def build_network(self, 
                     nodes: List[str], 
                     embeddings: np.ndarray,
                     min_similarity: float = 0.3) -> nx.Graph:
        """Build network from nodes and their embeddings."""
        G = nx.Graph()
        
        # Add nodes
        for i, node in enumerate(nodes):
            G.add_node(i, text=node)
        
        # Add edges based on cosine similarity
        similarities = cosine_similarity(embeddings)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if similarities[i, j] >= min_similarity:
                    G.add_edge(i, j, weight=similarities[i, j])
        
        return G
    
    def compute_centrality_metrics(self, G: nx.Graph) -> pd.DataFrame:
        """Compute various centrality metrics for nodes."""
        metrics = {
            'degree': nx.degree_centrality(G),
            'betweenness': nx.betweenness_centrality(G),
            'closeness': nx.closeness_centrality(G),
            'eigenvector': nx.eigenvector_centrality(G, max_iter=1000),
            'pagerank': nx.pagerank(G)
        }
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(metrics)
        # Add node text
        df['text'] = [G.nodes[n]['text'] for n in G.nodes()]
        return df
    
    def detect_communities(self, G: nx.Graph) -> Dict[int, List[str]]:
        """Detect communities using Louvain method."""
        communities = nx.community.louvain_communities(G)
        
        # Organize by community
        result = {}
        for i, community in enumerate(communities):
            result[i] = [G.nodes[n]['text'] for n in community]
            
        return result