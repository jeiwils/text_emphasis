from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity

from .f1_concept_embeddings import ConceptExtractor
from .z_utils import processed_text_path, graph_path, embeddings_path


class NetworkAnalyzer:
    def build_network(
        self,
        nodes: List[str],
        embeddings: np.ndarray,
        min_similarity: float = 0.3,
        normalize_embeddings: bool = False,
    ) -> nx.Graph:
        """Build an undirected graph from nodes and their embeddings."""
        if len(nodes) != embeddings.shape[0]:
            raise ValueError(
                f"nodes length ({len(nodes)}) does not match embeddings rows ({embeddings.shape[0]})."
            )

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("Embeddings contain zero vectors; cannot L2-normalize.")
            embeddings = embeddings / norms

        G = nx.Graph()

        for i, node in enumerate(nodes):
            G.add_node(i, text=node)

        similarities = cosine_similarity(embeddings)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if similarities[i, j] >= min_similarity:
                    G.add_edge(i, j, weight=similarities[i, j])

        return G

    def compute_centrality_metrics(self, G: nx.Graph) -> pd.DataFrame:
        """Compute various centrality metrics for nodes."""
        metrics = {
            "degree": nx.degree_centrality(G),
            "betweenness": nx.betweenness_centrality(G),
            "closeness": nx.closeness_centrality(G),
            "eigenvector": nx.eigenvector_centrality(G, max_iter=1000),
            "pagerank": nx.pagerank(G),
        }

        df = pd.DataFrame.from_dict(metrics)
        df["text"] = [G.nodes[n]["text"] for n in G.nodes()]
        return df

    def detect_communities(self, G: nx.Graph) -> Dict[int, List[str]]:
        """Detect communities using Louvain method."""
        communities = nx.community.louvain_communities(G)

        result = {}
        for i, community in enumerate(communities):
            result[i] = [G.nodes[n]["text"] for n in community]

        return result


def filter_top_n_phrases(phrases: List[str], n: int = 100) -> Tuple[List[str], List[int]]:
    """Keep only the top-n most frequent phrases and return them with their indices."""
    counts = Counter(phrases)
    top_phrases = [phrase for phrase, _ in counts.most_common(n)]
    filtered_indices = [i for i, p in enumerate(phrases) if p in top_phrases]
    filtered_phrases = [phrases[i] for i in filtered_indices]
    return filtered_phrases, filtered_indices


def generate_embeddings(cleaned_text_path: Path, top_n: int = 100, use_existing: bool = True):
    """Extract top-N noun phrases and generate or load embeddings."""
    extractor = ConceptExtractor()
    base_name = cleaned_text_path.stem.replace("_cleaned", "")

    with open(cleaned_text_path, "r", encoding="utf-8") as f:
        cleaned_text = f.read()

    all_phrases = extractor.extract_noun_phrases(cleaned_text, lemmatize=True)
    phrases, _ = filter_top_n_phrases(all_phrases, n=top_n)

    concept_dir = embeddings_path("concept") / base_name
    concept_dir.mkdir(parents=True, exist_ok=True)
    phrases_path = concept_dir / f"{base_name}_phrases.pkl"
    with open(phrases_path, "wb") as f:
        pickle.dump(phrases, f)

    embeddings_file = concept_dir / f"{base_name}_embeddings.pkl"
    if use_existing and embeddings_file.exists():
        with open(embeddings_file, "rb") as f:
            embeddings = pickle.load(f)
    else:
        embeddings = extractor.embed_phrases(phrases)
        with open(embeddings_file, "wb") as f:
            pickle.dump(embeddings, f)

    return cleaned_text, phrases, embeddings


def build_network_graph(
    phrases: List[str],
    embeddings: np.ndarray,
    min_similarity: float = 0.3,
    normalize_embeddings: bool = False,
):
    """Build a NetworkX graph and compute centrality + communities."""
    net_analyzer = NetworkAnalyzer()
    G = net_analyzer.build_network(
        phrases,
        embeddings,
        min_similarity=min_similarity,
        normalize_embeddings=normalize_embeddings,
    )

    centrality_df = net_analyzer.compute_centrality_metrics(G)

    communities = net_analyzer.detect_communities(G)
    node_to_community = {}
    for comm_id, nodes in communities.items():
        for node_text in nodes:
            node_idx = next(n for n, attr in G.nodes(data=True) if attr["text"] == node_text)
            node_to_community[node_idx] = comm_id

    return G, centrality_df, communities, node_to_community


def _normalize_array(values: np.ndarray) -> np.ndarray:
    """Normalize an array to 0-1, handling constant arrays gracefully."""
    min_val = values.min()
    max_val = values.max()
    if np.isclose(max_val, min_val):
        return np.zeros_like(values)
    return (values - min_val) / (max_val - min_val)


def plot_network(G, centrality_df, node_to_community, communities, base_name: str):
    """
    Plot and save network with styling:
      - Node size = degree
      - Node color = community
      - Node border thickness = eigenvector centrality
      - Edge width/opacity = betweenness
    """
    graph_dir = graph_path("network") / base_name
    graph_dir.mkdir(parents=True, exist_ok=True)

    num_comms = len(set(node_to_community.values()))
    cmap = plt.get_cmap("tab20", num_comms if num_comms > 0 else 1)
    node_colors = [cmap(node_to_community.get(n, 0)) for n in G.nodes()]

    node_sizes = [500 + 2000 * centrality_df.loc[i, "degree"] for i in G.nodes()]

    eigenvector_norm = _normalize_array(centrality_df["eigenvector"].to_numpy())
    node_border_widths = [1 + 4 * eigenvector_norm[i] for i in G.nodes()]

    betweenness_norm = _normalize_array(centrality_df["betweenness"].to_numpy())
    edge_widths = []
    edge_alphas = []
    for u, v in G.edges():
        bw = (betweenness_norm[u] + betweenness_norm[v]) / 2
        edge_widths.append(0.5 + 4 * bw)
        edge_alphas.append(0.3 + 0.7 * bw)

    plt.figure(figsize=(18, 18))
    pos = nx.fruchterman_reingold_layout(G, seed=42, k=0.5)

    for i, (u, v) in enumerate(G.edges()):
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=edge_widths[i],
            alpha=edge_alphas[i],
            edge_color="gray",
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=node_border_widths,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=nx.get_node_attributes(G, "text"),
        font_size=8,
    )

    comm_handles = [
        Line2D([0], [0], marker="o", linestyle="", color="w", label=f"Community {i}", markerfacecolor=cmap(i))
        for i in range(max(1, num_comms))
    ]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Degree {deg:.2f}",
            markerfacecolor="gray",
            markersize=np.sqrt(500 + 2000 * deg),
        )
        for deg in [0.1, 0.5, 1.0]
    ]
    border_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            label=f"Eigenvector {val:.2f}",
            markerfacecolor="white",
            markersize=10,
            markeredgewidth=1 + 4 * val,
        )
        for val in [0.0, 0.5, 1.0]
    ]
    edge_handles = [
        Line2D([0], [0], color="gray", lw=0.5, alpha=0.3, label="Low betweenness"),
        Line2D([0], [0], color="gray", lw=2.5, alpha=1.0, label="High betweenness"),
    ]

    plt.legend(
        handles=comm_handles + size_handles + border_handles + edge_handles,
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
        fontsize=10,
        frameon=True,
    )

    plt.title(f"Network graph for {base_name}")
    plt.tight_layout()
    plt.savefig(graph_dir / f"{base_name}_network.png", dpi=300)
    plt.close()

    with open(graph_dir / f"{base_name}_network.pkl", "wb") as f:
        pickle.dump(G, f)
    with open(graph_dir / f"{base_name}_communities.pkl", "wb") as f:
        pickle.dump(communities, f)


def run_text_pipeline(cleaned_text_path: Path, use_existing_embeddings: bool = True):
    """End-to-end pipeline for one cleaned text file."""
    base_name = cleaned_text_path.stem.replace("_cleaned", "")
    print(f"[INFO] Processing {base_name}")

    cleaned_text, phrases, embeddings = generate_embeddings(
        cleaned_text_path,
        use_existing=use_existing_embeddings,
    )
    G, centrality_df, communities, node_to_community = build_network_graph(phrases, embeddings)
    plot_network(G, centrality_df, node_to_community, communities, base_name)

    return {
        "cleaned_text": cleaned_text,
        "phrases": phrases,
        "embeddings": embeddings,
        "network": G,
        "centrality": centrality_df,
        "communities": communities,
    }


def run_pipeline_all_texts(use_existing_embeddings: bool = True):
    """Iterate over all *_cleaned.txt files and build their concept networks."""
    base_cleaned_dir = processed_text_path("cleaned")
    if not base_cleaned_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_cleaned_dir}")

    for subdir in base_cleaned_dir.iterdir():
        if subdir.is_dir():
            for txt_file in subdir.glob("*_cleaned.txt"):
                try:
                    run_text_pipeline(txt_file, use_existing_embeddings=use_existing_embeddings)
                except Exception as e:
                    print(f"[ERROR] Failed on {txt_file}: {e}")


if __name__ == "__main__":
    run_pipeline_all_texts(False)
