from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity

from b_concept_embeddings import generate_embeddings
from x_configs import DEFAULT_WINDOW_SIZE, GENRES
from z_utils import (
    analytics_path,
    graph_path,
    iter_genre_author_dirs,
    load_json,
    text_path,
)

"""

TO DO:
- make sure topic modelling networks are calculated across the same windows that the topic modeller produces them in... aggregated



"""




class NetworkAnalyzer:
    def build_network(
        self,
        nodes: List[str],
        embeddings: np.ndarray,
        min_similarity: float = 0.3,
    ) -> nx.Graph:
        """Build an undirected graph from nodes and their embeddings."""
        if len(nodes) != embeddings.shape[0]:
            raise ValueError(
                f"nodes length ({len(nodes)}) does not match embeddings rows ({embeddings.shape[0]})."
            )

        G = nx.Graph()

        for i, node in enumerate(nodes):
            G.add_node(i, text=node)

        similarities = cosine_similarity(embeddings)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if similarities[i, j] >= min_similarity:
                    G.add_edge(i, j, weight=similarities[i, j])

        return G

    def compute_centrality_metrics(self, G: nx.Graph, weight_attr: str = None) -> pd.DataFrame:
        """Compute various centrality metrics for nodes."""
        betweenness_kwargs = {"weight": weight_attr} if weight_attr else {}
        closeness_kwargs = {"distance": weight_attr} if weight_attr else {}
        metrics = {
            "degree": nx.degree_centrality(G),
            "betweenness": nx.betweenness_centrality(G, **betweenness_kwargs),
            "closeness": nx.closeness_centrality(G, **closeness_kwargs),
            "eigenvector": nx.eigenvector_centrality(G, max_iter=1000, weight=weight_attr),
            "pagerank": nx.pagerank(G, weight=weight_attr),
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


def build_network_graph(
    phrases: List[str],
    embeddings: np.ndarray,
    min_similarity: float = 0.3,
):
    """Build a NetworkX graph and compute centrality + communities."""
    net_analyzer = NetworkAnalyzer()
    G = net_analyzer.build_network(
        phrases,
        embeddings,
        min_similarity=min_similarity,
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


def plot_network(G, centrality_df, node_to_community, communities, base_name: str, node_size_attr: str = None):
    """
    Plot and save network with styling:
      - Node size = degree (or specified node attribute)
      - Node color = community
      - Node border thickness = eigenvector centrality
      - Edge width/opacity = betweenness
    """
    graph_dir = graph_path("network") / base_name
    graph_dir.mkdir(parents=True, exist_ok=True)

    num_comms = len(set(node_to_community.values()))
    cmap = plt.get_cmap("tab20", num_comms if num_comms > 0 else 1)
    node_colors = [cmap(node_to_community.get(n, 0)) for n in G.nodes()]

    if node_size_attr:
        raw_sizes = np.array([G.nodes[n].get(node_size_attr, 0.0) for n in G.nodes()], dtype=float)
        size_norm = _normalize_array(raw_sizes)
        node_sizes = [500 + 2000 * size_norm[idx] for idx, _ in enumerate(G.nodes())]
    else:
        node_sizes = [500 + 2000 * centrality_df.loc[i, "degree"] for i in G.nodes()]

    eigenvector_series = centrality_df["eigenvector"]
    eigenvector_norm = _normalize_array(eigenvector_series.to_numpy())
    eigen_lookup = dict(zip(eigenvector_series.index, eigenvector_norm))
    node_border_widths = [1 + 4 * eigen_lookup.get(i, 0) for i in G.nodes()]

    betweenness_series = centrality_df["betweenness"]
    betweenness_norm = _normalize_array(betweenness_series.to_numpy())
    betw_lookup = dict(zip(betweenness_series.index, betweenness_norm))
    edge_widths = []
    edge_alphas = []
    for u, v in G.edges():
        bw = (betw_lookup.get(u, 0) + betw_lookup.get(v, 0)) / 2
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


def run_text_pipeline(normalised_text_path: Path, use_existing_embeddings: bool = True):
    """End-to-end pipeline for one normalised text file."""
    base_name = normalised_text_path.stem.replace("_normalised", "")
    print(f"[INFO] Processing {base_name}")

    normalised_text, phrases, embeddings = generate_embeddings(
        normalised_text_path,
        use_existing=use_existing_embeddings,
    )
    G, centrality_df, communities, node_to_community = build_network_graph(phrases, embeddings)
    plot_network(G, centrality_df, node_to_community, communities, base_name)

    return {
        "normalised_text": normalised_text,
        "phrases": phrases,
        "embeddings": embeddings,
        "network": G,
        "centrality": centrality_df,
        "communities": communities,
    }


def run_pipeline_all_texts(use_existing_embeddings: bool = True):
    """Iterate over all *_normalised.json files and build their concept networks."""
    base_normalised_dir = text_path("processed", "normalised_texts")
    if not base_normalised_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_normalised_dir}")

    for genre, author, subdir in iter_genre_author_dirs(base_normalised_dir, GENRES):
        for txt_file in subdir.glob("*_normalised.json"):
            try:
                run_text_pipeline(txt_file, use_existing_embeddings=use_existing_embeddings)
            except Exception as e:
                print(f"[ERROR] Failed on {txt_file}: {e}")


def _topic_label(topic: Dict, max_keywords: int = 3) -> str:
    """Format a short, human-readable label for a topic."""
    keywords = topic.get("keywords") or []
    topic_id = topic.get("topic_id", "?")
    if keywords:
        return f"Topic {topic_id}: {', '.join(keywords[:max_keywords])}"
    return f"Topic {topic_id}"


def build_topic_cooccurrence_graph(
    topic_files: List[Path],
    min_edge_weight: int = 1,
    min_node_mentions: int = 1,
):
    """
    Build a co-occurrence graph from topic modelling outputs.

    Nodes = topics (sized by mention count).
    Edges = how often two topics appear in the same sentence/window.
    """
    node_counts: Counter = Counter()
    edge_counts: Counter = Counter()

    for topic_file in topic_files:
        data = load_json(topic_file)
        topics = data.get("topics", [])
        if not topics:
            continue

        topic_labels = {t["topic_id"]: _topic_label(t) for t in topics if "topic_id" in t}

        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        base_window_size = meta.get("base_window_size", DEFAULT_WINDOW_SIZE)
        num_sentences = meta.get("num_sentences")

        for topic in topics:
            label = topic_labels.get(topic.get("topic_id"))
            if not label:
                continue
            mentions = topic.get("mentions", [])
            node_counts[label] += len(mentions)

        # Map topic mentions to the base sliding windows used across metrics.
        window_topics: Dict[int, set] = {}
        for topic in topics:
            label = topic_labels.get(topic.get("topic_id"))
            if not label:
                continue
            for mention in topic.get("mentions", []):
                start_sentence = mention.get("sentence_index")
                end_sentence = mention.get("end_sentence", start_sentence)
                if start_sentence is None or end_sentence is None:
                    continue

                max_start = (
                    (num_sentences - base_window_size)
                    if num_sentences is not None
                    else end_sentence
                )
                first_base_window = max(0, start_sentence - base_window_size + 1)
                last_base_window = max_start if max_start is not None else end_sentence

                for window_start in range(first_base_window, last_base_window + 1):
                    window_end = window_start + base_window_size - 1
                    if window_end < start_sentence or window_start > end_sentence:
                        continue
                    window_topics.setdefault(window_start, set()).add(label)

        for topics_in_window in window_topics.values():
            if len(topics_in_window) < 2:
                continue
            for a, b in combinations(sorted(topics_in_window), 2):
                edge_counts[(a, b)] += 1

    G = nx.Graph()
    for label, count in node_counts.items():
        if count < min_node_mentions:
            continue
        G.add_node(label, text=label, size=count)

    for (a, b), weight in edge_counts.items():
        if weight < min_edge_weight:
            continue
        if a in G.nodes and b in G.nodes:
            G.add_edge(a, b, weight=weight)

    return G


def run_topic_cooccurrence_networks(
    min_edge_weight: int = 1,
    min_node_mentions: int = 1,
):
    """
    Build topic co-occurrence graphs from *_topics.json files under data/analytics/topic_modelling/<category>/.

    Node size = mention count; edge weight = co-occurrence count.
    """
    topic_root = analytics_path("topic")
    if not topic_root.exists():
        raise FileNotFoundError(f"Directory not found: {topic_root}")

    net_analyzer = NetworkAnalyzer()

    for genre, author, author_dir in iter_genre_author_dirs(topic_root, GENRES):
        topic_files = list(author_dir.rglob("*_clustered_topics.json"))
        if not topic_files:
            continue

        print(f"[INFO] Building topic co-occurrence network for {genre}/{author}")
        G = build_topic_cooccurrence_graph(
            topic_files,
            min_edge_weight=min_edge_weight,
            min_node_mentions=min_node_mentions,
        )

        if G.number_of_nodes() == 0:
            print(f"[WARN] No topic nodes kept after filtering for {genre}/{author}")
            continue

        centrality_df = net_analyzer.compute_centrality_metrics(G, weight_attr="weight")
        communities = net_analyzer.detect_communities(G)
        node_to_community = {
            node: comm_id for comm_id, comm_nodes in communities.items() for node in comm_nodes
        }

        base_name = f"{genre}_{author}_topics"
        plot_network(
            G,
            centrality_df,
            node_to_community,
            communities,
            base_name,
            node_size_attr="size",
        )

        graph_dir = graph_path("network") / base_name
        graph_dir.mkdir(parents=True, exist_ok=True)
        with open(graph_dir / f"{base_name}_node_stats.json", "w", encoding="utf-8") as f:
            json.dump(
                {node: {"mentions": G.nodes[node].get("size", 0)} for node in G.nodes()},
                f,
                indent=2,
            )
        with open(graph_dir / f"{base_name}_edge_stats.json", "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"source": u, "target": v, "weight": attrs.get("weight", 0)}
                    for u, v, attrs in G.edges(data=True)
                ],
                f,
                indent=2,
            )


if __name__ == "__main__":
    run_pipeline_all_texts(False)
