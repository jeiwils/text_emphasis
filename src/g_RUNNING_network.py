from pathlib import Path
from .z_utils import processed_text_path, graph_path, embeddings_path
from .f1_concept_embeddings import ConceptExtractor
from .f2_network import NetworkAnalyzer
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
from collections import Counter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np




"""





        Community → Node color
            - Nodes belonging to the same community have the same hue.
            - Highlights clusters of related concepts.

        Degree → Node size
            - Node size is proportional to its degree centrality.
            - Shows which nodes are more connected (hubs) in the network.

        Betweenness → Edge width & opacity
            - Edges connecting nodes with high betweenness are thicker and more opaque.
            - Highlights connections that act as bridges between different parts of the network.
            Higher betweenness → thicker & more opaque edges

        Eigenvector → Node border color
            - Node border color intensity represents eigenvector centrality.
            - Highlights influential nodes that are connected to other highly connected nodes.
        Higher eigenvector → more intense (darker/brighter) border color
        → highlights nodes that are “influential” because they connect to other well-connected nodes.



        Fill color (community): tells you which cluster a node belongs to.

Border color (eigenvector centrality): shows how central/influential the node is in the whole network.
purple → blue → green → yellow. - purple is least influential, yellow is most 


"""



def filter_top_n_phrases(phrases, n=100):
    """
    Keep only the top-n most frequent phrases and return their indices.
    """
    counts = Counter(phrases)
    top_phrases = [phrase for phrase, _ in counts.most_common(n)]
    filtered_indices = [i for i, p in enumerate(phrases) if p in top_phrases]
    filtered_phrases = [phrases[i] for i in filtered_indices]
    return filtered_phrases, filtered_indices


# -------------------- Function 1: Embeddings --------------------
def generate_embeddings(cleaned_text_path: Path, top_n: int = 100, use_existing: bool = True):
    """
    Extract top-N noun phrases and generate/load embeddings.
    Returns phrases and embeddings.
    """
    extractor = ConceptExtractor()
    base_name = cleaned_text_path.stem.replace("_cleaned", "")
    
    # Load cleaned text
    with open(cleaned_text_path, "r", encoding="utf-8") as f:
        cleaned_text = f.read()
    
    # Extract and filter phrases
    all_phrases = extractor.extract_noun_phrases(cleaned_text, lemmatize=True)
    phrases, _ = filter_top_n_phrases(all_phrases, n=top_n)
    
    # Save phrases
    concept_dir = embeddings_path("concept") / base_name
    concept_dir.mkdir(parents=True, exist_ok=True)
    phrases_path = concept_dir / f"{base_name}_phrases.pkl"
    with open(phrases_path, "wb") as f:
        pickle.dump(phrases, f)
    
    # Load or generate embeddings
    embeddings_file = concept_dir / f"{base_name}_embeddings.pkl"
    if use_existing and embeddings_file.exists():
        with open(embeddings_file, "rb") as f:
            embeddings = pickle.load(f)
    else:
        embeddings = extractor.embed_phrases(phrases)
        with open(embeddings_file, "wb") as f:
            pickle.dump(embeddings, f)
    
    return cleaned_text, phrases, embeddings


# -------------------- Function 2: Network Building --------------------
def build_network_graph(phrases, embeddings):
    """
    Build a NetworkX graph from phrases and embeddings.
    Returns the graph and centrality/community info.
    """
    net_analyzer = NetworkAnalyzer()
    G = net_analyzer.build_network(phrases, embeddings)
    
    # Centrality metrics
    centrality_df = net_analyzer.compute_centrality_metrics(G)
    
    # Community detection
    communities = net_analyzer.detect_communities(G)
    node_to_community = {}
    for comm_id, nodes in communities.items():
        for node_text in nodes:
            node_idx = next(n for n, attr in G.nodes(data=True) if attr['text'] == node_text)
            node_to_community[node_idx] = comm_id
    
    return G, centrality_df, communities, node_to_community







def plot_network(G, centrality_df, node_to_community, communities, base_name: str):
    """
    Plot and save network with styling:
      - Node size = degree
      - Node color = community
      - Node border thickness = eigenvector centrality
      - Edge width/opacity = betweenness
      - Legend included
    """
    graph_dir = graph_path("network") / base_name
    graph_dir.mkdir(parents=True, exist_ok=True)
    
    # Node color = community
    num_comms = len(set(node_to_community.values()))
    cmap = plt.get_cmap('tab20', num_comms)
    node_colors = [cmap(node_to_community.get(n, -1)) for n in G.nodes()]
    
    # Node size = degree
    node_sizes = [500 + 2000 * centrality_df.loc[i, 'degree'] for i in G.nodes()]
    
    # Node border thickness = eigenvector centrality
    eigenvector_norm = (centrality_df['eigenvector'] - centrality_df['eigenvector'].min()) / \
                       (centrality_df['eigenvector'].max() - centrality_df['eigenvector'].min())
    node_border_widths = [1 + 4 * eigenvector_norm[i] for i in G.nodes()]  # min width 1, max 5
    
    # Edge width & opacity = betweenness centrality
    betweenness_norm = (centrality_df['betweenness'] - centrality_df['betweenness'].min()) / \
                       (centrality_df['betweenness'].max() - centrality_df['betweenness'].min())
    edge_widths = []
    edge_alphas = []
    for u, v in G.edges():
        bw = (betweenness_norm[u] + betweenness_norm[v]) / 2
        edge_widths.append(0.5 + 4 * bw)
        edge_alphas.append(0.3 + 0.7 * bw)
    
    # Plot
    plt.figure(figsize=(18, 18))
    pos = nx.fruchterman_reingold_layout(G, seed=42, k=0.5)
    
    # Draw edges
    for i, (u, v) in enumerate(G.edges()):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=edge_widths[i],
            alpha=edge_alphas[i],
            edge_color='gray'
        )
    
    # Draw nodes with border thickness representing eigenvector centrality
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors='black',          # fixed border color
        linewidths=node_border_widths
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        labels=nx.get_node_attributes(G, "text"),
        font_size=8
    )
    
    # ---------------- Legend ----------------
    # Community color
    comm_handles = [
        Patch(facecolor=cmap(i), label=f"Community {i}")
        for i in range(num_comms)
    ]
    
    # Node size (degree) example handles
    size_handles = [
        Line2D([0], [0], marker='o', color='w', label=f"Degree {deg:.2f}",
               markerfacecolor='gray', markersize=np.sqrt(500 + 2000*deg))
        for deg in [0.1, 0.5, 1.0]  # example degrees
    ]
    
    # Node border thickness (eigenvector) example handles
    border_handles = [
        Line2D([0], [0], marker='o', color='black', label=f"Eigenvector {val:.2f}",
               markerfacecolor='white', markersize=10, markeredgewidth=1 + 4*val)
        for val in [0.0, 0.5, 1.0]
    ]
    
    # Edge width/opacity handles
    edge_handles = [
        Line2D([0], [0], color='gray', lw=0.5, alpha=0.3, label='Low betweenness'),
        Line2D([0], [0], color='gray', lw=2.5, alpha=1.0, label='High betweenness')
    ]
    
    plt.legend(handles=comm_handles + size_handles + border_handles + edge_handles,
               loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10, frameon=True)
    
    plt.title(f"Network graph for {base_name}")
    plt.tight_layout()
    plt.savefig(graph_dir / f"{base_name}_network.png", dpi=300)
    plt.close()
    
    # Save graph and community info
    with open(graph_dir / f"{base_name}_network.pkl", "wb") as f:
        pickle.dump(G, f)
    with open(graph_dir / f"{base_name}_communities.pkl", "wb") as f:
        pickle.dump(communities, f)




# -------------------- Full Pipeline --------------------
def run_text_pipeline(cleaned_text_path: Path, use_existing_embeddings: bool = True):
    """
    Run the full pipeline on a single cleaned text file.
    Returns all outputs.
    """
    base_name = cleaned_text_path.stem.replace("_cleaned", "")
    print(f"[INFO] Processing {base_name}")
    
    cleaned_text, phrases, embeddings = generate_embeddings(cleaned_text_path, use_existing=use_existing_embeddings)
    G, centrality_df, communities, node_to_community = build_network_graph(phrases, embeddings)
    plot_network(G, centrality_df, node_to_community, communities, base_name)

    return {
        "cleaned_text": cleaned_text,
        "phrases": phrases,
        "embeddings": embeddings,
        "network": G,
        "centrality": centrality_df,
        "communities": communities
    }


def run_pipeline_all_texts(use_existing_embeddings=True):
    """
    Iterate over all *_cleaned.txt files in cleaned_texts.
    """
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