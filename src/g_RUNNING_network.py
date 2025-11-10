from pathlib import Path
from .z_utils import processed_text_path
from .f1_concept_embeddings import ConceptExtractor
from .f2_network import NetworkAnalyzer
import pickle
import matplotlib.pyplot as plt
import networkx as nx

def run_text_pipeline(cleaned_text_filename: str):
    """
    Run concept extraction and network analysis on a pre-cleaned text file.

    cleaned_text_filename: str, e.g. "the_black_cat_cleaned.txt"
    """
    # Initialize modules
    extractor = ConceptExtractor()
    net_analyzer = NetworkAnalyzer()

    # Base name without extension
    base_name = Path(cleaned_text_filename).stem.replace("_cleaned", "")

    # 1️⃣ Load cleaned text
    cleaned_text_path = processed_text_path("cleaned", base_name, cleaned_text_filename)
    if not cleaned_text_path.exists():
        raise FileNotFoundError(f"Cleaned text not found at {cleaned_text_path}")
    
    with open(cleaned_text_path, "r", encoding="utf-8") as f:
        cleaned_text = f.read()

    # 2️⃣ Extract noun phrases (lemmatized)
    phrases = extractor.extract_noun_phrases(cleaned_text, lemmatize=True)

    # ---- Save phrases ----
    concept_dir = processed_text_path("concept", base_name)
    concept_dir.mkdir(parents=True, exist_ok=True)
    phrases_path = concept_dir / f"{base_name}_phrases.pkl"
    with open(phrases_path, "wb") as f:
        pickle.dump(phrases, f)

    # 3️⃣ Embed phrases
    embeddings = extractor.embed_phrases(phrases)

    # 4️⃣ Cluster embeddings
    clusters = extractor.cluster_embeddings(embeddings)

    # ---- Save concept-related outputs ----
    concept_path_clusters = concept_dir / f"{base_name}_clusters.pkl"
    concept_path_embeddings = concept_dir / f"{base_name}_embeddings.pkl"
    with open(concept_path_clusters, "wb") as f:
        pickle.dump(clusters, f)
    with open(concept_path_embeddings, "wb") as f:
        pickle.dump(embeddings, f)

    # ---- Build network using phrases as node labels ----
    # Flatten clusters to get all node indices
    node_indices = [idx for cluster in clusters.values() for idx in cluster]
    node_labels = [phrases[idx] for idx in node_indices]
    node_embeddings = embeddings[node_indices]

    G = net_analyzer.build_network(node_labels, node_embeddings)

    # ---- Save network-related outputs ----
    network_dir = processed_text_path("network", base_name)
    network_dir.mkdir(parents=True, exist_ok=True)

    network_path_graph = network_dir / f"{base_name}_network.pkl"
    with open(network_path_graph, "wb") as f:
        pickle.dump(G, f)

    centrality_df = net_analyzer.compute_centrality_metrics(G)
    centrality_path = network_dir / f"{base_name}_centrality.pkl"
    centrality_df.to_pickle(centrality_path)

    communities = net_analyzer.detect_communities(G)
    communities_path = network_dir / f"{base_name}_communities.pkl"
    with open(communities_path, "wb") as f:
        pickle.dump(communities, f)

    # ---- Save network visualization as PNG ----
    network_png_path = network_dir / f"{base_name}_network.png"
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos,
        labels=nx.get_node_attributes(G, "text"),  # ensure node labels are words
        with_labels=True,
        node_size=500,
        node_color="skyblue",
        edge_color="gray",
        font_size=8
    )
    plt.title(f"Network graph for {base_name}")
    plt.tight_layout()
    plt.savefig(network_png_path, dpi=300)
    plt.close()

    return {
        "cleaned_text": cleaned_text,
        "phrases": phrases,
        "clusters": clusters,
        "network": G,
        "centrality": centrality_df,
        "communities": communities
    }


if __name__ == "__main__":
    results = run_text_pipeline("the_black_cat_cleaned.txt")
    print("[CENTRALITY]")
    print(results["centrality"].head())
    print("[CLUSTERS]")
    print(results["clusters"])
