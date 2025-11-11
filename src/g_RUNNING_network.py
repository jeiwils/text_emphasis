from pathlib import Path
from .z_utils import processed_text_path
from .f1_concept_embeddings import ConceptExtractor
from .f2_network import NetworkAnalyzer
import pickle
import matplotlib.pyplot as plt
import networkx as nx



from collections import Counter




def filter_top_n_phrases(phrases, n=100):
    """
    Keep only the top-n most frequent phrases.
    Returns the filtered list and their original indices.
    """
    counts = Counter(phrases)
    top_phrases = [phrase for phrase, _ in counts.most_common(n)]
    
    # Keep only phrases in top_phrases, preserve original order
    filtered_indices = [i for i, p in enumerate(phrases) if p in top_phrases]
    filtered_phrases = [phrases[i] for i in filtered_indices]
    
    return filtered_phrases, filtered_indices



def run_text_pipeline(cleaned_text_path: Path, use_existing_embeddings: bool = True):
    """
    Run concept extraction and network analysis on a single pre-cleaned text file.

    use_existing_embeddings: if True, will load embeddings from disk if they exist
    """
    # Initialize modules
    extractor = ConceptExtractor()
    net_analyzer = NetworkAnalyzer()

    base_name = cleaned_text_path.stem.replace("_cleaned", "")
    print(f"[INFO] Processing {base_name}")

    # 1️⃣ Load cleaned text
    with open(cleaned_text_path, "r", encoding="utf-8") as f:
        cleaned_text = f.read()

    # 2️⃣ Extract noun phrases (lemmatized)
    all_phrases = extractor.extract_noun_phrases(cleaned_text, lemmatize=True)

    # Filter top-N most frequent phrases
    top_n = 100
    phrases, top_indices = filter_top_n_phrases(all_phrases, n=top_n)

    # ---- Save phrases ----
    concept_dir = processed_text_path("concept", base_name)
    concept_dir.mkdir(parents=True, exist_ok=True)
    phrases_path = concept_dir / f"{base_name}_phrases.pkl"
    with open(phrases_path, "wb") as f:
        pickle.dump(phrases, f)

    # ---- Embeddings ----
    embeddings_path = concept_dir / f"{base_name}_embeddings.pkl"
    if use_existing_embeddings and embeddings_path.exists():
        print(f"[INFO] Loading existing embeddings for {base_name}")
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
    else:
        print(f"[INFO] Generating embeddings for {base_name}")
        embeddings = extractor.embed_phrases(phrases)
        with open(embeddings_path, "wb") as f:
            pickle.dump(embeddings, f)

    # 4️⃣ Cluster embeddings
    clusters = extractor.cluster_embeddings(embeddings)

    # ---- Build network ----
    node_labels = phrases
    node_embeddings = embeddings
    G = net_analyzer.build_network(node_labels, node_embeddings)

    # ---- Save network outputs ----
    network_dir = processed_text_path("network", base_name)
    network_dir.mkdir(parents=True, exist_ok=True)

    with open(network_dir / f"{base_name}_network.pkl", "wb") as f:
        pickle.dump(G, f)

    centrality_df = net_analyzer.compute_centrality_metrics(G)
    centrality_df.to_pickle(network_dir / f"{base_name}_centrality.pkl")

    communities = net_analyzer.detect_communities(G)
    with open(network_dir / f"{base_name}_communities.pkl", "wb") as f:
        pickle.dump(communities, f)

    # ---- Save visualization ----
    network_png_path = network_dir / f"{base_name}_network.png"
    plt.figure(figsize=(18, 18))
    pos = nx.spring_layout(G, seed=42, k=0.5)
    nx.draw(
        G, pos,
        labels=nx.get_node_attributes(G, "text"),
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



def run_pipeline_all_texts(use_existing_embeddings = True):
    """
    Iterate over all subdirectories and all *_cleaned.txt files in cleaned_texts.
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