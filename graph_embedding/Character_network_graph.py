import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle

def build_undirected_graph_from_csv(csv_path: str) -> nx.Graph:
    # Build an undirected weighted graph from a CSV file.
    df = pd.read_csv(csv_path)
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
    return G

def save_graph_as_gpickle(G: nx.Graph, file_path: str):
    # Save the built graph as a .gpickle file.
    with open(file_path, "wb") as f:
        pickle.dump(G, f)

def load_graph_from_gpickle(file_path: str) -> nx.Graph:
    # Load the built graph.
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    return G

def visualize_graph(G: nx.Graph, title: str = "Graph Visualization"):
    # Visualize the graph.
    
    # Visualize with larger weights as shorter distances.
    G_dist = G.copy()
    for u, v in G_dist.edges():
        weight = G_dist[u][v]['weight']
        G_dist[u][v]['distance'] = 0.1 / weight

    pos = nx.spring_layout(G_dist, weight='distance', seed=42)
    
    edge_labels = nx.get_edge_attributes(G, 'weight') # Edge = weight
    
    # Draw
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=100)
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=5)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    input_dir = "graphs_csv/" # Input directory
    output_dir = "graphs_gpickle/" # Output directory
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".csv"):
            csv_path = os.path.join(input_dir, filename)          # Input CSV file path
            base_name = os.path.splitext(filename)[0]
            gpickle_path = os.path.join(output_dir, base_name + ".gpickle")    # File storage path

            print(f"Processing {filename} → {base_name}.gpickle")

            G = build_undirected_graph_from_csv(csv_path)   # Graph generation
            save_graph_as_gpickle(G, gpickle_path)          # Graph saving
            visualize_graph(G, title=f"Graph: {base_name}")   # Graph visualization