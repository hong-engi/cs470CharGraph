import os
import pickle
import networkx as nx
import numpy as np
from scipy.linalg import eigh

def load_graph_from_gpickle(file_path: str) -> nx.Graph:
    # Load a NetworkX graph(.gpickle)
    with open(file_path, "rb") as f:
        G = pickle.load(f)
    return G

def get_laplacian_eigen_feature_vector(G: nx.Graph) -> np.ndarray:
    # Compute the Laplacian eigenvalues and return a vector sorted in ascending order.
    L = nx.laplacian_matrix(G).todense()
    eigvals = np.sort(eigh(L, eigvals_only=True))
    return eigvals

def save_eigen_feature_vector_as_npy(feature_vector: np.ndarray, filename: str):
    # Save an eigen feature vector as npy(.npy)
    np.save(filename, feature_vector)

if __name__ == "__main__":
    input_dir = "graphs_gpickle/" # Input directory
    output_dir = "eigen_feature_vectors/" # Output directory
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".gpickle"):
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, base_name + ".npy")
            
            print(f"Processing {filename} → {base_name}.npy")

            G = load_graph_from_gpickle(input_path)

            feature_vector = get_laplacian_eigen_feature_vector(G)

            save_eigen_feature_vector_as_npy(feature_vector, output_path)

            print(f"Saved eigenvector to {output_path}")