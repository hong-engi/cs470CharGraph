import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_eigen_feature_vectors(directory: str):
    vectors = []
    names = []
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            path = os.path.join(directory, filename)
            vec = np.load(path)
            vectors.append(vec)
            names.append(filename)
    return vectors, names

def compute_closeness(vec1, vec2):
    min_len = min(len(vec1), len(vec2))
    return np.sum((vec1[:min_len] - vec2[:min_len]) ** 2)

def build_closeness_feature_matrix(vectors, k=3):
    n = len(vectors)
    feature_matrix = []

    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                dist = compute_closeness(vectors[i], vectors[j])
                distances.append((dist, j))
        distances.sort()  # Sort graphs in ascending order of proximity.
        feature_vector = [dist for dist, _ in distances[:k]]
        feature_matrix.append(feature_vector)

    return np.array(feature_matrix)

def attach_labels_to_feature_matrix(feature_matrix: np.ndarray, label_csv_path: str, encode_labels: bool = True):
    """
    Attach labels from a CSV file to the feature matrix.

    Parameters:
    - feature_matrix: (n, k) numpy array of closeness features
    - label_csv_path: path to CSV file containing labels (must align with feature rows)
    - encode_labels: whether to convert labels to integers using LabelEncoder

    Returns:
    - full_matrix: (n, k+1) numpy array with labels appended
    - labels: original or encoded label array
    - encoder: fitted LabelEncoder (or None if not used)
    """
    df = pd.read_csv(label_csv_path)
    labels = df.iloc[:, 1].values

    encoder = None
    if encode_labels:
        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels)

    labels = labels.reshape(-1, 1)
    full_matrix = np.hstack([feature_matrix, labels])

    return full_matrix, labels, encoder

if __name__ == "__main__":
    directory = "eigen_feature_vectors/"      # Eigen feature vectors path
    label_csv_path = "label/labels.csv"
    k = 3                              # top-k
    output_file = "feature_matrix_labeled.npy"  # Output file name
    output_file_csv = "feature_matrix_labeled.csv" # Output csv file name

    vectors, names = load_eigen_feature_vectors(directory)

    feature_matrix = build_closeness_feature_matrix(vectors, k)
    
    full_matrix, labels, encoder = attach_labels_to_feature_matrix(feature_matrix, label_csv_path, encode_labels=True)

    np.save(output_file, full_matrix)
    
    col_names = [f"closeness_{i+1}" for i in range(k)] + ["label"]
    df_out = pd.DataFrame(full_matrix, columns=col_names)
    df_out.to_csv(output_file_csv, index=False)

    print(f"Feature matrix saved to {output_file}")
    print("Shape:",full_matrix.shape)
    print("Sample row:", full_matrix)