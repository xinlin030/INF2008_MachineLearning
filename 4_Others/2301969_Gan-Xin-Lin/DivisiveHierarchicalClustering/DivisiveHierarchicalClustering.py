import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)


def preprocess_data(df):
    """Preprocess the data by cleaning text and extracting features"""
    df['Description'] = df['Description'].str.upper()
    df['procedure_type'] = df['Description'].str.split(',').str[0]
    df['is_bilateral'] = df['Description'].str.contains('BILATERAL').astype(int)
    df['is_unilateral'] = df['Description'].str.contains('UNILATERAL').astype(int)
    df['word_count'] = df['Description'].str.split().str.len()
    df['table_numeric'] = df['Table'].apply(lambda x: float(x[:-1]) if x[0].isdigit() else 0)
    return df


def create_procedure_features(df):
    """Create feature matrix for clustering"""
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(df['Description'].values)
    tfidf_features = pd.DataFrame(tfidf_matrix.toarray(),
                                  columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])

    feature_df = df[['Code', 'is_bilateral', 'is_unilateral', 'table_numeric', 'word_count']].copy()
    return pd.concat([feature_df, tfidf_features], axis=1)


def find_optimal_clusters(X, charts_dir, max_clusters=10):
    """Find optimal number of clusters using silhouette score"""
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
        print(f"Clusters: {k}, Silhouette Score: {score:.4f}")

    optimal_clusters = np.argmax(silhouette_scores) + 2

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score for Clusters')
    plt.xlabel('Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig(os.path.join(charts_dir, 'silhouette_score.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return optimal_clusters


def divisive_clustering(X, min_size=5, depth=0, max_depth=5):
    """Apply divisive clustering recursively"""
    if len(X) <= min_size or depth >= max_depth:
        return np.zeros(len(X), dtype=int)

    kmeans = KMeans(n_clusters=min(2, len(X)), random_state=42).fit(X)
    labels = kmeans.labels_

    if len(set(labels)) == 1:
        return np.zeros(len(X), dtype=int)

    result_labels = np.zeros(len(X), dtype=int)
    next_label = 0

    for i in set(labels):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) > min_size and depth < max_depth - 1:
            sub_labels = divisive_clustering(X[cluster_indices], min_size, depth + 1, max_depth)

            for sub_label in set(sub_labels):
                result_labels[cluster_indices[sub_labels == sub_label]] = next_label
                next_label += 1
        else:
            result_labels[cluster_indices] = next_label
            next_label += 1

    return result_labels


def plot_clusters(clustered_procedures, X_scaled, charts_dir):
    """Visualize clusters using t-SNE"""
    X_tsne = TSNE(n_components=2, random_state=42,
                  perplexity=min(30, len(X_scaled) - 1)).fit_transform(X_scaled)

    plt.figure(figsize=(12, 10))
    for cluster in sorted(set(clustered_procedures['cluster'])):
        cluster_data = clustered_procedures[clustered_procedures['cluster'] == cluster]
        plt.scatter(X_tsne[cluster_data.index, 0],
                    X_tsne[cluster_data.index, 1],
                    label=f'Cluster {cluster}')

    # Mark bilateral and unilateral procedures
    bilateral = clustered_procedures[clustered_procedures['is_bilateral'] == 1]
    unilateral = clustered_procedures[clustered_procedures['is_unilateral'] == 1]

    if not bilateral.empty:
        plt.scatter(X_tsne[bilateral.index, 0], X_tsne[bilateral.index, 1],
                    marker='o', edgecolors='black', s=100, facecolors='none',
                    linewidths=1.5, label='Bilateral')

    if not unilateral.empty:
        plt.scatter(X_tsne[unilateral.index, 0], X_tsne[unilateral.index, 1],
                    marker='s', edgecolors='black', s=100, facecolors='none',
                    linewidths=1.5, label='Unilateral')

    plt.legend()
    plt.title('t-SNE Visualization of Procedure Clusters')
    plt.savefig(os.path.join(charts_dir, 'cluster_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()


def compare_codes(code1, code2, processed_data, feature_df, X_scaled):
    """
    Compare two procedure codes and return their similarity score and descriptions.
    """
    # Check if codes exist in the dataset
    if code1 not in processed_data['Code'].values:
        return {"error": f"Code {code1} not found in dataset"}

    if code2 not in processed_data['Code'].values:
        return {"error": f"Code {code2} not found in dataset"}

    # Get descriptions
    desc1 = processed_data[processed_data['Code'] == code1]['Description'].values[0]
    desc2 = processed_data[processed_data['Code'] == code2]['Description'].values[0]

    # Find indices for the codes
    idx1 = feature_df[feature_df['Code'] == code1].index[0]
    idx2 = feature_df[feature_df['Code'] == code2].index[0]

    # Calculate Euclidean distance
    distance = euclidean(X_scaled[idx1], X_scaled[idx2])

    # Convert distance to similarity score (1 / (1 + distance))
    similarity = 1 / (1 + distance)

    # Get clusters if clustering has been performed
    cluster_info = ""
    if 'cluster' in feature_df.columns:
        cluster1 = feature_df.loc[idx1, 'cluster']
        cluster2 = feature_df.loc[idx2, 'cluster']

        if cluster1 == cluster2:
            cluster_info = f"Both codes are in cluster {cluster1}"
        else:
            cluster_info = f"Code {code1} is in cluster {cluster1}, Code {code2} is in cluster {cluster2}"

    # Check for potential conflicts
    conflict_type = None
    is_bilateral1 = processed_data[processed_data['Code'] == code1]['is_bilateral'].values[0]
    is_unilateral1 = processed_data[processed_data['Code'] == code1]['is_unilateral'].values[0]
    is_bilateral2 = processed_data[processed_data['Code'] == code2]['is_bilateral'].values[0]
    is_unilateral2 = processed_data[processed_data['Code'] == code2]['is_unilateral'].values[0]

    if (is_bilateral1 == 1 and is_unilateral2 == 1) or (is_bilateral2 == 1 and is_unilateral1 == 1):
        procedure_type1 = processed_data[processed_data['Code'] == code1]['procedure_type'].values[0]
        procedure_type2 = processed_data[processed_data['Code'] == code2]['procedure_type'].values[0]

        if procedure_type1 == procedure_type2:
            conflict_type = "Bilateral/Unilateral conflict detected"

    # Prepare result
    result = {
        "code1": code1,
        "code2": code2,
        "description1": desc1,
        "description2": desc2,
        "similarity_score": round(similarity, 4),
        "similarity_percentage": f"{round(similarity * 100, 2)}%"
    }

    if cluster_info:
        result["cluster_info"] = cluster_info

    if conflict_type:
        result["conflict_type"] = conflict_type

    return result


def interactive_code_comparison(processed_data, feature_df, X_scaled):
    print("\n==== Procedure Code Comparison Tool ====")
    print("Enter two procedure codes to compare their similarity.")
    print("Type 'exit' or 'quit' to return to main program.\n")

    # Get list of valid codes for validation
    valid_codes = processed_data['Code'].unique()

    while True:
        # Get first code
        code1 = input("Enter first procedure code (or 'exit' to quit): ").strip()
        if code1.lower() in ['exit', 'quit']:
            break

        # Validate first code
        if code1 not in valid_codes:
            print(f"Error: Code '{code1}' not found in dataset.")
            print(f"Available codes include: {', '.join(valid_codes[:5])}... (and {len(valid_codes) - 5} more)")
            continue

        # Get second code
        code2 = input("Enter second procedure code: ").strip()
        if code2.lower() in ['exit', 'quit']:
            break

        # Validate second code
        if code2 not in valid_codes:
            print(f"Error: Code '{code2}' not found in dataset.")
            print(f"Available codes include: {', '.join(valid_codes[:5])}... (and {len(valid_codes) - 5} more)")
            continue

        # Get comparison result
        result = compare_codes(code1, code2, processed_data, feature_df, X_scaled)

        # Display result in a formatted way
        print("\n==== Comparison Result ====")
        print(f"Code 1: {result['code1']}")
        print(f"Description: {result['description1']}")
        print("\n")
        print(f"Code 2: {result['code2']}")
        print(f"Description: {result['description2']}")
        print("\n")
        print(f"Similarity Score: {result['similarity_score']}")
        print(f"Similarity Percentage: {result['similarity_percentage']}")

        if "cluster_info" in result:
            print(f"\nCluster Information: {result['cluster_info']}")

        if "conflict_type" in result:
            print(f"\nWarning: {result['conflict_type']}")

        print("\n" + "=" * 30 + "\n")

        # Ask if user wants to continue
        cont = input("Compare another pair? (y/n): ").strip().lower()
        if cont != 'y':
            break

    print("Exiting code comparison tool.")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
    charts_dir = script_dir  # Save outputs in the same directory

    ensure_dir(charts_dir)

    data_path = '../../../1_DataPreprocessing/DataSets/CleanedDataset/SL_Eye.csv'
    print(f"Loading data from: {data_path}")
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        print("Please check the path and try again.")
        data_path = input("Enter the correct path to your data file: ")
        data = pd.read_csv(data_path)

    # Process data
    processed_data = preprocess_data(data)
    print(f"Processed {len(processed_data)} records")

    # Create features
    feature_df = create_procedure_features(processed_data)
    features = feature_df.drop(columns=['Code']).values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Find optimal clusters and apply clustering
    print("\nFinding optimal number of clusters...")
    optimal_clusters = find_optimal_clusters(X_scaled, charts_dir)
    print(f"Optimal number of clusters: {optimal_clusters}")

    print("Applying divisive clustering...")
    labels = divisive_clustering(X_scaled, min_size=max(5, len(feature_df) // (optimal_clusters * 2)), max_depth=int(np.log2(optimal_clusters)) + 2)

    # Add cluster labels to dataframe
    feature_df['cluster'] = labels

    # Visualize clusters
    print("Generating cluster visualization...")
    plot_clusters(feature_df, X_scaled, charts_dir)

    # Print cluster statistics
    print("\nCluster Statistics:")
    cluster_stats = feature_df.groupby('cluster').agg({
        'Code': 'count',
        'is_bilateral': 'sum',
        'is_unilateral': 'sum'
    }).rename(columns={'Code': 'count'})
    print(cluster_stats)

    # Start interactive code comparison
    while True:
        print("\n=== Options ===")
        print("1. Compare specific procedure codes")
        print("2. Exit")

        choice = input("Enter your choice (1-2): ").strip()

        if choice == '1':
            interactive_code_comparison(processed_data, feature_df, X_scaled)
        elif choice == '2':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")


if __name__ == '__main__':
    main()