import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Set the number of CPU cores to use
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


class TOSPDivisiveHierarchicalClustering:
    def __init__(self, n_clusters=5, charts_dir='DataSets/Charts'):
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.scaler = StandardScaler()
        self.n_clusters = n_clusters
        self.charts_dir = charts_dir
        self.cluster_history = []  # To store the cluster splits for dendrogram

        # Create charts directory if it doesn't exist
        os.makedirs(self.charts_dir, exist_ok=True)

    def preprocess_data(self, df):
        """
        Preprocess the TOSP data and create features
        """
        # Clean description text
        df['Description'] = df['Description'].str.upper()

        # Extract features
        df['procedure_type'] = df['Description'].apply(lambda x: x.split(',')[0])
        df['is_bilateral'] = df['Description'].str.contains('BILATERAL').astype(int)
        df['is_unilateral'] = df['Description'].str.contains('UNILATERAL').astype(int)
        df['word_count'] = df['Description'].apply(lambda x: len(x.split()))

        # Convert table to numeric
        df['table_numeric'] = df['Table'].apply(lambda x:
                                                float(x[:-1]) if x[0].isdigit() else 0)

        return df

    def create_procedure_features(self, df):
        """
        Create features for individual procedures
        """
        # Get TF-IDF features for descriptions
        tfidf_matrix = self.vectorizer.fit_transform(df['Description'].values)
        tfidf_features = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )

        # Combine with other features
        feature_df = pd.DataFrame({
            'code': df['Code'],
            'is_bilateral': df['is_bilateral'],
            'is_unilateral': df['is_unilateral'],
            'table_numeric': df['table_numeric'],
            'word_count': df['word_count']
        })

        # Combine all features
        feature_df = pd.concat([feature_df, tfidf_features], axis=1)

        return feature_df

    def find_optimal_clusters(self, X, max_clusters=10):
        """
        Find optimal number of clusters using silhouette score and save plot
        """
        silhouette_scores = []

        for n_clusters in range(2, max_clusters + 1):
            model = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = model.fit_predict(X)

            # Calculate silhouette score
            score = silhouette_score(X, cluster_labels)
            silhouette_scores.append(score)

            print(f"Clusters: {n_clusters}, Silhouette Score: {score:.4f}")

        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
        plt.title('Silhouette Score for Different Numbers of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.grid(True)

        # Save the plot
        plt.savefig(os.path.join(self.charts_dir, 'Divisive_Clustering_similarity.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # Get optimal number of clusters
        optimal_clusters = np.argmax(silhouette_scores) + 2
        return optimal_clusters

    def divisive_clustering(self, X, feature_df, min_cluster_size=5, max_depth=5, current_depth=0):
        """
        Apply divisive clustering recursively

        Args:
            X: Scaled feature matrix
            feature_df: Original feature dataframe
            min_cluster_size: Min items needed to split a cluster further
            max_depth: Maximum recursion depth for splitting
            current_depth: Current recursion depth
        """
        n_samples = X.shape[0]

        # Base case: Stop if cluster is too small or max depth reached
        if n_samples <= min_cluster_size or current_depth >= max_depth:
            return np.zeros(n_samples, dtype=int)

        # If only one sample, assign to cluster 0
        if n_samples == 1:
            return np.array([0])

        # Try to split using K-means (k=2)
        kmeans = KMeans(n_clusters=min(2, n_samples), random_state=42)
        labels = kmeans.fit_predict(X)

        # If split wasn't effective (only one resulting cluster), return
        if len(np.unique(labels)) == 1:
            return np.zeros(n_samples, dtype=int)

        # Record this split for dendrogram construction
        self.cluster_history.append({
            'depth': current_depth,
            'parent_size': n_samples,
            'children': [(labels == i).sum() for i in range(len(np.unique(labels)))]
        })

        # Recursively split each cluster
        result_labels = np.zeros(n_samples, dtype=int)
        next_label = 0

        for i in range(len(np.unique(labels))):
            # Get indices of samples in this cluster
            cluster_indices = np.where(labels == i)[0]

            if len(cluster_indices) > min_cluster_size and current_depth < max_depth - 1:
                # Recursively split this cluster
                sub_labels = self.divisive_clustering(
                    X[cluster_indices],
                    feature_df.iloc[cluster_indices],
                    min_cluster_size,
                    max_depth,
                    current_depth + 1
                )

                # Assign new global labels
                for j, sub_label in enumerate(np.unique(sub_labels)):
                    mask = sub_labels == sub_label
                    result_labels[cluster_indices[mask]] = next_label
                    next_label += 1
            else:
                # Don't split further, assign all to the same cluster
                result_labels[cluster_indices] = next_label
                next_label += 1

        return result_labels

    def cluster_procedures(self, feature_df, auto_tune=True):
        """
        Apply divisive hierarchical clustering to procedures
        """
        # Prepare features for clustering
        features_to_use = [col for col in feature_df.columns if col != 'code']
        X = feature_df[features_to_use].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Find optimal number of clusters if auto_tune is True
        if auto_tune:
            self.n_clusters = self.find_optimal_clusters(X_scaled)

        # Apply divisive clustering
        self.cluster_history = []  # Reset history before clustering
        labels = self.divisive_clustering(
            X_scaled,
            feature_df,
            min_cluster_size=max(5, len(feature_df) // (self.n_clusters * 2)),
            max_depth=int(np.log2(self.n_clusters)) + 2  # Adjust depth based on cluster count
        )

        # Remap labels to ensure they're consecutive
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        remapped_labels = np.array([label_map[l] for l in labels])

        # Add cluster assignments to the feature dataframe
        result_df = feature_df.copy()
        result_df['cluster'] = remapped_labels

        return result_df, X_scaled

    def plot_dendrogram(self, X, max_display=30):
        """
        Plot a simulated dendrogram for divisive clustering and save it
        """
        # Create a custom linkage matrix using our split history
        if not self.cluster_history:
            print("No cluster history available for dendrogram.")
            return

        # This is a simplified representation since scikit-learn doesn't have built-in
        # divisive clustering dendrograms
        plt.figure(figsize=(12, 8))

        # Plot levels of the hierarchy
        max_depth = max(item['depth'] for item in self.cluster_history) + 1

        # Count clusters at each level
        clusters_per_level = {}
        for item in self.cluster_history:
            depth = item['depth']
            if depth not in clusters_per_level:
                clusters_per_level[depth] = 0
            clusters_per_level[depth] += len(item['children'])

        depths = sorted(clusters_per_level.keys())
        counts = [clusters_per_level[d] for d in depths]

        plt.bar(range(len(depths)), counts, tick_label=[f"Level {d}" for d in depths])
        plt.title('Divisive Hierarchical Clustering Structure')
        plt.xlabel('Hierarchy Level')
        plt.ylabel('Number of Clusters')
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(self.charts_dir, 'Divisive_Clustering_Structure.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

    def identify_potential_conflicts(self, df, clustered_procedures):
        """
        Identify potential conflicts by analyzing procedures within the same cluster
        """
        conflicts = []

        # Group procedures by cluster
        for cluster_id in clustered_procedures['cluster'].unique():
            cluster_procs = clustered_procedures[clustered_procedures['cluster'] == cluster_id]

            # Check for bilateral/unilateral conflicts within cluster
            bilateral_procs = cluster_procs[cluster_procs['is_bilateral'] == 1]['code'].values
            unilateral_procs = cluster_procs[cluster_procs['is_unilateral'] == 1]['code'].values

            # Find potential conflicts
            for b_code in bilateral_procs:
                for u_code in unilateral_procs:
                    b_desc = df[df['Code'] == b_code]['Description'].values[0]
                    u_desc = df[df['Code'] == u_code]['Description'].values[0]

                    # Only flag as conflict if they appear to be the same procedure type
                    b_proc_type = df[df['Code'] == b_code]['procedure_type'].values[0]
                    u_proc_type = df[df['Code'] == u_code]['procedure_type'].values[0]

                    if b_proc_type == u_proc_type:
                        conflicts.append({
                            'code1': b_code,
                            'code2': u_code,
                            'desc1': b_desc,
                            'desc2': u_desc,
                            'cluster': cluster_id,
                            'conflict_type': 'Bilateral/Unilateral',
                            'score': 1.0  # Max confidence for this type of conflict
                        })

        return pd.DataFrame(conflicts)

    def calculate_pairwise_similarities(self, clustered_procedures, X_scaled):
        """
        Calculate pairwise similarities for procedures within the same cluster
        """
        # Compute pairwise distances
        dist_matrix = pdist(X_scaled, metric='euclidean')
        dist_matrix = squareform(dist_matrix)

        # Convert distances to similarities (1 / (1 + distance))
        sim_matrix = 1 / (1 + dist_matrix)

        # Create similarity dataframe
        codes = clustered_procedures['code'].values
        similarities = []

        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                cluster_i = clustered_procedures.iloc[i]['cluster']
                cluster_j = clustered_procedures.iloc[j]['cluster']

                # Only consider pairs in the same cluster
                if cluster_i == cluster_j:
                    similarities.append({
                        'code1': codes[i],
                        'code2': codes[j],
                        'similarity': sim_matrix[i, j],
                        'cluster': cluster_i
                    })

        return pd.DataFrame(similarities)

    def visualize_clusters(self, clustered_procedures, X_scaled):
        """
        Visualize clusters using dimensionality reduction and save the plot
        """
        from sklearn.manifold import TSNE

        # Apply t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled) - 1))
        X_tsne = tsne.fit_transform(X_scaled)

        # Create visualization dataframe
        viz_df = pd.DataFrame({
            'x': X_tsne[:, 0],
            'y': X_tsne[:, 1],
            'cluster': clustered_procedures['cluster'],
            'code': clustered_procedures['code'],
            'is_bilateral': clustered_procedures['is_bilateral'],
            'is_unilateral': clustered_procedures['is_unilateral']
        })

        # Plot clusters
        plt.figure(figsize=(12, 10))

        # Plot points colored by cluster
        for cluster_id in viz_df['cluster'].unique():
            cluster_data = viz_df[viz_df['cluster'] == cluster_id]
            plt.scatter(
                cluster_data['x'],
                cluster_data['y'],
                label=f'Cluster {cluster_id}',
                alpha=0.7
            )

        # Mark bilateral and unilateral procedures
        bilateral_points = viz_df[viz_df['is_bilateral'] == 1]
        unilateral_points = viz_df[viz_df['is_unilateral'] == 1]

        # Add a small marker for bilateral procedures
        plt.scatter(
            bilateral_points['x'],
            bilateral_points['y'],
            marker='o',
            edgecolors='black',
            s=100,
            facecolors='none',
            linewidths=1.5,
            label='Bilateral'
        )

        # Add a small marker for unilateral procedures
        plt.scatter(
            unilateral_points['x'],
            unilateral_points['y'],
            marker='s',
            edgecolors='black',
            s=100,
            facecolors='none',
            linewidths=1.5,
            label='Unilateral'
        )

        plt.title('t-SNE Visualization of Procedure Clusters (Divisive Clustering)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(self.charts_dir, 'Divisive_Clustering_Clusters.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_results_as_visualizations(self, conflicts, similarities):
        """
        Save results as visualizations instead of CSV files
        """
        # Visualize conflicts as a table
        if not conflicts.empty:
            plt.figure(figsize=(14, len(conflicts) * 0.5 + 2))
            plt.axis('off')

            # Create table data
            table_data = []
            for _, row in conflicts.iterrows():
                table_data.append([
                    row['code1'],
                    row['code2'],
                    row['conflict_type'],
                    row['cluster'],
                    row['score']
                ])

            # Create table
            table = plt.table(
                cellText=table_data,
                colLabels=['Code 1', 'Code 2', 'Conflict Type', 'Cluster', 'Score'],
                loc='center',
                cellLoc='center'
            )

            # Style table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            plt.title('Potential Conflicts (Divisive Clustering)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, 'Divisive_clustering_conflicts.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        # Visualize similarities as a heatmap
        if not similarities.empty and len(similarities) <= 100:  # Limit to 100 pairs for readability
            # Get top similarities if there are too many
            if len(similarities) > 100:
                similarities = similarities.sort_values('similarity', ascending=False).head(100)

            # Get unique codes
            all_codes = sorted(list(set(similarities['code1'].tolist() + similarities['code2'].tolist())))

            # Create a similarity matrix
            sim_matrix = pd.DataFrame(index=all_codes, columns=all_codes)

            # Fill diagonal with 1s (self-similarity)
            for code in all_codes:
                sim_matrix.loc[code, code] = 1.0

            # Fill in similarities from dataframe
            for _, row in similarities.iterrows():
                sim_matrix.loc[row['code1'], row['code2']] = row['similarity']
                sim_matrix.loc[row['code2'], row['code1']] = row['similarity']

            # Fill NaNs with 0s
            sim_matrix = sim_matrix.fillna(0)

            # Plot heatmap
            plt.figure(figsize=(12, 10))
            plt.imshow(sim_matrix.values, cmap='Blues', interpolation='nearest')

            # Add colorbar
            plt.colorbar(label='Similarity')

            # Label axes
            plt.xticks(range(len(all_codes)), all_codes, rotation=90)
            plt.yticks(range(len(all_codes)), all_codes)

            plt.title('Procedure Similarities (Divisive Clustering)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, 'Divisive_clustering_similarities.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()


# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('DataSets/CleanedDataset/SL_Eye.csv')

    # Initialize model with custom charts directory
    cluster_model = TOSPDivisiveHierarchicalClustering(n_clusters=5, charts_dir='DataSets/Charts')

    # Preprocess data
    processed_data = cluster_model.preprocess_data(data)

    # Create procedure features
    procedure_features = cluster_model.create_procedure_features(processed_data)

    # Apply clustering with auto-tuning
    clustered_procedures, X_scaled = cluster_model.cluster_procedures(procedure_features, auto_tune=True)

    # Plot dendrogram-like structure
    cluster_model.plot_dendrogram(X_scaled, max_display=30)

    # Visualize clusters
    cluster_model.visualize_clusters(clustered_procedures, X_scaled)

    # Identify potential conflicts
    conflicts = cluster_model.identify_potential_conflicts(processed_data, clustered_procedures)

    # Calculate pairwise similarities within clusters
    similarities = cluster_model.calculate_pairwise_similarities(clustered_procedures, X_scaled)

    # Save results as visualizations
    cluster_model.save_results_as_visualizations(conflicts, similarities)

    # Print cluster statistics
    print("\n=== Cluster Statistics ===")
    cluster_stats = clustered_procedures.groupby('cluster').agg({
        'code': 'count',
        'is_bilateral': 'sum',
        'is_unilateral': 'sum'
    }).rename(columns={'code': 'count'})
    print(cluster_stats)

    # Print top conflicts
    print("\n=== Top Potential Conflicts ===")
    if not conflicts.empty:
        for _, row in conflicts.head().iterrows():
            print(f"\nConflict: {row['code1']} and {row['code2']}")
            print(f"Conflict Type: {row['conflict_type']}")
            print(f"Cluster: {row['cluster']}")
            print(f"Description 1: {row['desc1']}")
            print(f"Description 2: {row['desc2']}")
    else:
        print("No conflicts identified.")

    # Print highly similar procedures
    print("\n=== Highly Similar Procedures ===")
    top_similar = similarities.sort_values('similarity', ascending=False).head(10)
    for _, row in top_similar.iterrows():
        print(f"\nPair: {row['code1']} and {row['code2']}")
        print(f"Similarity: {row['similarity']:.4f}")
        print(f"Cluster: {row['cluster']}")