import pandas as pd
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv("DataSets/CleanedDataset/combined_dataset.csv")
print("Dataset loaded successfully.")

# Display first few rows
print(df.head())

# Generate all possible pairs of codes
pairs = list(itertools.combinations(df['Code'], 2))

# Convert to DataFrame
pairs_df = pd.DataFrame(pairs, columns=['Code1', 'Code2'])

# Merge descriptions
pairs_df = pairs_df.merge(df[['Code', 'Description']], left_on='Code1', right_on='Code', how='left').rename(columns={'Description': 'Description1'}).drop(columns=['Code'])
pairs_df = pairs_df.merge(df[['Code', 'Description']], left_on='Code2', right_on='Code', how='left').rename(columns={'Description': 'Description2'}).drop(columns=['Code'])

print(pairs_df.head())

# Combine all descriptions
descriptions = df['Description'].tolist()

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(descriptions)

# Convert TF-IDF matrix to a dense NumPy array
tfidf_array = tfidf_matrix.toarray()

# Compute cosine similarity **once** for all descriptions
similarity_matrix = cosine_similarity(tfidf_array)

# Convert to a DataFrame (faster lookups)
similarity_df = pd.DataFrame(similarity_matrix, index=df['Code'], columns=df['Code'])

# Function to fetch similarity scores efficiently
def get_similarity(code1, code2):
    return similarity_df.at[code1, code2]

# Apply the function vectorized (MUCH faster than apply + loop)
pairs_df['Similarity'] = pairs_df.apply(lambda row: get_similarity(row['Code1'], row['Code2']), axis=1)

# print(pairs_df.head())

# pairs_df.to_csv('similarity_pairs.csv', index=False)

# Define similarity threshold (adjust as needed)
SIMILARITY_THRESHOLD = 0.75

# Filter pairs with high similarity
highly_similar_pairs = pairs_df[pairs_df['Similarity'] > SIMILARITY_THRESHOLD]

# Save to CSV
highly_similar_pairs.to_csv('highly_similar_pairs.csv', index=False)

print(highly_similar_pairs.head())

# DBSCAN clustering on TF-IDF vectors
dbscan = DBSCAN(eps=0.55, min_samples=5, metric="cosine")  # Adjust `eps`
labels = dbscan.fit_predict(tfidf_matrix)

# ---- DBSCAN clustering with optimized parameters ----
# Function to calculate the silhouette score for different `eps` and `min_samples`
# def calculate_silhouette(eps_values, min_samples_values, tfidf_matrix):
#     best_silhouette = -1
#     best_params = (None, None)
    
#     for eps in eps_values:
#         for min_samples in min_samples_values:
#             dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
#             labels = dbscan.fit_predict(tfidf_matrix)
#             if len(set(labels)) > 1:  # At least two clusters are needed
#                 score = silhouette_score(tfidf_matrix, labels, metric='cosine')
#                 if score > best_silhouette:
#                     best_silhouette = score
#                     best_params = (eps, min_samples)
    
#     return best_silhouette, best_params

# # Define range for `eps` and `min_samples`
# eps_values = np.linspace(0.1, 1.0, 10)  # Adjust as needed
# min_samples_values = [3, 4, 5, 6, 7]   # Adjust as needed

# # Get the best `eps` and `min_samples` based on silhouette score
# best_silhouette, best_params = calculate_silhouette(eps_values, min_samples_values, tfidf_matrix)

# print(f"Best silhouette score: {best_silhouette} with eps = {best_params[0]} and min_samples = {best_params[1]}")

# # Apply DBSCAN with the best parameters
# dbscan = DBSCAN(eps=best_params[0], min_samples=best_params[1], metric="cosine")
# labels = dbscan.fit_predict(tfidf_matrix)

# Use 4 nearest neighbors (rule of thumb for DBSCAN)
neighbors = NearestNeighbors(n_neighbors=4, metric="cosine")
neighbors_fit = neighbors.fit(tfidf_matrix)
distances, indices = neighbors_fit.kneighbors(tfidf_matrix)

# Sort and plot distances (4th nearest neighbor distance)
distances = np.sort(distances[:, -1])  
plt.plot(distances)
plt.xlabel("Points sorted by distance")
plt.ylabel("4th Nearest Neighbor Distance")
plt.title("Elbow Method for Finding `eps`")
plt.show()

# eps_values = [0.3, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21, 0.2]  # Try decreasing step by step

# for eps in eps_values:
#     dbscan = DBSCAN(eps=eps, min_samples=5, metric="cosine")  # Keep min_samples fixed for now
#     df["Cluster"] = dbscan.fit_predict(similarity_matrix)
    
#     unique_clusters, counts = np.unique(df["Cluster"], return_counts=True)
#     print(f"\nFor eps = {eps}:")
#     print("Unique Clusters Found:", unique_clusters)
#     print("Cluster Counts:", dict(zip(unique_clusters, counts)))

# # Define the range of eps and min_samples values to test
# eps_values = [0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31]
# min_samples_values = [5, 6, 7, 8, 9]

# # Initialize variables to store the best score and corresponding parameters
# best_score = -1
# best_eps = None
# best_min_samples = None

# # DataFrame to hold the results for visualization (optional)
# results = []

# # Loop over different combinations of eps and min_samples
# for eps in eps_values:
#     for min_samples in min_samples_values:
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")  # DBSCAN model
#         df["Cluster"] = dbscan.fit_predict(similarity_matrix)
        
#         # Exclude noise points (-1) from silhouette score calculation
#         labels = df["Cluster"].values
#         if len(np.unique(labels)) > 1:  # More than 1 cluster (excluding noise)
#             # Filter out points labeled as noise (-1)
#             filtered_labels = labels[labels != -1]
#             filtered_similarity_matrix = similarity_matrix[labels != -1]
            
#             # Calculate silhouette score
#             score = silhouette_score(filtered_similarity_matrix, filtered_labels, metric="cosine")
#             results.append({"eps": eps, "min_samples": min_samples, "silhouette_score": score})
            
#             # Update best score and parameters
#             if score > best_score:
#                 best_score = score
#                 best_eps = eps
#                 best_min_samples = min_samples
#         else:
#             results.append({"eps": eps, "min_samples": min_samples, "silhouette_score": None})

# # Print the best eps and min_samples based on silhouette score
# print(f"\nBest eps: {best_eps}, Best min_samples: {best_min_samples}, Best silhouette score: {best_score}")

# # Convert the results to a DataFrame for better visualization
# results_df = pd.DataFrame(results)

# # Optionally: Show the results as a table
# print("\nSilhouette Scores for Different eps and min_samples Combinations:")
# print(results_df)

# # If you want to visualize the results:
# import matplotlib.pyplot as plt
# pivot_results = results_df.pivot(index='min_samples', columns='eps', values='silhouette_score')

# # Plot a heatmap of silhouette scores for different eps and min_samples values
# plt.figure(figsize=(10, 8))
# plt.title("Silhouette Scores for Different eps and min_samples")
# plt.xlabel("eps")
# plt.ylabel("min_samples")
# plt.imshow(pivot_results, cmap='coolwarm', aspect='auto', interpolation='nearest')
# plt.colorbar(label='Silhouette Score')
# plt.xticks(ticks=np.arange(len(eps_values)), labels=eps_values)
# plt.yticks(ticks=np.arange(len(min_samples_values)), labels=min_samples_values)
# plt.show()

min_samples = int(np.log(len(df)))  # df contains all the surgical codes
print("Recommended min_samples:", min_samples)

# Add cluster labels to the original data
df['Cluster'] = labels

# Save clusters for analysis
df.to_csv('clustered_codes.csv', index=False)

print(df[df['Cluster'] != -1].head())  # Show meaningful clusters

# Reduce TF-IDF vectors to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
reduced_vectors = tsne.fit_transform(tfidf_matrix.toarray())

# Plot clusters
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=df['Cluster'], cmap='viridis')
plt.colorbar(label="Cluster Label")
plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.title("TOSP Code Clusters")
plt.show()

unique_clusters, counts = np.unique(df['Cluster'], return_counts=True)
print("Unique Clusters Found:", unique_clusters)
print("Cluster Counts:", dict(zip(unique_clusters, counts)))

# outliers = df[df['Cluster'] == -1]
# print("Number of outliers:", len(outliers))
# print(outliers[['Code', 'Description']].head(10))

for cluster in np.unique(df['Cluster']):
    print(f"\nCluster {cluster}:")
    print(df[df['Cluster'] == cluster][['Code', 'Description']].head(10))  # Show first 10 items