import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
df = pd.read_csv("DataSets/CleanedDataset/combined_dataset.csv")  # Change to your actual filename

# Keep only the 'Description' column
descriptions = df["Description"].astype(str)

# Initialize preprocessing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Tokenize words
    words = word_tokenize(text)
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Reconstruct cleaned text
    return " ".join(words)

# Apply preprocessing
df["Cleaned_Description"] = descriptions.apply(preprocess_text)

# Save cleaned dataset
df.to_csv("DataSets/DBSCAN_Datasets/cleaned_base_dataset.csv", index=False)

###############################

# Load the cleaned dataset
df = pd.read_csv("DataSets/DBSCAN_Datasets/cleaned_base_dataset.csv")

# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=500)  # Limit features for efficiency
tfidf_matrix = vectorizer.fit_transform(df["Cleaned_Description"])

# Reduce dimensions using PCA
pca = PCA(n_components=2, random_state=42)
reduced_embeddings = pca.fit_transform(tfidf_matrix.toarray())

# Store the reduced embeddings
df["PCA_X"] = reduced_embeddings[:, 0]
df["PCA_Y"] = reduced_embeddings[:, 1]

# Split data into training (80%) and validation (20%)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the training and validation sets
train_df.to_csv("DataSets/DBSCAN_Datasets/train_data.csv", index=False)
val_df.to_csv("DataSets/DBSCAN_Datasets/val_data.csv", index=False)

print("Feature engineering complete using TF-IDF + PCA.")
print("Training set saved as 'train_data.csv'. Validation set saved as 'val_data.csv'.")

####################

# Load the training data
df = pd.read_csv("DataSets/DBSCAN_Datasets/train_data.csv")

# Extract PCA-reduced features
X = df[["PCA_X", "PCA_Y"]].values

# Step 1: Determine optimal eps using k-distance plot
nearest_neighbors = NearestNeighbors(n_neighbors=5)
neighbors = nearest_neighbors.fit(X)
distances, indices = neighbors.kneighbors(X)

# Sort distances for k-distance graph
distances = np.sort(distances[:, 4], axis=0)

# Plot k-distance graph
plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.xlabel("Data Points (sorted)")
plt.ylabel("5th Nearest Neighbor Distance")
plt.title("K-Distance Graph for Optimal eps")
plt.show()

# Step 2: Set min_samples and apply DBSCAN ( manual hyperparameters :( )
eps = float(input("Enter optimal eps value from the graph: "))  # Manual input based on the plot
min_samples = 7  # Default, can be adjusted

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df["Cluster"] = dbscan.fit_predict(X)

# Save clustered dataset
df.to_csv("DataSets/DBSCAN_Datasets/dbscan_clustered.csv", index=False)

print("DBSCAN clustering complete. Results saved as 'dbscan_clustered.csv'.")

# Load the DBSCAN clustered data
df = pd.read_csv("DataSets/DBSCAN_Datasets/dbscan_clustered.csv")

# Extract the features (PCA-reduced)
X = df[["PCA_X", "PCA_Y"]].values

# Get the cluster labels (DBSCAN assigns -1 for noise)
labels = df["Cluster"].values

# Step 1: Calculate silhouette score
sil_score = silhouette_score(X, labels)
print(f"Silhouette Score: {sil_score:.3f}")

# Step 2: Visualize the clustering with colors
plt.scatter(df["PCA_X"], df["PCA_Y"], c=labels, cmap="Spectral", marker="o", edgecolor="k")
plt.title(f"DBSCAN Clustering (Silhouette Score: {sil_score:.3f})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")

# Save the plot as an image file
plt.savefig("dbscan_clustering_visualization.png")

plt.show()

print("Clustering visualization saved as 'dbscan_clustering_visualization.png'.")

# Here’s how to interpret the score:

# 0.6 - 0.7: Fairly good clustering. The points are reasonably well-clustered, and the separation between clusters is decent.
# 0.7 - 1.0: Very good clustering. The points are well-grouped within clusters, and the clusters are clearly defined.
# 0.0 - 0.6: Clustering can be improved. The clusters might be overlapping or not very distinct.
# Negative values: Points are likely misclassified or the clustering algorithm might not be appropriate.