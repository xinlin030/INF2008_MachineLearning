import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Download necessary NLP resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
df = pd.read_csv("DataSets/CleanedDataset/combined_dataset.csv")
print("Dataset loaded successfully.")

# --- Improved Text Preprocessing --- #
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization & stopword removal
    return " ".join(words)

df["Cleaned_Description"] = df["Description"].apply(preprocess_text)

# --- Text Similarity Using TF-IDF --- #
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Cleaned_Description"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Identify highly similar procedure pairs
similar_pairs = [
    (df.iloc[i]["Code"], df.iloc[j]["Code"], cosine_sim[i, j])
    for i, j in combinations(range(len(df)), 2) if cosine_sim[i, j] > 0.8
]

similar_pairs_df = pd.DataFrame(similar_pairs, columns=["Code1", "Code2", "Similarity"])
print(f"Found {len(similar_pairs)} highly similar procedure pairs.")
similar_pairs_df.to_csv("highly_similar_procedure_pairs.csv", index=False)
print("Saved highly similar procedure pairs to highly_similar_procedure_pairs.csv.")

# --- Clustering (K-Means) --- #
tfidf_dense = tfidf_matrix.toarray()

# Find optimal K using Silhouette Score
silhouette_scores = []
K_values = range(500, 2000, 100)  # Testing different k values

for k in K_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(tfidf_dense)
    score = silhouette_score(tfidf_dense, kmeans.labels_)
    silhouette_scores.append(score)

# Plot Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(K_values, silhouette_scores, marker="o", linestyle="-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score to Determine Optimal k")
plt.show()

# Find the best k (the one with the highest silhouette score)
optimal_k = K_values[np.argmax(silhouette_scores)]
print(f"The optimal number of clusters based on Silhouette Score is: {optimal_k}")

# Perform K-Means with the optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df["KMeans_Cluster"] = kmeans.fit_predict(tfidf_dense)

# --- Visualizing Clustering --- #
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df.index, y=df["KMeans_Cluster"], hue=df["KMeans_Cluster"], palette="viridis")
plt.title("K-Means Clustering of Procedures")
plt.xlabel("Procedure Index")
plt.ylabel("Cluster ID")
plt.show()

# --- Detecting Inconsistent Pairs --- #
df["Code_Index"] = df.index

# Merge cluster data
cluster_df = df[["Code", "KMeans_Cluster"]].copy()
cosine_sim_df = pd.DataFrame(cosine_sim, index=df["Code"], columns=df["Code"])

# Identify same-cluster pairs
same_cluster_pairs = pd.merge(cluster_df, cluster_df, on="KMeans_Cluster")
same_cluster_pairs = same_cluster_pairs[same_cluster_pairs["Code_x"] != same_cluster_pairs["Code_y"]]

# Add similarity scores
same_cluster_pairs["Similarity"] = same_cluster_pairs.apply(
    lambda row: cosine_sim_df.loc[row["Code_x"], row["Code_y"]], axis=1
)

# Filter inconsistent pairs
inconsistent_pairs = same_cluster_pairs[same_cluster_pairs["Similarity"] < 0.2]

print(f"Found {len(inconsistent_pairs)} potentially inconsistent procedure pairs.")
inconsistent_pairs.to_csv("inconsistent_tosp_pairs.csv", index=False)
print("Results saved to inconsistent_tosp_pairs.csv")
