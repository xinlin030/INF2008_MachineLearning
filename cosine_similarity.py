from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/xinlin030/INF2008_MachineLearning/refs/heads/main/DataSets/CleanedDataset/SD_Cardiovascular.csv?token=GHSAT0AAAAAAC5BKA4K7NUTZSXBDOMVHSM4Z5CJOXQ"

df = pd.read_csv(url)
df.drop(columns=['S/N'])

sentences = df['Description'].to_list()


model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(sentences)

model = DBSCAN(eps=0.2, min_samples=1, metric='cosine') 
labels = model.fit_predict(embeddings)

print("Cluster labels:", labels)
df["Labels"] = labels

df.to_csv("out.csv", index=False)