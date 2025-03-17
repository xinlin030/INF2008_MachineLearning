# Machine Learning Models

## Overview
The `2_Models` folder contains the primary machine learning models implemented for the project. The focus is on **KMeans, DBSCAN, Agglomerative Clustering,** and **NMF (Non-negative Matrix Factorization)** for clustering and data analysis.

## Folder Structure
```
2_Models/
│── 01_KMeans Clustering/
│   │── KMeans_Clustering.ipynb   # Jupyter Notebook for KMeans implementation
│   │── kmeans_clustering.py      # Python script for KMeans clustering
│   │── KMeans_Model.pkl          # Saved KMeans model
│── 02_DBScan/
│   │── DBSCAN_Clustering.ipynb   # Jupyter Notebook for DBSCAN
│   │── dbscan_clustering.py      # Python script for DBSCAN
│   │── DBSCAN_Model.pkl          # Saved DBSCAN model
│── 03_Agglomerative/
│   │── Agglomerative_Clustering.ipynb # Jupyter Notebook for Agglomerative clustering
│   │── agglomerative_clustering.py    # Python script for Agglomerative clustering
│   │── Agglomerative_Model.pkl        # Saved Agglomerative model
│   │── SJ_Endocrine.csv  # Sample dataset
│   │── SL_Eye.csv       # Sample dataset
│   │── test1.csv - test7.csv # Various test cases
│── 04_NMF/
│   │── NMF_Agglomerative_Model.pkl   # Saved model combining NMF & Agglomerative
│   │── Non_Matrix_Factorization_(NMF).ipynb # Jupyter Notebook for NMF
│   │── non_matrix_factorization_(nmf).py    # Python script for NMF
│── README.md                      # Documentation file
```

## Models Implemented
### **1. KMeans Clustering**
- Centroid-based clustering technique.
- Suitable for well-separated clusters.
- Implementations:
  - `kmeans_clustering.py` → Core Python script.
  - `KMeans_Clustering.ipynb` → Jupyter Notebook for interactive analysis.
  - `KMeans_Model.pkl` → Pre-trained model.

### **2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
- Detects clusters in spatial data with varying densities.
- Handles noise well.
- Implementations:
  - `dbscan_clustering.py` → Core DBSCAN script.
  - `DBSCAN_Clustering.ipynb` → Jupyter Notebook for interactive analysis.
  - `DBSCAN_Model.pkl` → Pre-trained model.

### **3. Agglomerative Hierarchical Clustering**
- Bottom-up hierarchical clustering approach.
- Creates a hierarchy of clusters based on similarity.
- Implementations:
  - `agglomerative_clustering.py` → Core Python script.
  - `Agglomerative_Clustering.ipynb` → Jupyter Notebook.
  - `Agglomerative_Model.pkl` → Pre-trained model.
  - Test datasets: `SJ_Endocrine.csv`, `SL_Eye.csv`, `test1.csv` - `test7.csv`.

### **4. Non-Negative Matrix Factorization (NMF)**
- Factorizes data into interpretable components.
- Often used for topic modeling and clustering.
- Implementations:
  - `non_matrix_factorization_(nmf).py` → Core Python script.
  - `Non_Matrix_Factorization_(NMF).ipynb` → Jupyter Notebook.
  - `NMF_Agglomerative_Model.pkl` → Model combining NMF and Agglomerative clustering.

## How to Run
1. Install dependencies:
   ```sh
   pip install numpy pandas scikit-learn
   ```
2. Navigate to the model directory (`01_KMeans Clustering/`, `02_DBScan/`, etc.).
3. Run the Python scripts:
   ```sh
   python kmeans_clustering.py
   ```
   or
   ```sh
   python dbscan_clustering.py
   ```
4. Use Jupyter Notebooks for interactive exploration:
   ```sh
   jupyter notebook KMeans_Clustering.ipynb
   ```

## Notes
- The `.pkl` files contain pre-trained models that can be loaded for evaluation.
- Test datasets are available in `03_Agglomerative/` for validation.

---
