# Machine Learning Models

## Overview
The `2_Models` folder contains the core machine learning algorithms implemented for the project. The primary focus is on **DBSCAN, NMF, KMeans,** and **Agglomerative Clustering**, while alternative models that were considered but not used are stored in the `Archives` folder.

## Folder Structure
```
2_Models/
│── Archives/                 # Algorithms that were considered but not used
│   │── DivisiveHierarchicalClustering.py
│   │── tf_idf_Random_Forest.py
│   │── XGBoost.py
│── DBScan/                    # DBSCAN clustering implementations
│   │── dbscanFinetune.py
│   │── r_dbscan.ipynb
│   │── r_dbscan.py
│── KMeans/                    # KMeans clustering implementations
│   │── r_kmeans.ipynb
│   │── r_kmeans.py
│── testCases/                 # Test cases for model evaluation
│   │── DBSCAN (3 reduction techniques)/
│   │── INF2008_MachineLearning/
│── README.md                  # Documentation file
```

## Chosen Models
### **1. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
- Used for discovering clusters in spatial data.
- Handles noise and varying cluster density.
- Fine-tuning available in `dbscanFinetune.py`.
- Implementations:
  - `r_dbscan.py` → Core DBSCAN clustering script.
  - `r_dbscan.ipynb` → Jupyter Notebook for interactive analysis.

### **2. KMeans Clustering**
- A centroid-based clustering approach.
- Suitable for well-separated clusters.
- Implementations:
  - `r_kmeans.py` → Core KMeans clustering script.
  - `r_kmeans.ipynb` → Jupyter Notebook for analysis.

### **3. NMF (Non-negative Matrix Factorization)**
- Used for dimensionality reduction and topic modeling.
- Helps extract hidden structures in data.

### **4. Agglomerative Hierarchical Clustering**
- Bottom-up hierarchical clustering approach.
- Creates a hierarchy of clusters based on similarity.

## Archived Models
These models were tested but not used in the final implementation:
- **Divisive Hierarchical Clustering**
- **Random Forest with TF-IDF**
- **XGBoost**

## How to Run
1. Install dependencies:
   ```sh
   pip install numpy pandas scikit-learn
   ```
2. Navigate to the model directory (`DBScan/` or `KMeans/`).
3. Run the Python scripts:
   ```sh
   python r_dbscan.py
   ```
   or
   ```sh
   python r_kmeans.py
   ```
4. Use Jupyter Notebooks for interactive exploration:
   ```sh
   jupyter notebook r_dbscan.ipynb
   ```

## Notes
- The `testCases/` folder contains test scenarios and validation cases for clustering models.
- The scripts are designed to be modular for further tuning and comparison.

---
