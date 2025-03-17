# Model Evaluation Results

## Overview
The `3_Results` folder contains the saved models and evaluation results from different clustering algorithms. These models can be used for further evaluation and inference.

## Folder Structure
```
3_Results/
│── Agglomerative_Model.pkl       # Pre-trained Agglomerative clustering model
│── DBSCAN_Model.pkl              # Pre-trained DBSCAN model
│── NMF_Agglomerative_Model.pkl   # Pre-trained NMF + Agglomerative model
│── final_model_comparisons.csv   # Model performance summary
│── README.md                     # Documentation file
```

## **File Descriptions**
### **1. Pre-Trained Models (`.pkl` files)**
These files contain pre-trained clustering models that can be loaded for evaluation and inference.
- `Agglomerative_Model.pkl`: Trained Agglomerative clustering model.
- `DBSCAN_Model.pkl`: Trained DBSCAN model.
- `NMF_Agglomerative_Model.pkl`: A hybrid model combining NMF and Agglomerative clustering.

### **2. `final_model_comparisons.csv`**
This file contains a summary of model performance across different test cases.

## **How to Use the Results**
1. **Load a trained model for evaluation:**
   ```python
   import pickle
   with open("DBSCAN_Model.pkl", "rb") as model_file:
       model = pickle.load(model_file)
   ```
2. **Analyze model performance using the `.pkl` files and comparison results.**
3. **Use the models for further inference or refinement.**

## **Conclusion**
- The saved models can be used for further analysis and testing.
- Different clustering methods provide varied results, and the best model depends on the dataset characteristics.
- Future work can focus on fine-tuning and improving clustering accuracy.

---
