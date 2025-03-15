# INF2008 P5 Group 12 Machine Learning Project  

## 📌 Problem Description  
Healthcare affordability is one of the key concerns of Singaporeans. Medical waste and abuse from inappropriate claims contribute to escalating costs, without benefiting patients. The [**Table of Surgical Procedures (TOSP)**](https://isomer-user-content.by.gov.sg/3/ca783b21-2842-4431-b2f0-3934be261852/table-of-surgical-procedures-(as-of-1-jan-2024).pdf) is an exhaustive list of procedures for which*MediSave / MediShield Life can be claimed.  

One common issue is that doctors may inadvertently submit multiple TOSP codes for a **single** surgical procedure instead of using a single, comprehensive code. For example, submitting separate codes for different components of Whipple’s procedure instead of the correct consolidated code.  

## 🎯 Task  

The goal of this project is to **detect and flag inappropriate TOSP code pairs** that are mutually exclusive using classical machine learning algorithms.  

#### 🔍 Examples of invalid code pairs:
- **"BILATERAL" vs "UNILATERAL"**  
- **"MULTIPLE" vs "SINGLE"**  
- **"MAJOR" vs "MINOR"**  
- **TOSP codes referring to the same procedure but submitted together**  

By identifying such invalid pairs, we aim to improve medical claim accuracy and reduce unnecessary healthcare costs. 

## 📂 File Structure  

The repository is organized into four main directories:  

```
📦 INF2008_P5_Group12
┣ 📂 1_DataProcessing
┣ 📂 2_Models
┣ 📂 3_Results
┣ 📂 4_Others
┗ README.md
```

### 📌 1_DataProcessing  
Contains Colab notebooks detailing how we extracted the **TOSP codes** from the PDF file into a structured `.csv` format for further analysis.  

### 📌 2_Models  
Includes separate folders and notebooks for different machine learning models:  
- **KMeans**  
- **DBSCAN**  
- **Agglomerative Clustering**  
- **Non-negative Matrix Factorization (NMF)**  

Each model has its own folder and dedicated notebook that outlines the methodology, training process, and results. The model was also download so that the code does not need to be re-ran again.  

### 📌 3_Results  
A notebook that **compares the cosine similarity scores** for all four models and evaluates their effectiveness in detecting inappropriate TOSP code pairs.  

### 📌 4_Others  
This directory contains intermediate work contributed by different team members. The final models in ** 📂 2_Models** were built by consolidating these efforts.  



## 🚀 How to Run the Project  

Running this project will require Google Colab. This project has already been executed, and all results have been pre-captured in the notebooks. **You do not need to rerun the code** unless you wish to modify or test new parameters.    

### **📌 Steps to Run:**
1️⃣ **Upload the notebook to Google Colab**  
   - Open **Google Colab** ([colab.research.google.com](https://colab.research.google.com))  
   - Click **File > Upload Notebook**  
   - Select the `.ipynb` file you want to run  

2️⃣ **Upload the required `.csv` files**  
   - Each notebook requires specific `.csv` files to run (combined_dataset.csv)  
   - Go to the **Files tab** in Colab (`📁` icon on the left)  
   - Upload the necessary `.csv` files  

3️⃣ **Run the notebook**  
   - Click **"Run All"** (`Runtime > Run all`)
   - The notebook is annotated with step-by-step explanations  

4️⃣ **Review the results**  
   - Outputs will be displayed in text, plots, and tables 
   - Results will include clustering performance and similarity scores 

## 📢 Contact & Contributions  
If there are issues with the notebooks, please reach out to **INF2008 P5 Group 12**.