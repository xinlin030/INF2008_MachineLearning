# Data Preprocessing Module

## Overview
This folder contains the necessary scripts and datasets for preprocessing surgical procedure data. The primary goal is to clean and organize raw datasets into structured CSV files for further analysis.

## Folder Structure
```
1_DataPreprocessing/
│── DataSets/
│   │── CleanedDataset/       # Processed datasets after cleaning
│   │── DBSCAN_Datasets/      # Datasets prepared for clustering (if applicable)
│   │── RawDataset/           # Original unprocessed datasets
│── dataExtraction.py         # Script for cleaning and organizing datasets
│── README.md                 # Documentation file
```

## Data Preprocessing Script: `dataExtraction.py`
This script processes the raw surgical procedure dataset and generates structured files for easier use.

### Key Features:
- Reads raw CSV file (`table-of-surgical-procedures-(as-of-1-jan-2024).csv`) from `RawDataset/`
- Cleans and formats data by:
  - Removing unnecessary rows
  - Merging split descriptions into a single row
  - Filtering only relevant entries based on procedure codes
- Categorizes procedures into specific body systems based on code prefixes
- Outputs cleaned datasets into `CleanedDataset/` as individual CSV files per category and a combined dataset

### File Naming Convention
Each file is named based on the first two characters of the procedure code:

| Prefix | Category |
|---------|----------------|
| SA | Integumentary |
| SB | Musculoskeletal |
| SC | Respiratory |
| SD | Cardiovascular |
| SE | Hemic & Lymphatic |
| SF | Digestive |
| SG | Urinary |
| SH | Male Genital |
| SI | Female Genital |
| SJ | Endocrine |
| SK | Nervous |
| SL | Eye |
| SM | ENT |

### Output Files:
- Individual category CSVs in `CleanedDataset/`
- A single `combined_dataset.csv` containing all cleaned records

## How to Run
1. Ensure Python and Pandas are installed:
   ```sh
   pip install pandas
   ```
2. Place the raw dataset CSV in `RawDataset/`
3. Run the script:
   ```sh
   python dataExtraction.py
   ```
4. Cleaned datasets will be available in `CleanedDataset/`

## Notes
- Ensure the input CSV follows the expected format before running the script.
- The script automatically creates necessary folders if they don’t exist.
