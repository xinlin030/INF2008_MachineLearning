import pandas as pd
import os

# Define input and output paths
input_file = './DataSets/RawDataset/table-of-surgical-procedures-(as-of-1-jan-2024).csv'
output_folder = './DataSets/CleanedDataset'
os.makedirs(output_folder, exist_ok=True)

# Mapping for file names based on prefixes
file_name_mapping = {
    "SA": "Integumentary",
    "SB": "Musculoskeletal",
    "SC": "Respiratory",
    "SD": "Cardiovascular",
    "SE": "Hemic & Lymphatic",
    "SF": "Digestive",
    "SG": "Urinary",
    "SH": "Male Genital",
    "SI": "Female Genital",
    "SJ": "Endocrine",
    "SK": "Nervous",
    "SL": "Eye",
    "SM": "ENT"
}

# Read the input CSV
# Skip rows before 12, and read all remaining rows
data = pd.read_csv(input_file, skiprows=11)

# Rename columns for clarity, skipping the empty columns
data = data[['S/N', 'Code', 'Description', 'Table', 'Classification']]

# Define the required columns
required_columns = ['S/N', 'Code', 'Description', 'Table', 'Classification']

# Process rows: combine descriptions and remove invalid rows
merged_data = []
current_row = None

for _, row in data.iterrows():
    # If the description is empty, skip this row entirely
    if pd.isna(row['Description']):
        continue

    # If S/N is empty, append the description to the previous row
    if pd.isna(row['S/N']):
        if current_row is not None:
            current_row['Description'] += f" {row['Description']}"  # Append the description
    else:
        # If a new row starts, save the current row and move to the next
        if current_row is not None:
            merged_data.append(current_row)
        current_row = row.copy()

# Append the last row
if current_row is not None:
    merged_data.append(current_row)

# Convert the merged rows back to a DataFrame
data = pd.DataFrame(merged_data)

# Filter out rows where essential columns are invalid
data = data.dropna(subset=['S/N', 'Code', 'Description'])  # Ensure no invalid rows remain
data = data[data['Code'].apply(lambda x: str(x).startswith(tuple(file_name_mapping.keys())))]  # Keep rows with valid Code prefixes
data['Description'] = data['Description'].str.replace(r'[\r\n]+', ' ', regex=True)

# Initialize variables
output_files = []
current_file_data = []
combined_data = []  # To store all data for the combined file
current_file_index = 1
new_file_triggered = False
current_file_prefix = None

# Iterate through rows to split based on "Code" and valid starting conditions
for index, row in data.iterrows():
    # Check for a new "Code" prefix to trigger a new file
    row_prefix = row['Code'][:2]  # Extract the prefix from the 'Code' column

    if current_file_prefix is None:
        current_file_prefix = row_prefix

    # If the prefix changes, save the current file
    if row_prefix != current_file_prefix:
        if current_file_data:
            if current_file_prefix in file_name_mapping:
                file_name = f"{current_file_prefix}_{file_name_mapping[current_file_prefix]}"
            else:
                file_name = f"file_{current_file_index}"
            output_file = os.path.join(output_folder, f'{file_name}.csv')
            output_df = pd.DataFrame(current_file_data, columns=required_columns)
            output_df.to_csv(output_file, index=False)
            output_files.append(output_file)
            current_file_index += 1
            current_file_data = []

        # Update the current file prefix
        current_file_prefix = row_prefix

    # Add valid rows to the current file data and the combined data
    current_file_data.append(row.values)
    combined_data.append(row.values)

# Explicitly save the last file if any data remains
if current_file_data:
    if current_file_prefix in file_name_mapping:
        file_name = f"{current_file_prefix}_{file_name_mapping[current_file_prefix]}"
    else:
        file_name = f"file_{current_file_index}"
    output_file = os.path.join(output_folder, f'{file_name}.csv')
    output_df = pd.DataFrame(current_file_data, columns=required_columns)
    output_df.to_csv(output_file, index=False)
    output_files.append(output_file)

# Save the combined data into a single file
combined_output_file = os.path.join(output_folder, 'combined_dataset.csv')
combined_df = pd.DataFrame(combined_data, columns=required_columns)
combined_df.to_csv(combined_output_file, index=False)

# Display results
print(f"Created {len(output_files)} cleaned CSV files in the folder '{output_folder}', including a combined file.")