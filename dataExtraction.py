import pandas as pd
import os

# Define input and output paths
input_file = 'DataSet/table-of-surgical-procedures-(as-of-1-jan-2024).csv'
output_folder = 'DataSet/cleaned_files'
os.makedirs(output_folder, exist_ok=True)

# Read the input CSV
# Skip rows before 12, and limit rows to end at 2423
data = pd.read_csv(input_file, skiprows=11, nrows=(2423 - 12 + 1))

# Rename columns for clarity, skipping the empty columns
data = data[['S/N', 'Code', 'Description', 'Table', 'Classification']]

# Define the required columns
required_columns = ['S/N', 'Code', 'Description', 'Table', 'Classification']

# Initialize variables
output_files = []
current_file_data = []
current_file_index = 1
is_new_file = False  # Flag to check when to start a new file

# Iterate through rows to split based on "S/N" and valid starting conditions
for index, row in data.iterrows():
    # Check for a new "S/N" header (when 'S/N' appears in the first column and the next row starts with 1)
    if row['S/N'] == 'S/N':
        is_new_file = True
        continue

    # If a new file should be started and the row starts with 1
    if is_new_file and str(row['S/N']).isdigit() and int(row['S/N']) == 1:
        # Save the current file data if it's not empty
        if current_file_data:
            output_df = pd.DataFrame(current_file_data, columns=required_columns)
            output_file = os.path.join(output_folder, f'file_{current_file_index}.csv')
            output_df.to_csv(output_file, index=False)
            output_files.append(output_file)
            current_file_index += 1
            current_file_data = []

        # Reset the flag
        is_new_file = False

    # Add valid rows to the current file data
    current_file_data.append(row.values)

# Save the last file if it exists
if current_file_data:
    output_df = pd.DataFrame(current_file_data, columns=required_columns)
    output_file = os.path.join(output_folder, f'file_{current_file_index}.csv')
    output_df.to_csv(output_file, index=False)
    output_files.append(output_file)

# Display results
print(f"Created {len(output_files)} cleaned CSV files in the folder '{output_folder}'.")
