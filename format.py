import pandas as pd
import shutil

# Dosya yolu
filename = '/Users/ranaturker/PycharmProjects/Capstone/hma_kinesics/data_in/397.poses_rad'

# Backup the original file
shutil.copy(filename, filename + '.backup')

corrected_lines = []

# Read and validate each line
with open(filename, 'r') as file:
    lines = file.readlines()

for i, line in enumerate(lines):
    columns = line.strip().split(',')

    # Handle inconsistent rows
    if len(columns) == 4:
        corrected_lines.append(line)
    elif len(columns) < 4:
        # Log incomplete rows for review
        print(f"Row {i + 1} is incomplete: {line.strip()}")
        columns += [''] * (4 - len(columns))  # Fill missing columns
        corrected_lines.append(','.join(columns) + '\n')
    else:
        # Log excess columns for review
        print(f"Row {i + 1} has extra columns: {line.strip()}")
        corrected_lines.append(','.join(columns[:4]) + '\n')

# Write the cleaned data back to the file
with open(filename, 'w') as file:
    file.writelines(corrected_lines)

# Validate by reading with pandas
df = pd.read_csv(filename)
print(df.head())

# Check for any remaining issues
if df.isnull().any().any():
    print("Warning: There are still NaN or invalid entries in the cleaned file.")
else:
    print("File cleaned successfully and loaded into DataFrame.")
