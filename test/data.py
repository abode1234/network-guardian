import pandas as pd
import os

# Define file paths for the datasets
train_data_path = './dataset/clean_dataset.csv'  # Path to the training dataset
test_data_path = './test_model/archive/WebAttacks-Thursday-no-metadata.parquet'  # Path to the test dataset

# Check if files exist
if not os.path.exists(train_data_path):
    raise FileNotFoundError(f"Training data file not found at {train_data_path}. Please check the path.")
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test data file not found at {test_data_path}. Please check the path.")

# Load the datasets
train_df = pd.read_csv(train_data_path)
test_df = pd.read_parquet(test_data_path)

# Get column names (stripping spaces to handle inconsistencies)
train_columns = set(train_df.columns.str.strip())
test_columns = set(test_df.columns.str.strip())

# Print column names
print("Columns in clean_dataset.csv (Training Data):")
print(sorted(train_columns))
print(f"Number of columns: {len(train_columns)}\n")

print("Columns in Benign-Monday-no-metadata.parquet (Test Data):")
print(sorted(test_columns))
print(f"Number of columns: {len(test_columns)}\n")

# Compare columns
common_columns = train_columns.intersection(test_columns)
train_only_columns = train_columns - test_columns
test_only_columns = test_columns - train_columns

print("Common columns between the two datasets:")
print(sorted(common_columns))
print(f"Number of common columns: {len(common_columns)}\n")

print("Columns in clean_dataset.csv but not in Benign-Monday-no-metadata.parquet:")
print(sorted(train_only_columns))
print(f"Number of unique training columns: {len(train_only_columns)}\n")

print("Columns in Benign-Monday-no-metadata.parquet but not in clean_dataset.csv:")
print(sorted(test_only_columns))
print(f"Number of unique test columns: {len(test_only_columns)}")
