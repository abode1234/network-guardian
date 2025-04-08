import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Define file paths for the model, scaler, and test dataset
model_path = './test/test_model/xgboost_netguardian_large_data.pkl'
scaler_path = './models/scaler.pkl'
data_path = './test_model/archive/WebAttacks-Thursday-no-metadata.parquet'

# Check if files exist
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path.")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Please check the path.")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at {data_path}. Please check the path.")

# Load the trained model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print(f"Model loaded successfully from {model_path}")
print(f"Scaler loaded successfully from {scaler_path}")

# Load the test dataset
df = pd.read_parquet(data_path)
print(f"Test data loaded successfully from {data_path}")
print("Dataset preview:")
print(df.head(5))

# Preprocess the test data
df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df.dropna(inplace=True)

# Standardize column names in test data by removing leading/trailing spaces initially
df.columns = df.columns.str.strip()

# Check if the column name is 'Label' or ' Label' and rename to match training format
if 'Label' in df.columns:
    df.rename(columns={'Label': ' Label'}, inplace=True)  # Match training data format (' Label')
elif ' Label' not in df.columns:
    raise ValueError("Column 'Label' or ' Label' not found. Ensure the dataset contains one of these columns.")

# Convert labels to binary (0 for BENIGN, 1 for others) using the correct column
df[' Label'] = df[' Label'].apply(lambda x: 0 if x.strip().upper() == 'BENIGN' else 1)

# Get the exact feature names expected by the scaler (preserving spaces)
expected_features = [col for col in scaler.feature_names_in_ if col not in ['Label', ' Label', 'source_file']]

# Create a DataFrame with all expected features, filling missing ones with 0
X = pd.DataFrame(columns=expected_features, index=df.index)

# Map test data columns to expected feature names (removing spaces for matching)
test_cols_stripped = {col.strip(): col for col in df.columns}
for col in expected_features:
    stripped_col = col.strip()
    if stripped_col in test_cols_stripped:
        X[col] = df[test_cols_stripped[stripped_col]]
    else:
        print(f"Warning: Column '{col}' missing in test data, filling with 0")
        X[col] = 0

# Separate features and labels
y = df[' Label']

# Scale the features using the pre-trained scaler
X_scaled = scaler.transform(X)
print("Test data scaling completed using pre-trained scaler")

# Make predictions directly using XGBClassifier
y_pred = model.predict(X_scaled)
print("Predictions completed")

# Print some diagnostics
print("True labels (y) for first 10 samples:", y[:10].values)
print("Predicted labels (y_pred) for first 10 samples:", y_pred[:10])
print("Count of predicted 0s:", sum(y_pred == 0))
print("Count of predicted 1s:", sum(y_pred == 1))
print("Total samples:", len(y))

# Calculate and display accuracy
accuracy = accuracy_score(y, y_pred)
print(f"True Accuracy: {accuracy:.4f}")
