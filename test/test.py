import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Define file paths for the model, scaler, and test dataset
model_path = './test/test_model/xgboost_netguardian_large_data.pkl'
scaler_path = './models/scaler.pkl'
data_path = './test/test_model/Wednesday-workingHours.pcap_ISCX.csv'

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
df = pd.read_csv(data_path)
print(f"Test data loaded successfully from {data_path}")
print("Dataset preview:")
print(df.head(5))

# Preprocess the test data
df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df.dropna(inplace=True)

# Check if the column name is ' Label' instead of 'Label'
if ' Label' not in df.columns:
    if 'Label' not in df.columns:
        raise ValueError("Column 'Label' or ' Label' not found. Ensure the dataset contains one of these columns.")
    else:
        # Rename 'Label' to ' Label' if found
        df.rename(columns={'Label': ' Label'}, inplace=True)

# Convert labels to binary (0 for BENIGN, 1 for others)
df[' Label'] = df[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# Separate features and labels
X = df.drop(columns=[' Label', 'source_file'], errors='ignore')
y = df[' Label']

# Scale the features using the pre-trained scaler
X_scaled = scaler.transform(X)
print("Test data scaling completed using pre-trained scaler")

# Make predictions directly using XGBClassifier (no DMatrix needed)
y_pred = model.predict(X_scaled)
print("Predictions completed")

# Calculate and display accuracy
accuracy = accuracy_score(y, y_pred)
print(f"True Accuracy: {accuracy:.4f}")
