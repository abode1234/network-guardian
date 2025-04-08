import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import joblib
import os

class DataPreprocessor:
    def __init__(self, logger):
        self.logger = logger
    
    def preprocess(self, df, label_column='Label'):
        """Clean and preprocess data."""
        self.logger.info("Preprocessing data...")
        
        # Handle missing/infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        # Binary encoding
        df[label_column] = df[label_column].apply(lambda x: 0 if x == 'BENIGN' else 1)
        
        return df
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        os.makedirs('./models', exist_ok=True)
        joblib.dump(scaler, './models/scaler.pkl')
        
        return X_train_scaled, X_test_scaled
    
    def balance_data(self, X, y):
        """Balance dataset using undersampling."""
        sampler = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        return X_balanced, y_balanced
