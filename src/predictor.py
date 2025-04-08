import joblib
import numpy as np

class Predictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load('./models/scaler.pkl')
    
    def predict(self, X):
        """Make predictions on new data."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
