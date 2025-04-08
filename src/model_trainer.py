import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib
import time
import os

class ModelTrainer:
    def __init__(self, logger):
        self.logger = logger
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model with hyperparameter tuning."""
        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            tree_method='hist',
            random_state=42
        )
        
        param_dist = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'scale_pos_weight': [scale_pos_weight],
            'n_estimators': [100, 200]
        }
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            scoring='f1',
            n_iter=10,
            cv=3,
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X_train, y_train, 
                 eval_set=[(X_test, y_test)],
                 early_stopping_rounds=10,
                 verbose=0)
        
        return search.best_estimator_
    
    def save_model(self, model, filename='xgboost_model.pkl'):
        """Save trained model to disk."""
        os.makedirs('./models', exist_ok=True)
        path = f'./models/{filename}'
        joblib.dump(model, path)
        model.save_model('/home/snorpiii/pro/AI/AI-cyber-security/NetGuardian/models/xgboost_model.pkl')
        return path
