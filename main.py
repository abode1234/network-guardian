from src.utils import setup_logger
from src.datset import DatasetLoader
from src.preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
import time
import gc

def main():
    logger = setup_logger("NetGuardian")
    logger.info("Starting NetGuardian training pipeline")
    
    try:
        # Load data
        loader = DatasetLoader(logger)
        df = loader.load_in_chunks('./dataset/clean_dataset.csv')
        
        # Preprocess
        preprocessor = DataPreprocessor(logger)
        df = preprocessor.preprocess(df)
        
        # Split data
        X = df.drop(columns=['Label'])
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        del df
        gc.collect()
        
        # Scale features
        X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
        
        # Balance data
        X_balanced, y_balanced = preprocessor.balance_data(X_train_scaled, y_train)
        
        # Train model
        trainer = ModelTrainer(logger)
        model = trainer.train(X_balanced, y_balanced, X_test_scaled, y_test)
        
        # Save model
        model_path = trainer.save_model(model)
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
