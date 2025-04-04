import pandas as pd
from sklearn.base import re
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import setup_logging, save_plot

logger = setup_logging()

class Preprocessor:
    def __init__(self, label_column='Label'):
        self.scaler = StandardScaler()
        self.top_features = None
        self.label_column = label_column

    def clean_data(self, df):
        df_raw = df.copy()
        df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
        df.dropna(inplace=True)

        logger.info(f"Columns in dataset: {list(df.columns)} \n")

        if self.label_column not in df.columns:
            raise KeyError(f"Column '{self.label_column}' not found in dataset. Available columns: {list(df.columns)}")

        df[self.label_column] = df[self.label_column].apply(lambda x: 0 if x == 'BENIGN' else 1)
        logger.info(f"Data after cleaning: {df.shape} \n")
        logger.info(f"Label distribution:\n{df[self.label_column].value_counts()} \n")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        sns.countplot(x=self.label_column, data=df_raw, ax=ax1)
        ax1.set_title("Label Distribution (Raw Data)")
        sns.countplot(x=self.label_column, data=df, ax=ax2)
        ax2.set_title("Label Distribution (Cleaned Data)")
        save_plot(fig, "label_distribution.png")

        return df

    def preprocess(self, df):
        df = self.clean_data(df)
        return df
