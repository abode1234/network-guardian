from src.datset import load_data
from src.preprocessor import Preprocessor

def main():

    df = load_data("./dataset/1/")


    preprocessor = Preprocessor(label_column=' Label')
    df_processed = preprocessor.clean_data(df)


    print("Processed Data Head:\n", df_processed.head())

if __name__ == "__main__":
    main()
