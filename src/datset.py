import pandas as pd
import os
from multiprocessing import Pool
from src.utils import setup_logging

logger = setup_logging()

def load_file(file_path):
    logger.info(f"Loading {file_path}")
    return pd.read_csv(file_path)

def load_data(data_dir="data/"):
    files_names = [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
    ]

    file_paths = [os.path.join(data_dir, fname) for fname in files_names]

    with Pool() as pool:
        df_list = pool.map(load_file, file_paths)

    df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Data loaded: {df.shape} \n")
    logger.info(f"Data header: {df.head(5)} \n")
    return df
