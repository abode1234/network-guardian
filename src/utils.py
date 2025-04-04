import logging
import  matplotlib.pyplot as plt
import seaborn as sns
import os

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("netguardian.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def save_plot(fig, filename, output_dir="plots/"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
