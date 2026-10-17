import os
import matplotlib.pyplot as plt
import seaborn as sns

DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "week3_plots")

def ensure_directories():
    os.makedirs(PLOTS_DIR, exist_ok=True)

def apply_statistical_style():
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["font.family"] = "sans-serif"