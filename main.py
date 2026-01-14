import numpy as np

from model.features import extract_averages
from model.model import train_basic
from model.utils import load_data
from model.visualize import visualize_dataset, simulate_model

def main():
    data = load_data()
    eth = data["ETH"]
    print(eth)
    visualize_dataset(eth)

if __name__ == '__main__':
    main()
