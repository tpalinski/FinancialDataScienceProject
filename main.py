import numpy as np

from model.features import extract_averages
from model.model import train_basic
from model.utils import load_data
from model.visualize import visualize_dataset, simulate_model

validation_days = 200

def main():
    data = load_data()
    eth = data["ETH"]
    btc = data["BTC"]
    xrp = data["XRP"]
    ltc = data["LTC"]
    xrm = data["XLM"]
    visualize_dataset(btc)
    train = xrm[:-validation_days]
    test = xrm[-validation_days:]
    model = train_basic(train)
    simulate_model(test, model)

if __name__ == '__main__':
    main()
