import numpy as np

from model.features import extract_averages
from model.model import train_basic
from model.utils import load_data
from model.visualize import visualize_dataset, simulate_model

def main():
    print("Downloading datasets")
    ldata = load_data(["GRND", "AAPL", "MSFT", "TSLA", "AMD", "XOM"])
    data = ldata["AAPL"]
    simulate_split = 300
    train_data = data[:-simulate_split]
    test_data = data[-simulate_split:]
    model = train_basic(train_data)
    visualize_dataset(test_data)
    simulate_model(test_data, model)
    simulate_model(data, model)
    simulate_model(ldata["GRND"], model)
    simulate_model(ldata["MSFT"], model)
    simulate_model(ldata["TSLA"], model)
    simulate_model(ldata["AMD"], model)
    simulate_model(ldata["XOM"], model)

if __name__ == '__main__':
    main()
