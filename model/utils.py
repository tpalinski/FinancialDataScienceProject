import numpy as np
import pandas as pd

def __label_markers(X: np.ndarray, window = 7, tp = 0.03, tl = 0.02):
    """
    Labels data using triple barrier labeling

    Returns:
        labels (np.ndarray) - array of labels of shape (X.shape[0] - window)
    """
    labels = np.empty(X.shape[0])
    for i in range(X.shape[0]):
        entry_price = X[i]
        profit_barrier = entry_price * (1 + tp)
        stop_barrier = entry_price * (1 - tl)
        future_prices = X[i+1:i+1+window]

        label = 0
        for price in future_prices:
            if price >= profit_barrier:
                label = 1
                break
            elif price <= stop_barrier:
                label = -1
                break
        labels[i] = label
    labels[-window:] = np.nan
    return labels

def load_data(path="data/dataset.csv"):
    df = pd.read_csv(path, sep=";", parse_dates=["Date"], index_col="Date")
    series_dict = {}
    for col in df.columns:
        price_series = df[col]
        labels = __label_markers(price_series.values)
        asset_df = pd.DataFrame({
            "close": price_series,
            "label": labels
        }, index=price_series.index)
        asset_df.dropna(inplace=True)
        series_dict[col] = asset_df
    return series_dict

