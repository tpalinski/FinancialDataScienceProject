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

def __label_markers_volatility(prices: pd.Series, window=7, tp=0.03, tl=0.02):
    """
    Vectorized triple barrier labeling with optional adaptive barriers
    """
    n = len(prices)
    labels = np.zeros(n)
    vol = pd.Series(prices).pct_change().rolling(window).std()

    for i in range(n - window):
        entry = prices[i]
        future = prices[i+1:i+1+window]
        profit = entry * (1 + tp * vol[i])
        stop   = entry * (1 - tl * vol[i])

        # check first barrier hit
        above = np.where(future >= profit)[0]
        below = np.where(future <= stop)[0]

        if len(above) > 0 and len(below) > 0:
            if above[0] < below[0]:
                labels[i] = 1
            else:
                labels[i] = -1
        elif len(above) > 0:
            labels[i] = 1
        elif len(below) > 0:
            labels[i] = -1
        else:
            labels[i] = 0

    labels[-window:] = np.nan
    return labels

def load_data(path="data/dataset.csv"):
    df = pd.read_csv(path, sep=";", parse_dates=["Date"], index_col="Date")
    series_dict = {}
    for col in df.columns:
        price_series = df[col].shift(1).dropna()
        labels = __label_markers_volatility(price_series.values)
        asset_df = pd.DataFrame({
            "close": price_series,
            "label": labels
        }, index=price_series.index)
        asset_df.dropna(inplace=True)
        series_dict[col] = asset_df
    return series_dict

