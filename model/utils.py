import numpy as np
import yfinance as yf

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

def load_data(indexes: list):
    res = {}
    for index in indexes:
        data = yf.download([index], interval="1d", period="10y")
        X = data["Close"].to_numpy().ravel()
        y = __label_markers(X)
        data["Label"] = y
        data.dropna(inplace=True)
        data.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in data.columns]
        data.rename(columns={
            f'Open_{index}': 'open',
            f'High_{index}': 'high',
            f'Low_{index}': 'low',
            f'Close_{index}': 'close',
            f'Volume_{index}': 'volume',
            'Label_': 'label'
        }, inplace=True)
        res[index] = data
    return res


