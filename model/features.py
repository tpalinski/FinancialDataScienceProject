import numpy as np
import pandas as pd
import ta
from ta.volatility import BollingerBands
from pandas.core.window import rolling

def extract_averages(X: np.ndarray, windows = [3, 5, 10, 20]):
    """
    Extract different rolling avgs for signal
    """
    res = np.empty((len(windows), X.shape[0]))
    for idx, window in enumerate(windows):
        partial = pd.Series(X).rolling(window=window).mean().to_numpy()
        res[idx] = partial
    res = np.transpose(res, (1, 0))
    return res

def extract_bollinger_bands(X: pd.Series, window=20, window_dev=2):
    bb = BollingerBands(close=X, window=window, window_dev=window_dev)
    res = np.column_stack(( bb.bollinger_hband().to_numpy(), bb.bollinger_lband().to_numpy()))
    return res

def extract_macd(X: pd.Series):
    macd = ta.trend.MACD(close=X)
    return np.column_stack((macd.macd().to_numpy(), macd.macd_signal().to_numpy()))

def extract_acc_dec(df: pd.DataFrame):
    ao = ta.momentum.AwesomeOscillatorIndicator(
        high=df['high'],
        low=df['low'],
        window1=5,
        window2=34
    )
    aom = ao.awesome_oscillator()

    # Step 2: Compute AC = AO - SMA(5) of AO
    ac = aom - aom.rolling(window=5).mean()
    return ac.to_numpy().reshape(-1, 1)

def extract_donchian_channel(df: pd.DataFrame):
    dc = ta.volatility.DonchianChannel(df["high"], df["low"], df["close"])
    return np.column_stack((dc.donchian_channel_hband().to_numpy(), dc.donchian_channel_mband().to_numpy(), dc.donchian_channel_lband().to_numpy()))

