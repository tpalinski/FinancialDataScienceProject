import numpy as np
import pandas as pd
import ta
from ta.volatility import BollingerBands
from pandas.core.window import rolling
from sklearn.linear_model import LinearRegression

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
    bb_percent = bb.bollinger_pband()
    rolling_std = X.rolling(20).std()
    z_score = (X - bb.bollinger_mavg()) / rolling_std
    w_band = bb.bollinger_wband()
    res = np.column_stack((bb_percent.to_numpy(), z_score.to_numpy(), w_band.to_numpy()))
    return res

def extract_macd(X: pd.Series):
    macd = ta.trend.MACD(close=X)
    return macd.macd().to_numpy()

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

def extract_log_returns(df: pd.Series, period = 1):
    return np.log(df / df.shift(period))

def extract_vol_of_vol(log_rets: np.ndarray):
    vol_20 = log_rets.rolling(20).std()
    v_20 = vol_20.rolling(20).std()
    vol_of_vol_z = (
        v_20 -
        v_20.rolling(60).mean()
    ) / v_20.rolling(60).std()
    return vol_of_vol_z

def extract_normalized_momentum(X: pd.Series):
    roc_10 = X.pct_change(10)
    mom_z_50 = (
        roc_10 -
        roc_10.rolling(50).mean()
    ) / roc_10.rolling(50).std()
    return mom_z_50

def extract_lr(close: pd.Series, window: int = 20):
    close = np.array(close)
    n = len(close)
    slopes = np.full(n, np.nan)
    r2s    = np.full(n, np.nan)

    x = np.arange(window).reshape(-1, 1)

    for i in range(window - 1, n):
        y = close[i - window + 1:i + 1].reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        slopes[i] = model.coef_[0, 0]
        r2s[i]    = model.score(x, y)

    # Stack slope and r2 as features
    return np.stack([slopes, r2s], axis=1)
