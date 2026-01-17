from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression

def visualize_dataset(df: pd.DataFrame):
    class LabelData(bt.feeds.PandasData):
        lines = ('label',)
        params = (
            ('close', 'close'),
            ('open', 'close'),
            ('label', 'label'),
        )

    data = LabelData(dataname=df)

    class LabelStrategy(bt.Strategy):
        trade_log = []

        def __init__(self) -> None:
            super().__init__()
            self.macd = bt.ind.MACD(self.data.close)
        def next(self):
            label = self.data.label[0]
            if not self.position and label > 0.5:
                cash = self.broker.get_cash()
                price = self.data.close[0]
                size = cash / price / 2
                if size > 0:
                    self.buy(size=size)
            elif self.position and label < -0.5:
                self.close()
        
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(LabelStrategy)
    cerebro.broker.set_cash(100)
    cerebro.run()
    cerebro.plot(high=False, low=False, volume=False)


class LogReturn(bt.Indicator):
    lines = ('logret',)
    params = dict(period=1)
    plotinfo = dict(subplot=True, plot=True)

    def __init__(self):
        self.addminperiod(self.p.period + 1)

    def next(self):
        # current and previous close
        prev_close = self.data[-self.p.period]
        curr_close = self.data[0]

        if prev_close != 0:
            self.lines.logret[0] = np.log(curr_close / prev_close)
        else:
            self.lines.logret[0] = 0.0


class VolOfVol(bt.Indicator):
    lines = ('volvol',)
    params = dict(period=20)

    def __init__(self):
        self.addminperiod(self.p.period + 1)
        self.ret = LogReturn(self.data, period=1)

    def next(self):
        returns = np.array([self.ret[i] for i in range(-self.p.period + 1, 1)])
        self.lines.volvol[0] = returns.std() if len(returns) > 1 else 0


class NormalizedMomentum(bt.Indicator):
    lines = ('mom',)
    params = dict(window=50)

    def __init__(self):
        self.addminperiod(self.p.window)

    def next(self):
        window_data = np.array([self.data[i] for i in range(-self.p.window + 1, 1)])
        if window_data.std() > 0:
            self.lines.mom[0] = (window_data[-1] - window_data.mean()) / window_data.std()
        else:
            self.lines.mom[0] = 0

class RollingLinearR2(bt.Indicator):
    lines = ('slope', 'r2',)
    params = dict(period=20)

    def __init__(self):
        self.addminperiod(self.p.period)

    def next(self):
        y = np.array([self.data[i] for i in range(-self.p.period + 1, 1)]).reshape(-1, 1)
        x = np.arange(self.p.period).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        self.lines.slope[0] = model.coef_[0, 0]
        self.lines.r2[0] = model.score(x, y)


class BollingerDerived(bt.Indicator):
    lines = ('percent', 'zscore', 'width',)
    
    params = dict(
        period=20,
        devfactor=2
    )

    def __init__(self):
        self.bb = bt.ind.BollingerBands(self.data, period=self.p.period, devfactor=self.p.devfactor)

    def next(self):
        mid = self.bb.mid[0]
        top = self.bb.top[0]
        bot = self.bb.bot[0]
        close = self.data[0]

        if top - bot != 0:
            self.lines.percent[0] = (close - bot) / (top - bot)
        else:
            self.lines.percent[0] = 0

        if top - mid != 0:
            self.lines.zscore[0] = (close - mid) / (top - mid)
        else:
            self.lines.zscore[0] = 0

        if mid != 0:
            self.lines.width[0] = (top - bot) / mid
        else:
            self.lines.width[0] = 0

def simulate_model(df: pd.DataFrame, model, largest_window, buy_prob=0.6, sell_prob=0.7, title="ETH, test set", old_model: None|LGBMClassifier = None):
    class LabelData(bt.feeds.PandasData):
        lines = ('label',)
        params = (
            ('close', 'close'),
            ('open', 'close'),
            ('label', 'label'),
        )

    data = LabelData(dataname=df)

    class BuyAndHold(bt.Strategy):
        """
        Baseline: Buy at the first bar, hold until the end.
        """
        def __init__(self):
            self.bought = False
            self.days = 0

        def next(self):
            if not self.bought and self.days >= largest_window:
                size = int(self.broker.getcash() / self.data.close[0])
                print(f"Buying and holding: {size}")
                self.buy(size=size)
                self.bought = True
            self.days += 1


    class ModelStrategy(bt.Strategy):
        plotinfo = dict(plotname=title)

        def __init__(self, model = model):
            self.model = model
            # Log returns
            self.logret1 = LogReturn(self.data.close, period=1, plot=False)
            self.logret20 = LogReturn(self.data.close, period=20, plot=False)
            # Bollinger
            self.bb_derived = BollingerDerived(self.data.close, period=20, plot=False)
            # MACD
            self.macd = bt.ind.MACD(self.data.close, plot=False)
            # Normalized momentum
            self.norm_mom_50 = NormalizedMomentum(self.data.close, window=50, plot=False)
            # Vol-of-vol
            self.vol_of_vol = VolOfVol(self.data.close, period=20, plot=False)
            # Linear regression features
            self.lr_7 = RollingLinearR2(self.data.close, period=7, plot=False)
            self.lr_20 = RollingLinearR2(self.data.close, period=20, plot=False)
            self.lr_50 = RollingLinearR2(self.data.close, period=50, plot=False)

        def next(self):
            if len(self) < 30:
                return
            features = np.array([
                self.logret1[0],
                self.logret20[0],
                self.bb_derived.percent[0],
                self.bb_derived.zscore[0],
                self.bb_derived.width[0],
                self.macd.macd[0],
                self.norm_mom_50[0],
                self.vol_of_vol[0],
                self.lr_20.slope[0],
                self.lr_20.r2[0],
                self.lr_7.slope[0],
                self.lr_7.r2[0],
                self.lr_50.slope[0],
                self.lr_50.r2[0]
            ]).reshape(1, -1)

            probs = self.model.predict(features)[0] 
            prob_buy = probs[2]
            prob_sell = probs[0]

            buy_threshold = buy_prob
            sell_threshold = sell_prob

            if prob_buy > buy_threshold and not self.position:
                size = int(self.broker.getcash() / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
            elif prob_sell > sell_threshold and self.position:
                self.close()

    class OldModelStrategy(ModelStrategy):
        def __init__(self):
            super().__init__(old_model)

        
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(ModelStrategy)
    cerebro.broker.set_cash(10000)
    cerebro.run()
    cerebro.plot(high=False, low=False, volume=False)

    # Compare equity curves
    cerebro_ml = bt.Cerebro()
    cerebro_ml.adddata(data)
    cerebro_ml.addstrategy(ModelStrategy)
    cerebro_ml.broker.setcash(10000.0)
    cerebro_ml.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    res_ml = cerebro_ml.run()[0]
    ml_returns = res_ml.analyzers.timereturn.get_analysis()

    cerebro_bh = bt.Cerebro()
    cerebro_bh.adddata(data)
    cerebro_bh.addstrategy(BuyAndHold)
    cerebro_bh.broker.setcash(10000.0)
    cerebro_bh.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
    res_bh = cerebro_bh.run()[0]
    bh_returns = res_bh.analyzers.timereturn.get_analysis()
    ml_eq = (1 + pd.Series(ml_returns.values())).cumprod()[largest_window:]
    bh_eq = (1 + pd.Series(bh_returns.values())).cumprod()[largest_window:]

    if old_model is not None:
        cerebro_old = bt.Cerebro()
        cerebro_old.adddata(data)
        cerebro_old.addstrategy(OldModelStrategy)
        cerebro_old.broker.setcash(10000.0)
        cerebro_old.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        res_old = cerebro_old.run()[0]
        old_returns = res_old.analyzers.timereturn.get_analysis()
        old_eq = (1 + pd.Series(old_returns.values())).cumprod()[largest_window:]


    plt.figure(figsize=(12,6))
    plt.plot(ml_eq, label="ML Strategy")
    plt.plot(bh_eq, label="Buy & Hold")
    if old_model is not None:
        plt.plot(old_eq, label="Dedicated Model Strategy")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.title(title)
    plt.legend()
    plt.show()

    print(f"Buy and hold result: {bh_eq[249]}")
    if old_model is not None:
        print(f"Old model result: {old_eq[249]}")
    print(f"Model result: {ml_eq[249]}")
