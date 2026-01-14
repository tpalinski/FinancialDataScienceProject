import numpy as np
import pandas as pd
import backtrader as bt
import math

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

def simulate_model(df: pd.DataFrame, model, title="ETH, test set"):
    class LabelData(bt.feeds.PandasData):
        lines = ('label',)
        params = (
            ('close', 'close'),
            ('open', 'close'),
            ('label', 'label'),
        )

    data = LabelData(dataname=df)

    class LogReturn(bt.Indicator):
        lines = ('logret',)
        params = dict(period=1)
        plotinfo = dict(subplot=True, plot=True)

        def __init__(self):
            pass  # no calculations here, do in next()

        def next(self):
            if len(self.data) <= self.p.period:
                self.lines.logret[0] = 0.0
            else:
                prev_close = self.data.close[-self.p.period]
                if prev_close != 0:
                    self.lines.logret[0] = np.log(self.data.close[0] / prev_close)
                else:
                    self.lines.logret[0] = 0.0

    class ModelStrategy(bt.Strategy):
        plotinfo = dict(plotname=title)

        def __init__(self):
            self.model = model

            self.macd = bt.ind.MACD(self.data.close)
            self.bb = bt.ind.BollingerBands(self.data.close, period=20, devfactor=2, plot=False)
            self.bb_close = bt.ind.BollingerBands(self.data.close, period=7, devfactor=2, plot=False)
            self.logret = LogReturn(self.data, period=20, plot=False)
            self.logret5 = LogReturn(self.data, period=5, plot=False)

        def next(self):
            if len(self) < 30:
                return
            features = np.array([
                self.logret[0],
                self.logret5[0],
                self.bb.lines.top[0],
                self.bb.lines.bot[0],
                self.bb_close.lines.top[0],
                self.bb_close.lines.bot[0],
                self.macd.macd[0],
                self.macd.signal[0],
            ]).reshape(1, -1)

            probs = self.model.predict(features)[0] 
            prob_buy = probs[2]
            prob_sell = probs[0]

            buy_threshold = 0.4
            sell_threshold = 0.5

            if prob_buy > buy_threshold and not self.position:
                size = int(self.broker.getcash() / self.data.close[0])
                if size > 0:
                    self.buy(size=size)
            elif prob_sell > sell_threshold and self.position:
                self.close()
        
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(ModelStrategy)
    cerebro.broker.set_cash(10000)
    cerebro.run()
    cerebro.plot()
