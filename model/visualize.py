import numpy as np
import pandas as pd
import backtrader as bt

def visualize_dataset(df: pd.DataFrame):
    class LabelData(bt.feeds.PandasData):
        lines = ('label',)
        params = (('label', -1),)

    data = LabelData(dataname=df)

    class LabelStrategy(bt.Strategy):
        trade_log = []

        def __init__(self) -> None:
            super().__init__()
            self.macd = bt.ind.MACD(self.data.close)
        def next(self):
            label = self.data.label[0]
            if not self.position and label == 1.0:
                cash = self.broker.get_cash()
                price = self.data.close[0]
                size = int(cash / price)
                if size > 0:
                    self.buy(size=size)
            elif self.position and label == -1.0:
                self.close()
        
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(LabelStrategy)
    cerebro.broker.set_cash(1000)
    cerebro.run()
    cerebro.plot()

def simulate_model(df: pd.DataFrame, model):
    class LabelData(bt.feeds.PandasData):
        lines = ('label',)
        params = (('label', -1),)

    data = LabelData(dataname=df)

    class DonchianChannels(bt.Indicator):
        '''
        Params Note:
          - `lookback` (default: -1)
            If `-1`, the bars to consider will start 1 bar in the past and the
            current high/low may break through the channel.
            If `0`, the current prices will be considered for the Donchian
            Channel. This means that the price will **NEVER** break through the
            upper/lower channel bands.
        '''

        alias = ('DCH', 'DonchianChannel',)

        lines = ('dcm', 'dch', 'dcl',)  # dc middle, dc high, dc low
        params = dict(
            period=20,
            lookback=-1,  # consider current bar or not
        )

        plotinfo = dict(subplot=False)  # plot along with data
        plotlines = dict(
            dcm=dict(ls='--'),  # dashed line
            dch=dict(_samecolor=True),  # use same color as prev line (dcm)
            dcl=dict(_samecolor=True),  # use same color as prev line (dch)
        )

    def __init__(self):
        hi, lo = self.data.high, self.data.low
        if self.p.lookback:  # move backwards as needed
            hi, lo = hi(self.p.lookback), lo(self.p.lookback)

        self.l.dch = bt.ind.Highest(hi, period=self.p.period)
        self.l.dcl = bt.ind.Lowest(lo, period=self.p.period)
        self.l.dcm = (self.l.dch + self.l.dcl) / 2.0  # avg of the above

    class ModelStrategy(bt.Strategy):

        def __init__(self):
            self.model = model

            self.macd = bt.ind.MACD(self.data.close)

            self.bb = bt.ind.BollingerBands(self.data.close, period=20, devfactor=2, plot=False)
            self.bb_close = bt.ind.BollingerBands(self.data.close, period=7, devfactor=2, plot=False)
            self.acdec = bt.ind.AccelerationDecelerationOscillator()

        def next(self):
            if len(self) < 30:
                return
            features = np.array([
                self.data.close[0],
                self.data.volume[0],
                self.bb.lines.top[0],
                self.bb.lines.bot[0],
                self.bb_close.lines.top[0],
                self.bb_close.lines.bot[0],
                self.macd.macd[0],
                self.macd.signal[0],
                self.acdec[0],
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
