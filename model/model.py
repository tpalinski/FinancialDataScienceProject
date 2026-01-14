import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from model.features import extract_acc_dec, extract_averages, extract_bollinger_bands, extract_donchian_channel, extract_macd

def train_basic(data: pd.DataFrame):
    prices = data["close"].to_numpy()
    volume = data["volume"].to_numpy()
    y = data["label"].to_numpy()+1
    bands = extract_bollinger_bands(data["close"])
    bands_close = extract_bollinger_bands(data["close"], window=7)
    macd = extract_macd(data["close"])
    ac = extract_acc_dec(data)
    X = np.column_stack((prices, volume.reshape(-1, 1), bands, bands_close, macd, ac))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_data = lgb.Dataset(X_train, y_train)
    test_data = lgb.Dataset(X_test, y_test)
    params = {
            'objective': 'multiclass',
            'num_class': 3,
            'max_depth': 6,
            'max_leaves': 32,
            'verbosity': 1,
            }
    model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)

    y_pred_prob = model.predict(X_train)
    y_pred = y_pred_prob.argmax(axis=1)
    target_names=['sell', 'hold', 'buy']
    print(classification_report(y_train, y_pred, target_names=target_names))
    print(confusion_matrix(y_train, y_pred))

    y_test_pred_prob = model.predict(X_test)
    y_pred_test = y_test_pred_prob.argmax(axis=1)
    print(classification_report(y_test, y_pred_test, target_names=target_names))
    print(confusion_matrix(y_test, y_pred_test))
    return model

    
