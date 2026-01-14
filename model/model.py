import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from model.features import extract_acc_dec, extract_averages, extract_bollinger_bands, extract_donchian_channel, extract_macd, extract_log_returns

def train_basic(data: pd.DataFrame, val_split = 0.1):
    prices = data["close"].to_numpy()
    y = data["label"].to_numpy()+1
    bands = extract_bollinger_bands(data["close"])
    bands_close = extract_bollinger_bands(data["close"], window=7)
    macd = extract_macd(data["close"])
    log_rets = extract_log_returns(data["close"], period=20)
    log_rets_5 = extract_log_returns(data["close"], period=5)
    X = np.column_stack((log_rets, log_rets_5, bands, bands_close, macd))
    val_amount = int(len(X) * val_split)
    X_train = X[:-val_amount]
    X_test = X[-val_amount:]
    y_train = y[:-val_amount]
    y_test = y[-val_amount:]
    train_data = lgb.Dataset(X_train, y_train)
    test_data = lgb.Dataset(X_test, y_test)
    params = {
            'objective': 'multiclass',
            'num_class': 3,
            'max_depth': 5,
            'max_leaves': 11,
            'verbosity': 0,
            'min_data_in_leaf': 70,
            'random_state': 67,
            'learning_rate': 0.015,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'lambda_l1': 0.2,
            'lambda_l2': 0.2,
            'bagging_freq': 1,
            }
    model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100)

    y_pred_prob = model.predict(X_train)
    y_pred = y_pred_prob.argmax(axis=1)
    target_names=['sell', 'hold', 'buy']
    labels = [0, 1, 2]
    print(classification_report(y_train, y_pred, target_names=target_names, labels=labels, zero_division=0))
    print(confusion_matrix(y_train, y_pred, labels=labels))
    print(f"Train accuracy: {accuracy_score(y_train, y_pred)}")

    y_test_pred_prob = model.predict(X_test)
    y_pred_test = y_test_pred_prob.argmax(axis=1)
    print(classification_report(y_test, y_pred_test, target_names=target_names, labels=labels, zero_division=0))
    print(confusion_matrix(y_test, y_pred_test, labels=labels))
    print(f"Validation accuracy: {accuracy_score(y_test, y_pred_test)}")
    return model

    
