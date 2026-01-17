import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from model.features import extract_acc_dec, extract_averages, extract_bollinger_bands, extract_donchian_channel, extract_macd, extract_log_returns, extract_normalized_momentum, extract_vol_of_vol, extract_lr

def extract_features(data: pd.DataFrame) -> pd.DataFrame:
    log_rets_1 = extract_log_returns(data["close"], period=1)
    bands_20 = extract_bollinger_bands(data["close"])
    macd = extract_macd(data["close"])
    log_rets_20 = extract_log_returns(data["close"], period=20)
    norm_mom_50 = extract_normalized_momentum(data["close"])
    vol_of_vol = extract_vol_of_vol(log_rets_1)
    lr = extract_lr(data["close"])
    lr_short = extract_lr(data["close"], window=7)
    lr_long = extract_lr(data["close"], window=50)
    X = pd.DataFrame(np.column_stack((log_rets_1, log_rets_20, bands_20, macd, norm_mom_50, vol_of_vol, lr, lr_short, lr_long)), columns = [
        'Log Returns (1)',
        'Log Returns (20)',
        'B% (20)',
        'Bollinger Band Z Score (20)',
        'Bollinger Band Width',
        'MACD',
        'Normalized Momentum (50)',
        'Volatility of Volatility',
        'Linear Regression Slope (20)',
        'Linear Regression R2 (20)',
        'Linear Regression Slope (7)',
        'Linear Regression R2 (7)',
        'Linear Regression Slope (50)',
        'Linear Regression R2 (50)'
    ])
    return X

def train_basic(data: pd.DataFrame, val_days=100, epochs=100):
    X = extract_features(data)
    print(X.corr().abs().sort_values(by="Log Returns (1)", ascending=False))
    y = data["label"].to_numpy()+1
    X_train = X[:-val_days]
    X_test = X[-val_days:]
    y_train = y[:-val_days]
    y_test = y[-val_days:]
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    weights_dict = dict(zip(classes, class_weights))
    train_data = lgb.Dataset(X_train, y_train, weight=[weights_dict[i] for i in y_train])
    test_data = lgb.Dataset(X_test, y_test)
    params = {
            'objective': 'multiclass',
            'num_class': 3,
            'max_depth': 5,
            'max_leaves': 15,
            'verbosity': -1,
            'min_data_in_leaf': 70,
            'random_state': 67,
            'learning_rate': 0.015,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'lambda_l1': 0.15,
            'lambda_l2': 0.15,
            'bagging_freq': 1,
            }
    model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=epochs)

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

    
def fine_tune_model(original_model, data: pd.DataFrame, val_days = 100, epochs=100):
    X = extract_features(data)
    y = data["label"].to_numpy()+1
    X_train = X[:-val_days]
    X_test = X[-val_days:]
    y_train = y[:-val_days]
    y_test = y[-val_days:]
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    weights_dict = dict(zip(classes, class_weights))
    train_data = lgb.Dataset(X_train, y_train, weight=[weights_dict[i] for i in y_train])
    test_data = lgb.Dataset(X_test, y_test)
    model = lgb.train(original_model.params, train_data, valid_sets=[test_data], num_boost_round=epochs, keep_training_booster=True, init_model=original_model)

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
