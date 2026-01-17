import pandas as pd
import matplotlib.pyplot as plt
import shap

from .features import extract_log_returns, extract_macd, extract_log_returns, extract_bollinger_bands
from .model import extract_features

def explain_model(model, data: pd.DataFrame, suffix = ""):
    X = extract_features(data)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    shap.summary_plot(shap_vals[:, :, 0], X, show=False)
    plt.title(f"Shap Values for Sell: {suffix}")
    plt.show()
    shap.summary_plot(shap_vals[:, :, 1], X, show=False)
    plt.title(f"Shap Values for Hold: {suffix}")
    plt.show()
    shap.summary_plot(shap_vals[:, :, 2], X, show=False)
    plt.title(f"Shap Values for Buy: {suffix}")
    plt.show()

    return shap_vals
