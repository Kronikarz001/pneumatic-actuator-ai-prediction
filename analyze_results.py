import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curves():
    data = pd.read_csv("Data/ctuator_data.csv")
    lstm_results = pd.read_csv("Data/lstm_results.csv")
    monte_carlo_results = pd.read_csv("Data/monte_carlo_results.csv")

    plt.figure(figsize=(10, 6))

    fpr_lstm, tpr_lstm, _ = roc_curve(data["failure"], lstm_results["predicted_failure"])
    roc_auc_lstm = auc(fpr_lstm, tpr_lstm)
    plt.plot(fpr_lstm, tpr_lstm, label=f"Rule-based LSTM (AUC = {roc_auc_lstm:.2f})")

    fpr_mc, tpr_mc, _ = roc_curve(data["failure"], monte_carlo_results["predicted_failure"])
    roc_auc_mc = auc(fpr_mc, tpr_mc)
    plt.plot(fpr_mc, tpr_mc, label=f"Monte Carlo (AUC = {roc_auc_mc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Guess (AUC = 0.50)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    plot_roc_curves()
