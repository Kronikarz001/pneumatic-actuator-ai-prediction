import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os

# Sprawdzenie, czy folder Data/ istnieje
if not os.path.exists("Data"):
    os.makedirs("Data")

# Wczytanie dostępnych wyników
available_files = os.listdir("Data/")

actuator_data = pd.read_csv("Data/actuator_data.csv") if "actuator_data.csv" in available_files else None
pid_results = pd.read_csv("Data/pid_results.csv") if "pid_results.csv" in available_files else None
lstm_results = pd.read_csv("Data/lstm_results.csv") if "lstm_results.csv" in available_files else None
monte_carlo_results = pd.read_csv("Data/monte_carlo_results.csv") if "monte_carlo_results.csv" in available_files else None
bayesian_results = pd.read_csv("Data/bayesian_results.csv") if "bayesian_results.csv" in available_files else None

# Funkcja do analizy modelu
def analyze_model(name, true_values, predictions):
    cm = confusion_matrix(true_values, predictions)
    report = classification_report(true_values, predictions, output_dict=True)
    return {
        "Method": name,
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1-Score": report["1"]["f1-score"],
        "Accuracy": report["accuracy"],
        "Confusion Matrix": cm
    }

# Analiza wszystkich metod, jeśli pliki istnieją
results = []
if pid_results is not None:
    results.append(analyze_model("PID", actuator_data["failure"], pid_results["optimized_failure"]))
if lstm_results is not None:
    results.append(analyze_model("Rule-based LSTM", actuator_data["failure"], lstm_results["predicted_failure"]))
if monte_carlo_results is not None:
    results.append(analyze_model("Monte Carlo", actuator_data["failure"], monte_carlo_results["predicted_failure"]))
if bayesian_results is not None:
    results.append(analyze_model("Bayesian Network", actuator_data["failure"], bayesian_results["predicted_failure"]))

# Jeśli nie ma żadnych wyników, zakończ program
if not results:
    print("Brak wyników do analizy. Upewnij się, że pliki wynikowe istnieją w folderze Data/.")
    exit()

# Tworzenie tabeli wyników
results_df = pd.DataFrame(results).drop(columns=["Confusion Matrix"])
results_df.to_csv("Data/comparison_results.csv", index=False)
print("\nTabela wyników zapisana w Data/comparison_results.csv.")
print(results_df)

# MACIERZE KONFUZJI
fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
if len(results) == 1:
    axes = [axes]

for ax, model in zip(axes, results):
    cm = model["Confusion Matrix"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {model['Method']}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.show()

# WYKRES ROC
plt.figure(figsize=(10, 6))
for model, pred in zip(results, [pid_results, lstm_results, monte_carlo_results, bayesian_results]):
    if pred is not None:
        fpr, tpr, _ = roc_curve(actuator_data["failure"], pred.iloc[:, -1])
        plt.plot(fpr, tpr, label=f"{model['Method']} (AUC = {auc(fpr, tpr):.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Random Guess (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.show()

# WYKRES PORÓWNAWCZY
results_df.set_index("Method")[["Precision", "Recall", "F1-Score", "Accuracy"]].plot(kind="bar", figsize=(10, 6))
plt.title("Performance Comparison Across Models")
plt.ylabel("Wartość metryki")
plt.xticks(rotation=45)
plt.show()
