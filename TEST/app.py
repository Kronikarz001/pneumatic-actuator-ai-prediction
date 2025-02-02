# STEP 1: GENERATE SIMULATION DATA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Generate balanced simulation data
def generate_balanced_simulation_data(num_cycles=1000, failure_rate=0.05):
    np.random.seed(42)
    data = {
        "cycle": np.arange(1, num_cycles + 1),
        "pressure": np.random.normal(5, 0.2, num_cycles),
        "flow": np.random.normal(50, 5, num_cycles),
        "position": np.random.uniform(0, 100, num_cycles),
        "temperature": np.random.normal(25, 2, num_cycles),
        "failure": np.random.choice([0, 1], num_cycles, p=[1 - failure_rate, failure_rate])
    }
    return pd.DataFrame(data)

simulation_data = generate_balanced_simulation_data()

# STEP 2: IMPLEMENT METHODS

# PID Controller
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, actual_position):
        error = setpoint - actual_position
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def run_pid_simulation(data):
    pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
    pid_results = []
    for _, row in data.iterrows():
        setpoint = 50
        control_signal = pid.compute(setpoint, row["position"])
        predicted_failure = 1 if abs(control_signal) > 3 else 0  # Threshold
        pid_results.append({
            "cycle": row["cycle"],
            "actual_failure": row["failure"],
            "predicted_failure": predicted_failure
        })
    pd.DataFrame(pid_results).to_csv("pid_results.csv", index=False)

# LSTM (simplified for rule-based simulation)
def run_lstm_simulation(data):
    lstm_results = []
    for _, row in data.iterrows():
        predicted_failure = 1 if row["pressure"] < 4.9 or row["flow"] > 52 else 0
        lstm_results.append({
            "cycle": row["cycle"],
            "actual_failure": row["failure"],
            "predicted_failure": predicted_failure
        })
    pd.DataFrame(lstm_results).to_csv("lstm_results.csv", index=False)

# Bayesian Network (simplified implementation)
def run_bayesian_network_simulation(data):
    bn_results = []
    for _, row in data.iterrows():
        # Simulate probabilistic dependencies
        if row["pressure"] < 4.8 and row["flow"] > 53:
            predicted_failure = 1
        else:
            predicted_failure = 0
        bn_results.append({
            "cycle": row["cycle"],
            "actual_failure": row["failure"],
            "predicted_failure": predicted_failure
        })
    pd.DataFrame(bn_results).to_csv("bayesian_network_results.csv", index=False)

# Monte Carlo Simulation
def run_monte_carlo_simulation(data):
    mc_results = []
    for _, row in data.iterrows():
        failures = 0
        for _ in range(1000):  # Simulate 1000 iterations per cycle
            pressure = np.random.normal(row["pressure"], 0.15)
            flow = np.random.normal(row["flow"], 4)
            if pressure < 4.85 or flow > 53:  # Thresholds for failure
                failures += 1
        predicted_failure = 1 if failures / 1000 > 0.08 else 0  # 8% threshold
        mc_results.append({
            "cycle": row["cycle"],
            "actual_failure": row["failure"],
            "predicted_failure": predicted_failure
        })
    pd.DataFrame(mc_results).to_csv("monte_carlo_results.csv", index=False)

# Run all simulations
run_pid_simulation(simulation_data)
run_lstm_simulation(simulation_data)
run_bayesian_network_simulation(simulation_data)
run_monte_carlo_simulation(simulation_data)

# STEP 3: ANALYSIS PROGRAM
def analyze_results():
    methods = ["pid_results.csv", "lstm_results.csv", "bayesian_network_results.csv", "monte_carlo_results.csv"]
    method_names = ["PID", "LSTM", "Bayesian Network", "Monte Carlo"]
    stats = []

    for method, name in zip(methods, method_names):
        results = pd.read_csv(method)
        y_true = results["actual_failure"]
        y_pred = results["predicted_failure"]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        stats.append({
            "Method": name,
            "True Positives": cm[1, 1],
            "False Positives": cm[0, 1],
            "True Negatives": cm[0, 0],
            "False Negatives": cm[1, 0],
            "Precision": report["1"]["precision"],
            "Recall": report["1"]["recall"],
            "F1-Score": report["1"]["f1-score"],
            "Accuracy": accuracy
        })
        # Plot Confusion Matrix
        plt.figure(figsize=(6, 4))
        plt.matshow(cm, cmap="Blues", fignum=1)
        plt.title(f"Confusion Matrix for {name}")
        plt.colorbar()
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.show()

    # Save and display comparison statistics
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv("comparison_statistics.csv", index=False)
    print("Comparison statistics saved to 'comparison_statistics.csv'")
    import ace_tools as tools; tools.display_dataframe_to_user(name="Comparison Statistics", dataframe=stats_df)

# Run the analysis
analyze_results()
