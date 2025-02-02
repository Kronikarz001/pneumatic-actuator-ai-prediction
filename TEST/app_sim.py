# STEP 1: Generate Simulation Data for Pneumatic Actuators

# STEP 1: GENERATE SIMULATION DATA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def generate_actuator_data(num_samples=1000, failure_rate=0.1, warning_threshold=0.03):
    """
    Generate simulation data for pneumatic actuators with parameters for normal and failure conditions.
    """
    np.random.seed(42)
    data = []
    for i in range(num_samples):
        # Normal operation
        if np.random.rand() > failure_rate:
            pressure = np.random.normal(5, 0.2)  # Stable pressure around 5 bar
            flow = np.random.normal(50, 5)  # Stable flow rate
            speed = np.random.uniform(0.8, 1.2)  # Normalized piston speed
            failure = 0
        else:
            # Failure or warning condition
            pressure = np.random.choice([np.random.normal(4.5, 0.3), np.random.normal(5.5, 0.3)])  # Pressure spike or drop
            flow = np.random.choice([np.random.normal(45, 5), np.random.normal(55, 5)])  # Flow spike or drop
            speed = np.random.uniform(0.5, 0.7)  # Slower piston speed indicating an issue
            failure = 1 if (abs(pressure - 5) > 0.5 or abs(flow - 50) > 5 or speed < 0.7) else 0
        
        # Add warning condition if parameters deviate slightly
        warning = 1 if failure == 0 and (abs(pressure - 5) > warning_threshold or abs(flow - 50) > 2) else 0
        
        data.append({
            "cycle": i + 1,
            "pressure": pressure,
            "flow": flow,
            "speed": speed,
            "warning": warning,
            "failure": failure
        })
    
    return pd.DataFrame(data)

# Generate the dataset
actuator_data = generate_actuator_data()
actuator_data.to_csv("actuator_data.csv", index=False)
print("Generated actuator data saved to 'actuator_data.csv'.")

# STEP 2: Define Predictive Models for Actuator Optimization

# PID Optimization
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, actual_value):
        error = setpoint - actual_value
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def optimize_with_pid(data):
    pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
    results = []
    for _, row in data.iterrows():
        pressure_control = pid.compute(5, row["pressure"])  # Setpoint for pressure
        flow_control = pid.compute(50, row["flow"])  # Setpoint for flow
        speed_control = pid.compute(1.0, row["speed"])  # Setpoint for speed (normalized)
        
        # Determine warning or failure based on optimization
        optimized_warning = 1 if abs(pressure_control) > 0.2 or abs(flow_control) > 5 or abs(speed_control) > 0.1 else 0
        optimized_failure = 1 if abs(pressure_control) > 0.5 or abs(flow_control) > 10 or abs(speed_control) > 0.3 else 0
        
        results.append({
            "cycle": row["cycle"],
            "pressure_control": pressure_control,
            "flow_control": flow_control,
            "speed_control": speed_control,
            "optimized_warning": optimized_warning,
            "optimized_failure": optimized_failure
        })
    
    return pd.DataFrame(results)

# Run the PID optimization
pid_optimized_results = optimize_with_pid(actuator_data)
pid_optimized_results.to_csv("pid_optimized_results.csv", index=False)
print("PID optimized results saved to 'pid_optimized_results.csv'.")

# STEP 3: Analysis of Results (Warnings and Failures)

def analyze_actuator_performance(data, optimized_data):
    # Merge original and optimized results for analysis
    combined = pd.merge(data, optimized_data, on="cycle")
    
    # Calculate overall statistics
    stats = {
        "Total Cycles": len(combined),
        "Total Warnings": combined["warning"].sum(),
        "Total Failures": combined["failure"].sum(),
        "Optimized Warnings": combined["optimized_warning"].sum(),
        "Optimized Failures": combined["optimized_failure"].sum(),
    }
    
    # Generate warning and failure reduction percentages
    stats["Warning Reduction (%)"] = 100 * (1 - (stats["Optimized Warnings"] / stats["Total Warnings"]))
    stats["Failure Reduction (%)"] = 100 * (1 - (stats["Optimized Failures"] / stats["Total Failures"]))
    
    # Plot results for warnings and failures
    plt.figure(figsize=(10, 6))
    labels = ["Warnings", "Failures"]
    original_counts = [stats["Total Warnings"], stats["Total Failures"]]
    optimized_counts = [stats["Optimized Warnings"], stats["Optimized Failures"]]
    x = np.arange(len(labels))
    
    plt.bar(x - 0.2, original_counts, width=0.4, label="Original")
    plt.bar(x + 0.2, optimized_counts, width=0.4, label="Optimized")
    plt.xticks(x, labels)
    plt.ylabel("Count")
    plt.title("Comparison of Warnings and Failures (Original vs Optimized)")
    plt.legend()
    plt.show()
    
    return stats

# Analyze results
performance_stats = analyze_actuator_performance(actuator_data, pid_optimized_results)
print("Performance Statistics:", performance_stats)
