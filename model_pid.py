import pandas as pd
import numpy as np

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

def run_pid(data):
    pid = PIDController(kp=1.2, ki=0.02, kd=0.01)  # Zmniejszone KI, KD, aby uniknąć skoków
    results = []
    for _, row in data.iterrows():
        pressure_control = pid.compute(5, row["pressure"])
        flow_control = pid.compute(50, row["flow"])
        speed_control = pid.compute(1.0, row["speed"])

        optimized_warning = 1 if abs(pressure_control) > 0.2 or abs(flow_control) > 3 else 0
        optimized_failure = 1 if abs(pressure_control) > 0.5 or abs(flow_control) > 8 else 0

        results.append({
            "cycle": row["cycle"],
            "optimized_warning": optimized_warning,
            "optimized_failure": optimized_failure
        })

    return pd.DataFrame(results)

data = pd.read_csv("Data/actuator_data.csv")
pid_results = run_pid(data)
pid_results.to_csv("Data/pid_results.csv", index=False)
print("PID model results saved.")
