import pandas as pd

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
    pid = PIDController(kp=1.2, ki=0.05, kd=0.03)
    results = []
    for _, row in data.iterrows():
        pressure_control = pid.compute(5, row["pressure"])
        flow_control = pid.compute(50, row["flow"])
        speed_control = pid.compute(1.0, row["speed"])

        optimized_warning = 1 if abs(pressure_control) > 0.15 or abs(flow_control) > 4 else 0
        optimized_failure = 1 if abs(pressure_control) > 0.45 or abs(flow_control) > 9 else 0

        results.append({
            "cycle": row["cycle"],
            "optimized_warning": optimized_warning,
            "optimized_failure": optimized_failure
        })

    df = pd.DataFrame(results)
    df.to_csv("Data/pid_results.csv", index=False)
    print("Wyniki PID zapisane w pid_results.csv.")

if __name__ == "__main__":
    data = pd.read_csv("Data/actuator_data.csv")
    run_pid(data)
