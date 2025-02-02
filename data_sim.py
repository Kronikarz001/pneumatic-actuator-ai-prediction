import numpy as np
import pandas as pd

def generate_actuator_data(num_samples=1000, warning_threshold=0.03, failure_threshold=0.1):
    """
    Generuje dane dla siłowników pneumatycznych. Oznacza ostrzeżenia i awarie.
    """
    np.random.seed(42)
    data = []
    for i in range(num_samples):
        if np.random.rand() > failure_threshold:
            pressure = np.random.normal(5, 0.2)
            flow = np.random.normal(50, 5)
            speed = np.random.uniform(0.8, 1.2)
            failure = 0
        else:
            pressure = np.random.choice([np.random.normal(4.5, 0.3), np.random.normal(5.5, 0.3)])
            flow = np.random.choice([np.random.normal(45, 5), np.random.normal(55, 5)])
            speed = np.random.uniform(0.5, 0.7)
            failure = 1

        warning = 1 if not failure and (abs(pressure - 5) > warning_threshold or abs(flow - 50) > 2) else 0

        data.append({
            "cycle": i + 1,
            "pressure": pressure,
            "flow": flow,
            "speed": speed,
            "warning": warning,
            "failure": failure
        })
    
    return pd.DataFrame(data)

# Generowanie danych
actuator_data = generate_actuator_data()
actuator_data.to_csv("actuator_data.csv", index=False)
print("Dane siłownika zapisane w 'actuator_data.csv'.")
