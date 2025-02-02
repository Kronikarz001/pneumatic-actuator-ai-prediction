import pandas as pd
import numpy as np

def run_monte_carlo(data, simulations=1000):
    np.random.seed(42)
    results = []
    
    for _, row in data.iterrows():
        failure_count = 0
        for _ in range(simulations):
            simulated_pressure = row["pressure"] + np.random.normal(0, 0.3)
            simulated_flow = row["flow"] + np.random.normal(0, 2.0)
            simulated_speed = row["speed"] + np.random.normal(0, 0.05)
            
            if simulated_pressure > 5.5 or simulated_flow > 55:
                failure_count += 1

        predicted_failure = 1 if (failure_count / simulations) > 0.15 else 0  # Wyższy próg awarii
        results.append({"cycle": row["cycle"], "predicted_failure": predicted_failure})

    return pd.DataFrame(results)

data = pd.read_csv("Data/actuator_data.csv")
monte_carlo_results = run_monte_carlo(data)
monte_carlo_results.to_csv("Data/monte_carlo_results.csv", index=False)
print("Monte Carlo results saved.")
