import pandas as pd
import numpy as np

def run_monte_carlo(data):
    results = []
    for _, row in data.iterrows():
        failures = sum(np.random.normal(row["pressure"], 0.12) < 4.86 or np.random.normal(row["flow"], 3.5) > 52.5
                       for _ in range(1000))
        predicted_failure = 1 if failures / 1000 > 0.07 else 0
        results.append({
            "cycle": row["cycle"],
            "predicted_failure": predicted_failure
        })
    
    df = pd.DataFrame(results)
    df.to_csv("Data/monte_carlo_results.csv", index=False)
    print("Wyniki Monte Carlo zapisane w monte_carlo_results.csv.")

if __name__ == "__main__":
    data = pd.read_csv("Data/actuator_data.csv")
    run_monte_carlo(data)
