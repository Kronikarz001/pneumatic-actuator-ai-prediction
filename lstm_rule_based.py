import pandas as pd

def run_rule_based_lstm(data):
    results = []
    for _, row in data.iterrows():
        predicted_failure = 1 if row["pressure"] < 4.85 or row["flow"] > 54 or row["speed"] < 0.72 else 0
        results.append({
            "cycle": row["cycle"],
            "predicted_failure": predicted_failure
        })
    
    df = pd.DataFrame(results)
    df.to_csv("Data/lstm_results.csv", index=False)
    print("Wyniki Rule-based LSTM zapisane w lstm_results.csv.")

if __name__ == "__main__":
    data = pd.read_csv("Data/actuator_data.csv")
    run_rule_based_lstm(data)
