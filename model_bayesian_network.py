import pandas as pd
import numpy as np
import os
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer
from tqdm import tqdm  

if not os.path.exists("Data"):
    os.makedirs("Data")

def discretize_data(data, n_bins=4):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    data[["pressure", "flow", "speed"]] = discretizer.fit_transform(data[["pressure", "flow", "speed"]]).astype(int)
    return data

def train_bayesian_network(data):
    model = BayesianNetwork([
        ("pressure", "flow"),
        ("flow", "speed"),
        ("speed", "failure"),
        ("pressure", "failure")
    ])
    
    model.fit(data, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=50)
    inference = VariableElimination(model)
    return model, inference

def predict_bayesian(data, inference):
    predictions = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Predicting failures"):
        query_result = inference.map_query(variables=["failure"], evidence={
            "pressure": int(row["pressure"]),
            "flow": int(row["flow"]),
            "speed": int(row["speed"])
        }, show_progress=False)
        
        predictions.append({"cycle": row["cycle"], "predicted_failure": query_result["failure"]})

    df = pd.DataFrame(predictions)
    df.to_csv("Data/bayesian_results.csv", index=False)
    print("Bayesian Network results saved.")

data = pd.read_csv("Data/actuator_data.csv")
data = discretize_data(data)
model, inference = train_bayesian_network(data)
predict_bayesian(data, inference)
