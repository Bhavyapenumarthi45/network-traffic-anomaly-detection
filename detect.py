"""Detect anomalous network flows with an Isolation Forest.

Reads data/flows.csv (see generate_data.py), engineers flow features,
trains an unsupervised Isolation Forest, and evaluates it against the
ground-truth labels. Saves a scatter plot of flagged flows.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "flows.csv"
PLOT = ROOT / "data" / "anomalies.png"

FEATURES = ["duration_s", "bytes", "packets", "dst_port"]


def build_features(df):
    X = df[FEATURES].copy()
    X["bytes_per_packet"] = df["bytes"] / df["packets"].clip(lower=1)
    X["is_tcp"] = (df["protocol"] == "tcp").astype(int)
    # Log-scale heavy-tailed fields so one huge transfer doesn't dominate
    for col in ["duration_s", "bytes", "packets", "bytes_per_packet"]:
        X[col] = np.log1p(X[col])
    return X


def main():
    df = pd.read_csv(DATA)
    X = build_features(df)
    Xs = StandardScaler().fit_transform(X)

    model = IsolationForest(contamination=0.05, random_state=42)
    # IsolationForest: -1 = anomaly, 1 = normal -> map to 1/0 labels
    pred = (model.fit_predict(Xs) == -1).astype(int)

    y = df["label"].to_numpy()
    print("Confusion matrix [tn fp / fn tp]:")
    print(confusion_matrix(y, pred))
    print(f"Precision: {precision_score(y, pred):.3f}")
    print(f"Recall:    {recall_score(y, pred):.3f}")
    print(f"F1:        {f1_score(y, pred):.3f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["duration_s"], df["bytes"], c="lightgray", s=6, label="normal")
    flagged = df[pred == 1]
    ax.scatter(flagged["duration_s"], flagged["bytes"], c="red", s=14, label="flagged")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Flow duration (s)"); ax.set_ylabel("Bytes")
    ax.set_title("Isolation Forest: flagged anomalous flows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT, dpi=120)
    print(f"Plot saved to {PLOT}")


if __name__ == "__main__":
    main()
