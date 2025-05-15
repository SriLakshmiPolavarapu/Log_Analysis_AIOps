import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

def load_and_parse(path):
    with open(path, "r") as f:
        lines = f.readlines()
    records = []
    for entry in lines:
        parts = entry.strip().split(maxsplit=3)
        if len(parts) != 4:
            continue
        timestamp = f"{parts[0]} {parts[1]}"
        severity  = parts[2]
        text      = parts[3]
        records.append((timestamp, severity, text))
    return pd.DataFrame(records, columns=["when", "severity", "text"])

def engineer_features(df):
    df["when"] = pd.to_datetime(df["when"])
    severity_map = {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
    df["sev_score"] = df["severity"].map(severity_map)
    df["text_length"] = df["text"].str.len()
    return df

def detect_anomalies_lof(df, n_neighbors=20, contamination=0.1):
    features = df[["sev_score", "text_length"]]
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    df["flag"] = lof.fit_predict(features)
    df["label"] = df["flag"].map({1: "Normal", -1: "Anomaly"})
    return df

if __name__ == "__main__":
    logfile = "system_logs.txt"
    df = load_and_parse(logfile)
    df = engineer_features(df)
    df = detect_anomalies_lof(df, n_neighbors=15, contamination=0.1)

    anomalies = df[df["label"] == "Anomaly"]
    print("\nDetected anomalies:\n", anomalies)
