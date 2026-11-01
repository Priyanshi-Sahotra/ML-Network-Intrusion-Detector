import pandas as pd
import joblib

def predict_network_traffic(file):

    saved = joblib.load("models/intrusion_model.pkl")

    model = saved["model"]
    features = saved["features"]

    data = pd.read_csv(file)

    missing = [col for col in features if col not in data.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    X = data[features].apply(
        pd.to_numeric, errors="coerce"
    )

    predictions = model.predict(X)

    results = data.copy()

    results["Prediction"] = [
        "ATTACK" if value == 1 else "BENIGN"
        for value in predictions
    ]

    return results

