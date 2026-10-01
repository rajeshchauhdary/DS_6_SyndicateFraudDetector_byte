import time
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "models/fraud_model.joblib"
DATA_PATH = "data/creditcard.csv"

FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount"
]

FRAUD_INJECTION_RATE = 0.25  # how often the stream includes a real fraud-like transaction


def load_fraud_samples():
    """Load real fraudulent transactions to seed the simulated stream, so the
    95% alert path actually gets exercised during a demo run (pure random
    noise almost never resembles a real fraud pattern)."""
    try:
        df = pd.read_csv(DATA_PATH)
        return df[df["Class"] == 1][FEATURE_COLUMNS].reset_index(drop=True)
    except FileNotFoundError:
        print(f"Warning: {DATA_PATH} not found, stream will be fully synthetic.\n")
        return None


def generate_transaction(fraud_samples):
    """Generate one dummy transaction. Most of the time it's random noise
    shaped like a genuine transaction; sometimes it's a real fraud example
    with light noise added, to simulate fraud actually arriving in the stream."""
    use_real_fraud = fraud_samples is not None and np.random.random() < FRAUD_INJECTION_RATE

    if use_real_fraud:
        row = fraud_samples.sample(1).iloc[0].copy()
        noise = np.random.normal(0, 0.05, len(FEATURE_COLUMNS) - 2)
        row[[c for c in FEATURE_COLUMNS if c not in ("Time", "Amount")]] += noise
        transaction = row.values.astype(float)
    else:
        transaction = np.random.normal(0, 1, 30)
        transaction[0] = np.random.uniform(0, 172800)  # Time
        transaction[-1] = np.random.uniform(1, 1000)   # Amount

    return pd.DataFrame([transaction], columns=FEATURE_COLUMNS), use_real_fraud


def main():
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    scaler = artifact["scaler"]
    fraud_samples = load_fraud_samples()

    print("Real-time fraud detection started...\n")

    for i in range(20):
        transaction, is_injected_fraud = generate_transaction(fraud_samples)

        transaction[["Time", "Amount"]] = scaler.transform(
            transaction[["Time", "Amount"]]
        )

        fraud_probability = model.predict_proba(transaction)[0][1]

        tag = " [injected fraud sample]" if is_injected_fraud else ""
        print(
            f"Transaction {i + 1}: "
            f"Fraud probability = {fraud_probability:.2%}{tag}"
        )

        if fraud_probability > 0.95:
            print("*** FRAUD ALERT: Probability exceeds 95%! ***")

        time.sleep(1)


if __name__ == "__main__":
    main()
