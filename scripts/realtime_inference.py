import time
import joblib
import numpy as np
import pandas as pd


MODEL_PATH = "models/fraud_model.joblib"

FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount"
]


def generate_transaction():
    """Generate one dummy transaction with the correct feature structure."""
    transaction = np.random.normal(0, 1, 30)

    transaction[0] = np.random.uniform(0, 172800)  # Time
    transaction[-1] = np.random.uniform(1, 1000)  # Amount

    return pd.DataFrame(
        [transaction],
        columns=FEATURE_COLUMNS
    )


def main():
    model = joblib.load(MODEL_PATH)

    print("Real-time fraud detection started...\n")

    for i in range(10):
        transaction = generate_transaction()

        fraud_probability = model.predict_proba(transaction)[0][1]

        print(
            f"Transaction {i + 1}: "
            f"Fraud probability = {fraud_probability:.2%}"
        )

        if fraud_probability > 0.95:
            print("🚨 FRAUD ALERT: Probability exceeds 95%!")

        time.sleep(1)


if __name__ == "__main__":
    main()