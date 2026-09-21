import joblib
from sklearn.ensemble import RandomForestClassifier

from preprocessing import prepare_data


def train_model():
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        "data/creditcard.csv"
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler
        },
        "models/fraud_model.joblib"
    )

    return model, X_test, y_test, scaler


if __name__ == "__main__":
    model, X_test, y_test, scaler = train_model()

    print("Model trained successfully.")
    print("Test samples:", len(X_test))