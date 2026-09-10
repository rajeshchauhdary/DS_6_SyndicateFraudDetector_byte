import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def prepare_data(path: str):
    df = pd.read_csv(path)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    scaler = StandardScaler()

    for column in ["Time", "Amount"]:
        X_train[column] = scaler.fit_transform(X_train[[column]])
        X_test[column] = scaler.transform(X_test[[column]])

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train, y_train
    )

    return X_train_resampled, X_test, y_train_resampled, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data(
        "data/creditcard.csv"
    )

    print("Training data after SMOTE:", X_train.shape)
    print("\nClass distribution after SMOTE:")
    print(y_train.value_counts())
    print("\nTest data:", X_test.shape)