import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load the credit card fraud dataset."""
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_data("data/creditcard.csv")

    print("Dataset shape:", df.shape)
    print("\nClass distribution:")
    print(df["Class"].value_counts())