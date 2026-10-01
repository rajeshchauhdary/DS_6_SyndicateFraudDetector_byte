# Data

This project uses the Credit Card Fraud Detection dataset published by the Machine Learning
Group at Universite Libre de Bruxelles (ULB):

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Getting the file

The raw CSV (~150 MB) is not committed to this repository (see .gitignore).

1. Download creditcard.csv from the Kaggle link above.
2. Place it at data/creditcard.csv.

## Schema

| Column | Description |
|---|---|
| Time | Seconds elapsed since the first transaction in the dataset |
| V1-V28 | PCA components of the original (confidential) transaction features |
| Amount | Transaction amount |
| Class | Target: 1 = fraud, 0 = genuine |

284,807 transactions total, 492 fraudulent (0.172%). No missing values.
