# The Syndicate — Real-Time Credit Card Fraud Detector

A fraud detection pipeline for highly imbalanced credit card transaction data: preprocessing,
class balancing with SMOTE, a Random Forest classifier producing fraud probability scores, and
a simulated real-time inference loop that raises a console alert whenever a transaction's fraud
probability exceeds 95%.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) -
Machine Learning Group, Universite Libre de Bruxelles (ULB), hosted on Kaggle.

- 284,807 transactions made by European cardholders over two days in September 2013
- 492 fraudulent transactions (0.172% of all transactions) - highly imbalanced
- Features V1-V28 are PCA components, applied for confidentiality
- Time (seconds since first transaction) and Amount are the only non-PCA features
- Class is the target: 1 = fraud, 0 = genuine

See data/README.md for how to obtain the file.

## Setup

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Download creditcard.csv from Kaggle and place it at data/creditcard.csv.

## Usage

python src/data_loader.py
python -m src.preprocessing
python -m src.model
python -m src.evaluation
python scripts/realtime_inference.py

## Method

1. Split first, balance second - train/test split is stratified before any resampling.
2. Only Time and Amount are scaled (StandardScaler, fit on train only).
3. SMOTE is applied to the training set only, after the split.
4. RandomForestClassifier with class_weight="balanced" on top of SMOTE.
5. Evaluated with precision/recall on the fraud class, confusion matrix, ROC-AUC, PR-AUC.

## Real-time inference simulation

scripts/realtime_inference.py loads the saved model + scaler, streams dummy transactions,
scores each one, and prints an alert whenever fraud probability exceeds 95%.

## License

See LICENSE.
