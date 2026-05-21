import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from scipy.stats import zscore

import joblib

# =========================
# READ DATASET
# =========================

data = pd.read_csv("bank_transactions_data_2.csv")

# Clean column names

data.columns = data.columns.str.strip()

# =========================
# CREATE FRAUD LABELS
# =========================

# Calculate Z-Score

data["ZScore"] = zscore(
    data["TransactionAmount"]
)

# Better threshold

data["Fraud"] = (
    abs(data["ZScore"]) > 1
).astype(int)

# =========================
# FEATURES & TARGET
# =========================

X = data[[
    "TransactionAmount",
    "AccountBalance",
    "LoginAttempts",
    "TransactionDuration"
]]

y = data["Fraud"]

# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# =========================
# TRAIN MODEL
# =========================

model.fit(X_train, y_train)

# =========================
# PREDICT
# =========================

y_pred = model.predict(X_test)

# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# =========================
# REPORT
# =========================

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "fraud_model.pkl")

print("\nModel saved as fraud_model.pkl")

# =========================
# PIE CHART
# =========================

fraud_counts = (
    data["Fraud"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(6, 6))

plt.pie(
    fraud_counts,
    labels=["Safe", "Fraud"],
    autopct='%1.1f%%'
)

plt.title("Fraud vs Safe Transactions")

plt.show()