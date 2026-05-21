from flask import Flask, render_template, request
import pandas as pd
import joblib

# =========================
# CREATE APP
# =========================

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================

model = joblib.load("fraud_model.pkl")

# =========================
# HOME PAGE
# =========================

@app.route('/')
def home():
    return render_template("index.html")

# =========================
# PREDICTION
# =========================

@app.route('/predict', methods=['POST'])
def predict():

    # Get form values

    transaction_amount = float(
        request.form['transaction_amount']
    )

    account_balance = float(
        request.form['account_balance']
    )

    login_attempts = int(
        request.form['login_attempts']
    )

    transaction_duration = float(
        request.form['transaction_duration']
    )


    # =========================
    # CREATE DATAFRAME
    # =========================

    input_data = pd.DataFrame([{
        "TransactionAmount": transaction_amount,
        "AccountBalance": account_balance,
        "LoginAttempts": login_attempts,
        "TransactionDuration": transaction_duration
    }])

    # =========================
    # PREDICT
    # =========================

    prediction = model.predict(input_data)

    # Fraud probability

    probability = model.predict_proba(
        input_data
    )[0][1]

    fraud_percent = round(
        probability * 100,
        2
    )

    # =========================
    # RESULT
    # =========================

    if prediction[0] == 1:
        result = (
            f"⚠️ Fraud Risk Detected "
            f"({fraud_percent}% risk)"
        )
    else:
        result = (
            f"✅ Safe Transaction "
            f"({fraud_percent}% risk)"
        )

    return render_template(
        "index.html",
        prediction_text=result
    )

# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(debug=True)