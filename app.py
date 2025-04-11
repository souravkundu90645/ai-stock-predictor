import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping
import smtplib
from email.message import EmailMessage

# ---------- CONFIG ---------- #
TICKER = '^NSEI'             # NIFTY 50
PERIOD = '6mo'               # Last 6 months
EPOCHS = 20
BATCH_SIZE = 32
WINDOW_SIZE = 60
TEST_SPLIT = 0.2
EMAIL_ENABLED = False
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_RECEIVER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"  # Use an app password
# ---------------------------- #

# 📦 Load and preprocess data
df = yf.download(TICKER, period=PERIOD, interval='1d')
df = df[['Close']].dropna()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

X, y = [], []
for i in range(WINDOW_SIZE, len(scaled_data)):
    X.append(scaled_data[i - WINDOW_SIZE:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 📊 Train/Test Split
split_index = int(len(X) * (1 - TEST_SPLIT))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 🧠 Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# 🚀 Train
early_stop = EarlyStopping(monitor='loss', patience=5)
history = model.fit(X_train, y_train, epochs=EPOCHS,
                    batch_size=BATCH_SIZE, callbacks=[early_stop])

# 📈 Evaluate
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
rmse = math.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

# 💾 Save model with timestamp
timestamp = datetime.now().strftime("%Y-%m")
model_name = f"lstm_model_{timestamp}.h5"
model.save(model_name)
print(f"✅ Model saved as {model_name}")

# 📝 Log training results
log_path = "training_log.csv"
log_entry = f"{timestamp},{TICKER},{rmse:.4f},{EPOCHS}\n"

if not os.path.exists(log_path) or os.stat(log_path).st_size == 0:
    with open(log_path, "w") as f:
        f.write("Timestamp,Ticker,RMSE,Epochs\n")
with open(log_path, "a") as f:
    f.write(log_entry)

# 📬 Email training summary (optional)
def send_email(subject, body):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.send_message(msg)

if EMAIL_ENABLED:
    summary = f"""📊 LSTM Training Summary

Model: {model_name}
Ticker: {TICKER}
RMSE: {rmse:.4f}
Epochs: {EPOCHS}
"""
    send_email("📈 Monthly LSTM Training Report", summary)
    print("📧 Email summary sent.")

# 🖼️ (Optional) Plot actual vs predicted
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(y_pred_rescaled, label='Predicted')
plt.title(f"{TICKER} - Actual vs Predicted")
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.savefig(f"prediction_plot_{timestamp}.png")
