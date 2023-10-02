# bharat_intern- data science tasks
# TASK 1 (STOCK PREDICTION)
# write a code and Take stock price of any company you want and predicts its price by using LSTM.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the stock symbol and date range
symbol = "AAPL"  # Change to the stock symbol you w-ant to predict
start_date = "2020-01-01"
end_date = "2021-12-31"

# Download historical stock data
data = yf.download(symbol, start=start_date, end=end_date)

# Extract the 'Close' prices
df = data[['Close']]

# Normalize the data
scaler = MinMaxScaler()
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Create sequences for the LSTM model
sequence_length = 10  # You can adjust this value
sequences = []
target = []

for i in range(len(df) - sequence_length):
    sequences.append(df.iloc[i:i+sequence_length, 0].values)
    target.append(df.iloc[i+sequence_length, 0])

sequences = np.array(sequences)
target = np.array(target)

# Split data into training and testing sets
train_size = int(0.8 * len(sequences))
X_train, X_test = sequences[:train_size], sequences[train_size:]
y_train, y_test = target[:train_size], target[train_size:]

# Create the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Plot the actual vs. predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.legend()
plt.title(f"{symbol} Stock Price Prediction")
plt.show()
