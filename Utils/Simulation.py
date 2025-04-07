import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for simulation
n_samples = 50  # Number of data points (samples)

# Simulate random values for RSI (between 0 and 100), MACD (between -1 and 1), and SMA (between 90 and 110)
rsi_values = np.random.uniform(30, 70, size=n_samples)
macd_values = np.random.uniform(-1, 1, size=n_samples)
sma_values = np.random.uniform(90, 110, size=n_samples)

# Define the function to calculate returns based on the indicators (with some randomness)
# Using random weights for each signal, plus some noise to simulate market unpredictability
w_rsi = 5   # Weight for RSI
w_macd = 10  # Weight for MACD
w_sma = 2    # Weight for SMA
noise = np.random.normal(0, 0.2, size=n_samples)
# Simulate returns based on a combination of RSI, MACD, and SMA


returns = w_rsi * rsi_values + w_macd * macd_values + w_sma * sma_values + noise

# Create the DataFrame with simulated data
data = pd.DataFrame({
    'RSI': rsi_values,
    'MACD': macd_values,
    'SMA': sma_values,
    'Returns': returns
})

# Display the first few rows
print(data.head())

from sklearn.linear_model import LinearRegression

# Feature matrix (RSI, MACD, SMA) and target (Returns)
X = data[['RSI', 'MACD', 'SMA']]
y = data['Returns']

# Initialize and train the model
model = LinearRegression()
model.fit(X, y)

# Print the learned weights (coefficients)
print("Learned weights (coefficients):", model.coef_)