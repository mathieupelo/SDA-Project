import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def simulate_realistic_signal_data(n_days=500, seed=42, true_weights=None, signal_noise_std=0.05, return_noise_std=50):
    np.random.seed(seed)
    n_signals = len(true_weights)

    # Simulate 3 underlying "signal processes" with some correlation
    base_signal = np.random.normal(0, 1, (n_days, 1))  # common market factor
    noise = np.random.normal(0, signal_noise_std, (n_days, n_signals))  # small individual noise

    # Simulated signals: base + small noise per signal
    signals = base_signal + noise

    # Simulated total return using a true linear combination of the signals
    total_return = signals @ np.array(true_weights) + np.random.normal(0, return_noise_std, n_days)

    # Match your naming convention
    columns = ['rsi_avg', 'macd_avg', 'sma_avg'][:n_signals]
    df = pd.DataFrame(signals, columns=columns)
    df['total_return'] = total_return

    return df, true_weights


def simulate_training(n_day, true_weights):
    df_train, _ = simulate_realistic_signal_data(n_days=n_day, true_weights=true_weights)
    
    signal_cols = ['rsi_avg', 'macd_avg', 'sma_avg'][:len(true_weights)]
    X = df_train[signal_cols].values
    y = df_train['total_return'].values.reshape(-1, 1)

    W, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ W

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mse_weights = np.mean((W.ravel() - np.array(true_weights)) ** 2)

    return {
        'n_day': n_day,
        'mse': mse,
        'r2': r2,
        'mse_weights': mse_weights
    }


# Simulate with true signal weights (e.g., based on intuition or experimentation)
true_weights = [0.4, 0.35, 0.25]  # These should sum to 1, like your softmaxed W

# Simulate a dataset with 500 days of data
df_simulated, used_weights = simulate_realistic_signal_data(n_days=500, true_weights=true_weights)

# Display first few rows to inspect
print(df_simulated.head())
