import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def simulate_signal_data(n_days=500, seed=42, true_weights=[0.4, 0.3, 0.3]):
    np.random.seed(seed)
    rsi_avg = np.random.normal(0, 1, n_days)
    macd_avg = np.random.normal(0, 1, n_days)
    sma_avg = np.random.normal(0, 1, n_days)

    signals = np.vstack([rsi_avg, macd_avg, sma_avg]).T
    noise = np.random.normal(0, 0.2, n_days)
    total_return = signals @ true_weights + noise

    df_train = pd.DataFrame({
        'rsi_avg': rsi_avg,
        'macd_avg': macd_avg,
        'sma_avg': sma_avg,
        'total_return': total_return
    })

    return df_train, true_weights


def simulate_training(n_day):
    df_train, true_weights = simulate_signal_data(n_days=n_day, true_weights=[0.8, 0.1, 0.1])
    X = df_train[['rsi_avg', 'macd_avg', 'sma_avg']].values
    y = df_train['total_return'].values.reshape(-1, 1)

    W, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ W

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mse_weights = np.mean((W.ravel() - true_weights) ** 2)

    return {
        'n_day': n_day,
        'mse': mse,
        'r2': r2,
        'mse_weights': mse_weights
    }


# List of different training durations
training_days = [5, 10, 20, 83, 250, 500, 750, 1000, 1250]

# Run simulations and collect metrics
results = [simulate_training(n) for n in training_days]

# Convert to DataFrame for easy plotting
df_results = pd.DataFrame(results)

# --- Plotting ---
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axs[0].plot(df_results['n_day'], df_results['mse'], marker='o')
axs[0].set_title("MSE vs Training Days")
axs[0].set_ylabel("MSE")
axs[0].grid(True)

axs[1].plot(df_results['n_day'], df_results['r2'], marker='o', color='green')
axs[1].set_title("R² vs Training Days")
axs[1].set_ylabel("R²")
axs[1].grid(True)

axs[2].plot(df_results['n_day'], df_results['mse_weights'], marker='o', color='red')
axs[2].set_title("MSE between learned and true weights vs Training Days")
axs[2].set_xlabel("Training Days")
axs[2].set_ylabel("MSE Weights")
axs[2].grid(True)

plt.tight_layout()
plt.show()
