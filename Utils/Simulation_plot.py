import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


def simulate_signal_data(n_days=500, seed=42, true_weights=None):
    np.random.seed(seed)
    n_signals = len(true_weights)

    # Generate signals
    signals = np.random.normal(0, 1, (n_days, n_signals))
    
    # Generate noisy target (total return)
    noise = np.random.normal(0, 50, n_days)
    total_return = signals @ np.array(true_weights) + noise

    # Create column names dynamically
    columns = [f'signal_{i+1}' for i in range(n_signals)]
    df_train = pd.DataFrame(signals, columns=columns)
    df_train['total_return'] = total_return

    return df_train, true_weights


def simulate_training(n_day, true_weights):
    df_train, _ = simulate_signal_data(n_days=n_day, true_weights=true_weights)
    
    signal_cols = [col for col in df_train.columns if col.startswith('signal_')]
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


# Configurable parameters
training_days = [5, 10, 20, 83, 250, 500, 750, 1000, 1250]
signal_sizes = [3, 5, 10]  

# Store results
all_results = {}

for n_signals in signal_sizes:
    # Generate normalized random true weights
    rng = np.random.default_rng(seed=42 + n_signals)
    raw_weights = rng.random(n_signals)
    true_weights = raw_weights / raw_weights.sum()

    # Simulate for all training days
    results = [simulate_training(n, true_weights) for n in training_days]
    df_results = pd.DataFrame(results)
    all_results[n_signals] = df_results

# --- Plotting all on same graph ---
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']  # Extend if needed

for idx, (n_signals, df_results) in enumerate(all_results.items()):
    label = f"{n_signals} signals"
    color = colors[idx % len(colors)]
    
    axs[0].plot(df_results['n_day'], df_results['mse'], marker='o', label=label, color=color)
    axs[1].plot(df_results['n_day'], df_results['r2'], marker='o', label=label, color=color)
    axs[2].plot(df_results['n_day'], df_results['mse_weights'], marker='o', label=label, color=color)

# Titles and labels
axs[0].set_title("MSE vs Training Days")
axs[0].set_ylabel("MSE")
axs[0].grid(True)

axs[1].set_title("R² vs Training Days")
axs[1].set_ylabel("R²")
axs[1].grid(True)

axs[2].set_title("MSE between learned and true weights vs Training Days")
axs[2].set_xlabel("Training Days")
axs[2].set_ylabel("MSE Weights")
axs[2].grid(True)

# Add legends to each subplot
for ax in axs:
    ax.legend()

plt.tight_layout()
plt.show()