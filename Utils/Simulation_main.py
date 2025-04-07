import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from Signals import *
from Solver import *


import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score

def simulate_signal_data(n_days=500, seed=4, true_weights=[0.4, 0.3, 0.3]):
    np.random.seed(seed)

    # Simulate daily signals
    rsi_avg = np.random.normal(0, 1, n_days)
    macd_avg = np.random.normal(0, 1, n_days)
    sma_avg = np.random.normal(0, 1, n_days)

    # Generate total return as weighted sum of signals + noise
    signals = np.vstack([rsi_avg, macd_avg, sma_avg]).T
    noise = np.random.normal(0, 0.2, n_days)
    total_return = signals @ true_weights + noise

    # Put everything into a DataFrame
    df_train = pd.DataFrame({
        'rsi_avg': rsi_avg,
        'macd_avg': macd_avg,
        'sma_avg': sma_avg,
        'total_return': total_return
    })

    return df_train, true_weights



def simulate_training(n_day):
    print(f"\nSimulating training for a training of {n_day} days..")
    # Simulate training data
    df_train, true_weights = simulate_signal_data(n_days=n_day, true_weights=[0.8, 0.1, 0.1])

    # Prepare X and y
    X = df_train[['rsi_avg', 'macd_avg', 'sma_avg']].values
    y = df_train['total_return'].values.reshape(-1, 1)

    # Fit using least squares
    W, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    print(f"Learned Weight Matrix (W_{n_day}): \t", W.ravel())

    print(f"True Weights Used to Generate Data: \t", true_weights)

    y_pred = X @ W
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    mse_weights = np.mean((W.ravel() - true_weights) ** 2)
    print(f"MSE {n_day}: {mse:.4f}, R² {n_day}: {r2:.4f}, MSE between weights: {mse_weights:.6f}\n")

# 1 weeks
simulate_training(n_day=5)

# 2 weeks
simulate_training(n_day=10)

# 1 month
simulate_training(n_day=20)

# 3 months
simulate_training(n_day=83)

# A year
simulate_training(n_day=250)

# 5 years
simulate_training(n_day=1250)

# 10 years
simulate_training(n_day=2500)

# 20 years
simulate_training(n_day=5000)

