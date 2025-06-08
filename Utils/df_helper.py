from datetime import datetime, timedelta, date
from data.api import API
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any

def combine_signals_from_df(df_scores: pd.DataFrame, tickers: List[str], signal_weights: Dict[str, float]) -> pd.DataFrame:
    # Prepare a DataFrame to store combined signals with the same index as input
    combined_scores = pd.DataFrame(index=df_scores.index)

    for ticker in tickers:
        weighted_sum = pd.Series(0.0, index=df_scores.index)
        total_weight = 0.0

        for signal_name, weight in signal_weights.items():
            col = (signal_name, ticker)
            if col in df_scores.columns:
                weighted_sum += df_scores[col] * weight
                total_weight += weight
            else:
                print(f"Warning: column {col} not found in df_scores")

        # Normalize by total weight (in case some signals are missing)
        if total_weight > 0:
            weighted_sum /= total_weight

        # Assign combined signal for this ticker
        combined_scores[ticker] = weighted_sum

    # Add the date column at the beginning if available
    if ('date', '') in df_scores.columns:
        combined_scores['date'] = df_scores[('date', '')]
        combined_scores.set_index('date', inplace=True)

    return combined_scores


def get_price_history_for_tickers_df(tickers: list[str], 
                                     start_date: date, 
                                     end_date: date):
    
    api = API('192.168.0.165')
    api.ensure_database_is_up_to_date()

    data_from_db = api.get_price_history_for_tickers(tickers, 
                                                     start_date=start_date, 
                                                     end_date=end_date)
    
    df_data = pd.DataFrame.from_dict(data_from_db, orient='columns')
    df_data = df_data.T.dropna(how="all", axis=0)
    df_data = df_data[sorted(df_data.columns)]
    df_data.columns.name = 'Ticker'
    df_data.index.name = 'Date'
    
    return df_data
