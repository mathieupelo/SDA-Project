from datetime import datetime, timedelta, date
from data.api import API
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any

def combine_signals_scores(
    signal_scores: Dict[date, Dict[str, Dict[str, float]]],
    signal_weights: Dict[str, float]
) -> pd.DataFrame:

    # Sort dates for deterministic ordering
    sorted_dates = sorted(signal_scores.keys())
    combined_scores = { }

    for dt in sorted_dates:
        day_data = signal_scores[dt]
        combined_day = { }

        for ticker, ticker_signals in day_data.items():
            weighted_sum = 0.0
            total_weight = 0.0

            for signal_name, weight in signal_weights.items():
                if signal_name in ticker_signals:
                    weighted_sum += ticker_signals[signal_name] * weight
                    total_weight += weight
                else:
                    print(f"Warning: signal '{signal_name}' missing for {ticker} on {dt}")

            combined_day[ticker] = weighted_sum / total_weight if total_weight > 0 else 0.0

        combined_scores[dt] = combined_day

    # Convert to DataFrame
    df_result = pd.DataFrame.from_dict(combined_scores, orient='index')
    df_result.index.name = 'date'
    df_result.sort_index(inplace=True)

    return df_result



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
