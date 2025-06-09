from data.api import API
from data.signals import store_signal_scores_for_ticker, get_missing_signal_scores_for_ticker
from datetime import date, timedelta, datetime
from data.utils.database import connect_to_database
from signals.macd import MACDSignal
from signals.rsi import RSISignal
from signals.signal_registry import SignalRegistry
from signals.sma import SMASignal
import math
import pandas as pd


def run_signal_pipeline(host: str):
    """
    Evaluates signals on all stocks for missing dates and stores the results to the database.
    Args:
        host: Hostname or IP address of the database server
    """
    api = API(host)

    # Prepare signal registry
    print('Initialize signal registry...')
    signal_registry = SignalRegistry()
    signal_registry.register(RSISignal(period=14))
    signal_registry.register(MACDSignal())
    signal_registry.register(SMASignal())

    print('Ensure database is up to date...')
    api.ensure_database_is_up_to_date()

    print('Fetch tickers from all universes...')
    tickers = api.get_tickers_from_all_universes()

    print('Fetch price history...')
    price_history = api.get_price_history_for_tickers(tickers)

    print('Prepare Pandas DataFrame...')
    price_history_df = pd.DataFrame.from_dict(price_history, orient='columns').T.dropna(how="all", axis=0)

    for ticker in tickers:

        if ticker not in price_history_df.columns:
            print(f"ERROR | No price history for {ticker}. Skipping signal computation.")
            continue

        print(f'[{ticker}] | Fetch missing signal scores...')
        missing_signal_scores = get_missing_signal_scores_for_ticker(host, ticker)

        if not missing_signal_scores:
            print(f"[{ticker}] | No signal scores missing.")
            continue

        for signal_id, latest in missing_signal_scores:

            print(f'[{ticker}] | Computing {signal_id} scores from {latest}')
            signal = signal_registry.get_signal(signal_id)

            if signal is None:
                print(f"ERROR | Signal {signal_id} not found in registry.")
                continue

            min_lookback = signal.get_min_lookback_period()
            max_lookback = signal.get_max_lookback_period()

            if min_lookback > max_lookback:
                print(f'ERROR | Signal {signal_id} has min lookback ({min_lookback}) bigger than max lookback ({max_lookback}).')
                continue

            computed_scores: list[tuple[str, datetime.date, float]] = []

            # Find the first valid day after `latest` with enough data
            valid_start_day = None
            scan_day = latest

            while scan_day < date.today():
                scan_day += timedelta(days=1)

                if (scan_day not in price_history_df.index) or pd.isna(price_history_df.at[scan_day, ticker]):
                    continue

                close_prices = price_history_df.loc[:scan_day, ticker].dropna().tail(max_lookback)

                if len(close_prices) >= min_lookback:
                    valid_start_day = scan_day - timedelta(days=1)
                    break

            if valid_start_day is None:
                print(f"[{ticker}] | [{signal_id}] | Not enough data available to compute scores.")
                continue

            day = valid_start_day

            while day < date.today():
                day += timedelta(days=1)

                # No signal score for days the market is closed
                if (day not in price_history_df.index) or pd.isna(price_history_df.at[day, ticker]):
                    continue

                # Extract the relevant close prices before the target date
                close_prices = price_history_df.loc[:day, ticker].dropna().tail(max_lookback)

                # Don't evaluate signal if the lookback period is not big enough.
                if len(close_prices) < min_lookback:
                    print(f'[{ticker}] | [{signal_id}] | Lookback too small on {day}.')
                    break

                score = signal.calculate(close_prices, ticker, day)

                if (score is None) or (math.isnan(score)):
                    print(f"WARNING | [{signal_id}] Signal {signal_id} .")
                    continue

                computed_scores.append((signal_id, day, score))

            print(f'[{ticker}] | Computed {len(computed_scores)} {signal_id} score(s).')

            if len(computed_scores) > 0:
                conn = connect_to_database(host)
                store_signal_scores_for_ticker(conn, ticker, computed_scores)


    print(f"SUCCESS | Finished computing signal scores for {len(tickers)} ticker(s).")