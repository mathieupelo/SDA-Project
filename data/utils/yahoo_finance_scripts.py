from datetime import date, timedelta
import yfinance as yf
import pandas as pd


def fill_stocks_price_table_from_yahoo_finance(table: dict[str, float], day: date, tickers: list[str]):
    for ticker in tickers:
        try:
            start_range = day - timedelta(days=30)
            data = yf.download(ticker, start=start_range, end=day + timedelta(days=1), progress=False)
            if 'Close' in data and not data['Close'].empty:
                close_series = data['Close'].dropna()
                valid_data = close_series[close_series.index <= pd.Timestamp(day)]
                if not valid_data.empty:
                    table[ticker] = valid_data.iloc[-1].item()
        except Exception as e:
            print(f"[ERROR] Failed to fetch {ticker} from yfinance up to {day}: {e}")


def fill_stocks_price_history_matrix_from_yahoo_finance(matrix: dict[date, dict[str, float]], first_day: date, last_day: date, tickers: list[str]):
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=first_day, end=last_day, progress=False)
            if 'Close' not in data or data['Close'].empty:
                print(f"[WARN] {ticker} not found on yfinance.")
                continue

            # Ensure index is datetime and clean
            close_series = data['Close'][ticker]

            for time, price in close_series.items():
                time = pd.Timestamp(time).date()
                matrix[time][ticker] = price

        except Exception as e:
            print(f"[ERROR] Failed to fetch from yfinance: {ticker} – {e}")