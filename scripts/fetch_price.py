from typing import List

import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from data.database import connect_to_database
from data.stocks import get_stocks


def fetch_prices(host: str, since: date | None = None, tickers: List[str] | None = None):

    conn = connect_to_database(host)
    stocks = get_stocks(conn)
    today = date.today()
    insert_sql = "INSERT IGNORE INTO stock_price (stock_id, date, close_price) VALUES (%s, %s, %s)"
    since = since or date(1900, 1, 1)

    if tickers is not None:
        stocks = [stock for stock in stocks if stock.ticker in tickers]

    for stock in stocks:
        try:
            data = yf.download(stock.ticker, start=since, end=today + timedelta(days=1), progress=False)

            # Skip if there's no 'Close' data
            if data.empty or 'Close' not in data.columns:
                print(f"[WARN] No data for {stock.ticker}")
                continue

            # Handle multi-index (e.g., data['Close']['AAPL']) if multiple tickers were fetched
            if isinstance(data.columns, pd.MultiIndex):
                if stock.ticker not in data['Close'].columns:
                    print(f"[WARN] No 'Close' data for {stock.ticker}")
                    continue
                close_series = data['Close'][stock.ticker]
            else:
                close_series = data['Close']

            # Make sure the index is datetime
            close_series.index = pd.to_datetime(close_series.index, errors='coerce')
            close_series = close_series[close_series.index.notnull()]

            # Map date -> price or None
            close_dict = {
                d.date(): None if pd.isna(v) else float(v)
                for d, v in close_series.items()
            }

            # Fallback if no data
            if not close_dict:
                print(f"[INFO] No price entries found for {stock.ticker}")
                continue

            min_date = min(close_dict.keys())
            current_date = min_date

            with conn.cursor() as cursor:
                while current_date <= today:
                    price = close_dict.get(current_date)  # May be None
                    cursor.execute(insert_sql, (stock.id, current_date, price))
                    current_date += timedelta(days=1)

            conn.commit()
            print(f"[OK] {stock.ticker} inserted")

        except Exception as e:
            print(f"[ERROR] Failed for {stock.ticker}: {e}")
