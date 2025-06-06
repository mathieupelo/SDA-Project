from collections import defaultdict
from datetime import date, timedelta
from typing import List
from data.database import connect_to_database
from data.stocks import get_stocks
import yfinance as yf
import pandas as pd

class API:
    def __init__(self, host: str):
        self._host = host



    def get_price_history(self, tickers: List[str], start_date: date, end_date: date) -> dict[date, dict[str, float]]:

        conn = connect_to_database(self._host)
        stocks = get_stocks(conn)
        known_stocks = {s.ticker: s.id for s in stocks}
        known_tickers = set(known_stocks.keys())

        tickers_known = [t for t in tickers if t in known_tickers]
        tickers_unknown = [t for t in tickers if t not in known_tickers]

        result: dict[date, dict[str, float]] = defaultdict(dict)

        # For tracking dates we pulled from DB only
        db_dates_seen: set[date] = set()

        # --- 1. Fetch from DB for known stocks ---
        if tickers_known:
            placeholders = ', '.join(['%s'] * len(tickers_known))
            sql = f"""
                SELECT sp.date, s.ticker, sp.close_price
                FROM stock_price sp
                JOIN stock s ON sp.stock_id = s.id
                WHERE s.ticker IN ({placeholders})
                  AND sp.date BETWEEN %s AND %s
                ORDER BY sp.date
            """
            params = tickers_known + [start_date, end_date]

            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()

            for dt, ticker, price in rows:
                result[dt][ticker] = price
                db_dates_seen.add(dt)

        # --- 2. Fetch from Yahoo Finance for unknown stocks ---
        for ticker in tickers_unknown:
            try:
                data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), progress=False)
                if 'Close' not in data or data['Close'].empty:
                    print(f"[WARN] {ticker} not found on yfinance.")
                    continue

                # Ensure index is datetime and clean
                close_series = data['Close'][ticker]

                for time, price in close_series.items():
                    day = pd.Timestamp(time).date()
                    result[day][ticker] = price

            except Exception as e:
                print(f"[ERROR] Failed to fetch from yfinance: {ticker} – {e}")

        # --- 3. Validate known DB stocks: warn if any date in range is missing for that ticker ---
        expected_dates = {start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)}

        # Build per-ticker date set from DB result
        ticker_dates_in_db: dict[str, set[date]] = defaultdict(set)
        for dt, ticker, _ in rows:
            ticker_dates_in_db[ticker].add(dt)

        for ticker in tickers_known:
            present_dates = ticker_dates_in_db.get(ticker, set())
            missing_dates = expected_dates - present_dates

            for d in sorted(missing_dates):
                print(f"[WARN] Missing DB price data for {ticker} on {d}")

        return dict(result)


