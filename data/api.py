from collections import defaultdict
from datetime import date, timedelta
from typing import List
from data.database import connect_to_database
from data.stocks import get_stocks
import yfinance as yf
import pandas as pd


class API:
    def __init__(self, host: str):
        """
        Args:
            host: Host address of the database machine.
        """
        self._host = host

    def get_price_for_tickers(self, tickers: List[str], day: date) -> dict[str, float]:
        """
        Returns the latest known price for each ticker *at or before* the given day.
        """
        conn = connect_to_database(self._host)
        stocks = get_stocks(conn)
        known_stocks = {s.ticker: s.id for s in stocks}
        known_tickers = set(known_stocks.keys())

        tickers_known = [t for t in tickers if t in known_tickers]
        tickers_unknown = [t for t in tickers if t not in known_tickers]

        result: dict[str, float] = {}

        # --- 1. Query DB for known tickers ---
        if tickers_known:
            placeholders = ', '.join(['%s'] * len(tickers_known))
            sql = f"""
                SELECT t.ticker, sp.close_price
                FROM stock_price sp
                JOIN stock t ON sp.stock_id = t.id
                JOIN (
                    SELECT stock_id, MAX(date) as max_date
                    FROM stock_price
                    WHERE date <= %s
                      AND stock_id IN (
                          SELECT id FROM stock WHERE ticker IN ({placeholders})
                      )
                    GROUP BY stock_id
                ) AS latest
                ON sp.stock_id = latest.stock_id AND sp.date = latest.max_date
            """
            params = [day] + tickers_known

            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                for ticker, price in cursor.fetchall():
                    if price is not None:
                        result[ticker] = float(price)

        # --- 2. Fallback to Yahoo Finance for unknown tickers ---
        for ticker in tickers_unknown:
            try:
                start_range = day - timedelta(days=30)
                data = yf.download(ticker, start=start_range, end=day + timedelta(days=1), progress=False)
                if 'Close' in data and not data['Close'].empty:
                    close_series = data['Close'].dropna()
                    valid_data = close_series[close_series.index <= pd.Timestamp(day)]
                    if not valid_data.empty:
                        result[ticker] = valid_data.iloc[-1].item()
            except Exception as e:
                print(f"[ERROR] Failed to fetch {ticker} from yfinance up to {day}: {e}")

        return result



    def get_price_history_for_tickers(self, tickers: List[str], start_date: date, end_date: date) -> dict[date, dict[str, float]]:
        """
        Args:
            tickers: list of ticker symbols to fetch the price_history for
            start_date: first date to fetch the price_history for
            end_date: last date to fetch the price_history for

        Returns:
            the price history table indexed by date, then by ticker symbol
        """
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
