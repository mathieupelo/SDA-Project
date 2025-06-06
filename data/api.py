from collections import defaultdict
from datetime import date, timedelta
from typing import List
from data.database import connect_to_database
from data.stock_price import get_last_price_date_for_stock
from data.stocks import get_stocks, Stock
import yfinance as yf
import pandas as pd

from scripts.fetch_price import fetch_prices


class API:
    def __init__(self, host: str):
        """
        Args:
            host: Host address of the database machine.
        """
        self._host = host


    def ensure_database_is_up_to_date(self):
        conn = connect_to_database(self._host)
        stocks = get_stocks(conn)
        today = date.today()
        yesterday = today - timedelta(days=1)

        for stock in stocks:
            last_date = get_last_price_date_for_stock(conn, stock.id)

            # If no price exists at all, start from a reasonable default
            start_date = (last_date or date(1900, 1, 1)) + timedelta(days=1)

            if start_date > yesterday:
                continue  # Already up-to-date

            fetch_prices(self._host, tickers=[stock.ticker])


    def get_price_for_tickers(self, tickers: List[str], day: date) -> dict[str, float]:
        """
        Returns the latest known price for each ticker *at or before* the given day.
        """
        conn = connect_to_database(self._host)
        stocks: List[Stock] = get_stocks(conn)
        known_stocks: List[Stock] = [stock for stock in stocks if stock.ticker in tickers]
        unknown_tickers: List[str] = [t for t in tickers if all(s.ticker != t for s in stocks)]
        result: dict[str, float] = { }

        # --- 1. Query DB for known tickers ---
        if known_stocks:
            placeholders = ', '.join(['%s'] * len(known_stocks))
            sql = f"""
                SELECT t.ticker, sp.close_price
                FROM stock_price sp
                JOIN stock t ON sp.stock_id = t.id
                JOIN (
                    SELECT stock_id, MAX(date) as max_date
                    FROM stock_price
                    WHERE date <= %s
                      AND close_price is not null
                      AND stock_id IN ({placeholders})
                    GROUP BY stock_id
                ) AS latest
                ON sp.stock_id = latest.stock_id AND sp.date = latest.max_date
            """
            params = [day] + [stock.id for stock in known_stocks]

            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                for ticker, price in cursor.fetchall():
                    if price is not None:
                        result[ticker] = float(price)


        # --- 2. Fallback to Yahoo Finance for unknown tickers ---
        for ticker in unknown_tickers:
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
        known_stocks: List[Stock] = [stock for stock in stocks if stock.ticker in tickers]
        unknown_tickers: List[str] = [t for t in tickers if all(s.ticker != t for s in stocks)]
        result: dict[date, dict[str, float]] = defaultdict(dict)

        # For tracking dates we pulled from DB only
        db_dates_seen: dict[str, set[date]] = { }

        # --- 1. Fetch from DB for known stocks ---
        if known_stocks:
            placeholders = ', '.join(['%s'] * len(known_stocks))
            sql = f"""
                SELECT sp.date, s.ticker, sp.close_price
                FROM stock_price sp
                JOIN stock s ON sp.stock_id = s.id
                WHERE s.id IN ({placeholders})
                  AND sp.date BETWEEN %s AND %s
                ORDER BY sp.date
            """
            params = [stock.id for stock in known_stocks] + [str(start_date), str(end_date)]

            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                rows = cursor.fetchall()

            for time, ticker, price in rows:
                if db_dates_seen.get(ticker) is None:
                    db_dates_seen[ticker] = set()

                day = pd.Timestamp(time).date()
                result[day][ticker] = price
                db_dates_seen[ticker].add(day)

        # --- 2. Fetch from Yahoo Finance for unknown stocks ---
        for ticker in unknown_tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), progress=False)
                if 'Close' not in data or data['Close'].empty:
                    print(f"[WARN] {ticker} not found on yfinance.")
                    continue

                # Ensure index is datetime and clean
                close_series = data['Close'][ticker]

                for time, price in close_series.items():
                    time = pd.Timestamp(time).date()
                    result[time][ticker] = price

            except Exception as e:
                print(f"[ERROR] Failed to fetch from yfinance: {ticker} – {e}")

        # --- 3. Validate known DB stocks: warn if any date in range is missing for that ticker ---
        expected_dates = {start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)}


        for stock in known_stocks:
            present_dates = db_dates_seen.get(stock.ticker, set())
            missing_dates = expected_dates - present_dates

            for day in sorted(missing_dates):
                print(f"[WARN] Missing DB price data for {stock.ticker} on {day}")

        return dict(result)
