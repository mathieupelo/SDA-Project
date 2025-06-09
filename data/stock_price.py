from datetime import date, timedelta
from mysql.connector.abstracts import MySQLConnectionAbstract
from data.stocks import Stock
import pandas as pd


def fill_stocks_price_history_matrix(
        conn: MySQLConnectionAbstract,
        matrix: dict[date, dict[str, float]],
        first_day: date,
        last_day: date,
        stocks: list[Stock]):
    """
    Fetches historical stock prices for a given stock list between the given first and last days and stores it into
    the matrix.
    """

    if not stocks:
        return

    sql_params = [stock.id for stock in stocks] + [str(first_day), str(last_day)]
    sql_placeholders = ', '.join(['%s'] * len(stocks))
    sql = f"""
        SELECT sp.date, s.ticker, sp.close_price
        FROM stock_price sp
        JOIN stock s ON sp.stock_id = s.id
        WHERE s.id IN ({sql_placeholders})
          AND sp.date >= %s AND sp.date < %s
        ORDER BY sp.date
    """

    with conn.cursor() as cursor:
        cursor.execute(sql, sql_params)
        rows = cursor.fetchall()

    seen_tickers = set()

    for time, ticker, price in rows:
        day = pd.Timestamp(time).date()
        matrix[day][ticker] = float(price) if price else None
        seen_tickers.add(ticker)

    missing_tickers = [stock.ticker for stock in stocks if stock.ticker not in seen_tickers]
    if missing_tickers:
        print(f"[WARNING] No price data found for tickers: {', '.join(missing_tickers)}")



def fill_stocks_price_table(
        conn: MySQLConnectionAbstract,
        table: dict[str, float],
        day: date,
        stocks: list[Stock]):
    """
    Fetches historical stock prices for a given stock list at the given day and stores it into the table.

    Parameters:
        conn: MySQL connection object.
        table: table to fill the historical stock prices in.
        day: Date to fetch the historical stock prices for.
        stocks: List of stocks to fetch the historical stock prices for.
    """
    if len(stocks) == 0:
        return

    sql_params = [day] + [stock.id for stock in stocks]
    sql_placeholder = ', '.join(['%s'] * len(stocks))
    sql = f"""
                    SELECT t.ticker, sp.close_price
                    FROM stock_price sp
                    JOIN stock t ON sp.stock_id = t.id
                    JOIN (
                        SELECT stock_id, MAX(date) as max_date
                        FROM stock_price
                        WHERE date <= %s
                          AND close_price is not null
                          AND stock_id IN ({sql_placeholder})
                        GROUP BY stock_id
                    ) AS latest
                    ON sp.stock_id = latest.stock_id AND sp.date = latest.max_date
                """

    with conn.cursor() as cursor:
        cursor.execute(sql, sql_params)
        for ticker, price in cursor.fetchall():
            if price is not None:
                table[ticker] = float(price)


def get_stock_price(conn: MySQLConnectionAbstract, stock_id: str, when: date | str) -> float or None:
    """
    Fetches the closing price of a stock for a specific date.

    Parameters:
        conn: MySQL connection object.
        stock_id (str): UUID of the stock in the 'stock' table.
        when (str or datetime.date): The date to retrieve the price for.

    Returns:
        Decimal or None: The closing price if found, otherwise None.
    """
    sql = """
          SELECT close_price
          FROM stock_price
          WHERE stock_id = %s \
            AND date = %s
          LIMIT 1 \
          """

    cursor = conn.cursor()
    cursor.execute(sql, (stock_id, when))
    result = cursor.fetchone()

    return result[0] if result else None


def get_last_price_date_for_stock(conn : MySQLConnectionAbstract, stock_id: str) -> date | None:
    """
    Returns the most recent date (regardless of price being NULL) for which the given stock has a row in stock_price.
    """
    sql = "SELECT MAX(date) FROM stock_price WHERE stock_id = %s"

    with conn.cursor() as cursor:
        cursor.execute(sql, (stock_id,))
        row = cursor.fetchone()
        return row[0] if row and row[0] else None