import uuid

import pandas as pd
from mysql.connector.abstracts import MySQLConnectionAbstract
from typing import List, Dict
from datetime import date


class Stock:
    def __init__(self, ticker: str, stock_id: str = None, name: str = None):
        self._id = stock_id or str(uuid.uuid1())
        self._name = name or f"{ticker}_TEST"
        self._ticker = ticker

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def ticker(self):
        return self._ticker

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        return self.id == other.id and self.ticker == other.ticker and self.name == other.name


def get_stocks(conn: MySQLConnectionAbstract) -> List[Stock]:
    """
    Retrieves the stock list from the database.

    Parameters:
    - conn: The MySQL connection object (you can get it from database_utils).

    Returns:
        List[Stock]: List of Stock instances from the database.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, ticker FROM stock")
    rows = cursor.fetchall()
    stocks = [Stock(stock_id, name, ticker) for (stock_id, name, ticker) in rows]
    return stocks


def get_stock(conn: MySQLConnectionAbstract, ticker: str) -> Stock | None:
    """
    Attempts to fetch a stock from the database for ticker symbol.

    Parameters:
    - ticker: The ticker symbol.
    - conn: The MySQL connection object (you can get it from database_utils).

    Returns:
        Stock instance from the database, if any.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, ticker FROM stock WHERE ticker = '%s'" % ticker)
    row = cursor.fetchone()

    if row:
        stock_id, name, ticker_symbol = row
        return Stock(stock_id, name, ticker_symbol)

    return None


def get_stock_price_table(conn: MySQLConnectionAbstract, stock_id: str, start: date, end: date) \
        -> dict[date, float] | None:
    """
    Fetches historical stock prices for a given stock_id and date range.

    Parameters:
        stock_id (str): UUID of the stock in the stock table.
        conn: MySQL connection object.
        start (str or datetime.date): Start date (inclusive), e.g. '2020-01-01'.
        end (str or datetime.date): End date (inclusive), e.g. '2024-12-31'.

    Returns:
        dict[date, float] | None: A dict with {date: close_price} or None if no rows are found.
    """
    sql = """
          SELECT date, close_price
          FROM stock_price
          WHERE stock_id = %s
            AND date BETWEEN %s AND %s
          ORDER BY date
          """

    cursor = conn.cursor()
    cursor.execute(sql, (stock_id, start, end))
    rows = cursor.fetchall()

    if not rows:
        return None

    return {row[0]: float(row[1]) for row in rows}


def get_stock_price(conn: MySQLConnectionAbstract, stock_id: str, when: date | str) -> float or None:
    """
    Fetches the closing price of a stock for a specific date.

    Parameters:
        conn: MySQL connection object.
        stock_id (str): UUID of the stock in the 'stock' table.
        when (str or datetime.date): The date to retrieve the price for.

    Returns:
        float or None: The closing price if found, otherwise None.
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