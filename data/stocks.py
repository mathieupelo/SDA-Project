from mysql.connector.abstracts import MySQLConnectionAbstract
from typing import List


class Stock:
    def __init__(self, stock_id: str, name: str, ticker: str):
        self._id = stock_id
        self._name = name
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


