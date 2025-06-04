from typing import List
from mysql.connector.abstracts import MySQLConnectionAbstract
from data.stocks import Stock
from datetime import date

class Portfolio:
    class PStock:
        class PSignal:
            def __init__(self, signal, version: int, score: float):
                self.signal = signal
                self.version = version
                self.score = score

        # Portfolio.Stock
        def __init__(self, stock: Stock, weight: float, alpha_score: float, signals: List[PSignal]):
            self.stock = stock
            self.weight = weight
            self.alpha_score = alpha_score
            self.signals = signals

    # Portfolio
    def __init__(self, p_id: str, creation_date: date, stocks: List[PStock]):
        self.id = p_id
        self.creation_date = creation_date
        self.stocks = stocks




    # def get_signals(self) -> List[SignalBase]:
    # def get_weight_table(self) -> pd.DataFrame:

def get_portfolio(conn: MySQLConnectionAbstract, portfolio_id: str) -> Portfolio | None:
    """
    Attempts to fetch a portfolio from the database for portfolio ID.

    Parameters:
    - conn: The MySQL connection object (you can get it from database_utils).
    - portfolio_id: The id of the portfolio to fetch.

    Returns:
        Portfolio | None if the stock does not exist.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT date FROM portfolio WHERE id = '%s'" % portfolio_id)
    row = cursor.fetchone()

    if not row:
        return None

    creation_date = row[0]

    # Get associated stocks from portfolio_stock
    cursor.execute("""
                   SELECT s.id, s.name, s.ticker, ps.weight, ps.alpha_score
                   FROM portfolio_stock ps
                            JOIN stock s ON s.id = ps.stock_id
                   WHERE ps.portfolio_id = %s
                   """, (portfolio_id,))
    rows = cursor.fetchall()

    stocks: List[Portfolio.PStock] = []
    for stock_id, name, ticker, weight, alpha_score in rows:
        stock = Stock(stock_id, name, ticker)
        stocks.append(Portfolio.PStock(stock, weight, alpha_score, signals=[]))

    return Portfolio(p_id=portfolio_id, creation_date=creation_date, stocks=stocks)


def cache_portfolio(conn: MySQLConnectionAbstract, portfolio: Portfolio) -> None:
    """
    Caches a Portfolio into the database.

    Parameters:
    - conn: The MySQL connection object.
    - portfolio: The Portfolio object to insert.
    """
    cursor = conn.cursor()

    # Insert into portfolio table
    cursor.execute("""
        INSERT INTO portfolio (id, date)
        VALUES (%s, %s)
    """, (portfolio.id, portfolio.creation_date))

    # Insert into portfolio_stock table
    stock_rows = [
        (portfolio.id, pstock.stock.id, pstock.weight, pstock.alpha_score)
        for pstock in portfolio.stocks
    ]

    if stock_rows:
        cursor.executemany("""
            INSERT INTO portfolio_stock (portfolio_id, stock_id, weight, alpha_score)
            VALUES (%s, %s, %s, %s)
        """, stock_rows)

    conn.commit()