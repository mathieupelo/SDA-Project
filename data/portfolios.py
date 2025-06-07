from mysql.connector.abstracts import MySQLConnectionAbstract
from Utils.signals import SignalBase
from data.solver_config import SolverConfig
from data.stocks import Stock
from datetime import date
from typing import Iterable

class Portfolio:
    class StockMetadata:
        def __init__(self, weight: float, alpha_score: float):
            self._weight = weight
            self._alpha_score = alpha_score

        @property
        def weight(self) -> float:
            return self._weight

        @property
        def alpha_score(self) -> float:
            return self._alpha_score

        def __repr__(self) -> str:
            return f"(w={self.weight:.4f}, a={self.alpha_score:.4f})"

    # Portfolio
    def __init__(self, p_id: str, creation_date: date, stocks: dict[Stock, StockMetadata], config: SolverConfig):
        self._id = p_id
        self._creation_date = creation_date
        self._stocks = stocks
        self._config = config

    @property
    def id(self) -> str:
        return self._id

    @property
    def creation_date(self) -> date:
        return self._creation_date

    @property
    def stocks(self) -> dict[Stock, StockMetadata]:
        return self._stocks

    @property
    def config(self) -> SolverConfig:
        return self._config

    def get_weight_table(self) -> dict[str, float]:
        return { stock.ticker: metadata.weight for stock, metadata in self._stocks.items() }



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
    stocks: dict[Stock, Portfolio.StockMetadata] = { }

    for stock_id, name, ticker, weight, alpha_score in rows:
        stock = Stock(stock_id, name, ticker)
        metadata = Portfolio.StockMetadata(weight, alpha_score)
        stocks[stock] = metadata

    return Portfolio(p_id=portfolio_id, creation_date=creation_date, stocks=stocks)


def cache_portfolio_data(
        conn: MySQLConnectionAbstract,
        portfolio: Portfolio,
        signals: dict[SignalBase, float],
        yearly_return: float):
    """
    Caches a Portfolio into the database.

    Parameters:
    - conn: The MySQL connection object.
    - portfolio: The Portfolio object to insert.
    """
    ensure_signals_are_stored_in_db(conn, signals)
    cursor = conn.cursor()

    # Cache to 'portfolio' database table.
    cursor.execute("""
        INSERT INTO portfolio (id, date, risk_aversion, max_weight, yearly_return)
        VALUES (%s, %s, %s, %s, %s)
    """, (portfolio.id, portfolio.creation_date, portfolio.config.risk_aversion, portfolio.config.max_weight_threshold, yearly_return))

    stock_rows = [
        (portfolio.id, stock.id, metadata.weight, metadata.alpha_score)
        for stock, metadata in portfolio.stocks.items()
    ]

    if stock_rows:
        cursor.executemany("""
            INSERT INTO portfolio_stock (portfolio_id, stock_id, weight, alpha_score)
            VALUES (%s, %s, %s, %s)
        """, stock_rows)

    signal_rows = [
        (portfolio.id, signal.name, weight)
        for signal, weight in signals.items()
    ]

    if stock_rows:
        cursor.executemany("""
            INSERT INTO portfolio_signal (portfolio_id, signal_id, weight)
            VALUES (%s, %s, %s)
        """, signal_rows)

    conn.commit()



def ensure_signals_are_stored_in_db(conn: MySQLConnectionAbstract, signals: Iterable[SignalBase]):
    """
    Ensures all signals exist in the database by their name (signal.name),
    inserting any missing ones.

    Parameters:
    - conn: The MySQL connection object.
    - signals: An iterable of SignalBase objects.
    """
    cursor = conn.cursor()
    signals = list(signals)
    names = [s.name for s in signals]

    if not names:
        return

    # Step 1: Find existing signal names
    placeholders = ', '.join(['%s'] * len(names))
    cursor.execute(f"SELECT id FROM sda.signal WHERE id IN ({placeholders})", names)
    existing_names = {row[0] for row in cursor.fetchall()}

    # Step 2: Insert missing
    to_insert = list({signal.name for signal in signals if signal.name not in existing_names})

    if to_insert:
        cursor.executemany("INSERT INTO sda.signal (id) VALUES (%s)", to_insert)
        conn.commit()