import uuid
from typing import List
from mysql.connector.abstracts import MySQLConnectionAbstract
from data.solver_config import SolverConfig
from datetime import date, timedelta


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
    def __init__(self, p_id: str, creation_date: date, stocks: dict[str, StockMetadata], config: SolverConfig):
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
    def stocks(self) -> dict[str, StockMetadata]:
        return self._stocks

    @property
    def config(self) -> SolverConfig:
        return self._config

    def get_weight_table(self) -> dict[str, float]:
        return { ticker: metadata.weight for ticker, metadata in self._stocks.items()}



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
    cursor.execute("""
                   SELECT date, risk_aversion, max_weight
                   FROM portfolio
                   WHERE id = %s
                   """, (portfolio_id,))
    row = cursor.fetchone()

    if not row:
        return None

    creation_date, risk_aversion, max_weight = row

    # Get associated stocks from portfolio_stock
    cursor.execute("""
                   SELECT s.id, s.name, s.ticker, ps.weight, ps.alpha_score
                   FROM portfolio_stock ps
                            JOIN stock s ON s.id = ps.stock_id
                   WHERE ps.portfolio_id = %s
                   """, (portfolio_id,))
    rows = cursor.fetchall()
    stocks: dict[str, Portfolio.StockMetadata] = { }

    for stock_id, name, ticker, weight, alpha_score in rows:
        metadata = Portfolio.StockMetadata(weight, alpha_score)
        stocks[name] = metadata

    config = SolverConfig(
        risk_aversion=risk_aversion,
        max_weight_threshold=max_weight
    )

    return Portfolio(p_id=portfolio_id, creation_date=creation_date, stocks=stocks, config=config)


def cache_portfolio_data(
        conn: MySQLConnectionAbstract,
        portfolio: Portfolio,
        signals: dict[str, float],
        yearly_return: float,
        ticker_map: dict[str, str]):
    """
    Caches a Portfolio into the database.

    Parameters:
    - conn: The MySQL connection object.
    - portfolio: The Portfolio object to insert.
    - signals: Mapping from signal name to its weight.
    - yearly_return: Annualized return to store.
    - ticker_to_id: Mapping from stock ticker to its corresponding database ID.
    """
    cursor = conn.cursor()

    # Cache to 'portfolio' database table.
    cursor.execute("""
        INSERT INTO portfolio (id, date, risk_aversion, max_weight, yearly_return)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        portfolio.id,
        portfolio.creation_date,
        portfolio.config.risk_aversion,
        portfolio.config.max_weight_threshold,
        yearly_return
    ))

    stock_rows = []
    for ticker, metadata in portfolio.stocks.items():
        stock_id = ticker_map.get(ticker)
        if stock_id is None:
            raise ValueError(f"Ticker '{ticker}' not found in ticker_to_id mapping.")
        stock_rows.append((portfolio.id, stock_id, metadata.weight, metadata.alpha_score))

    if stock_rows:
        cursor.executemany("""
            INSERT INTO portfolio_stock (portfolio_id, stock_id, weight, alpha_score)
            VALUES (%s, %s, %s, %s)
        """, stock_rows)

    signal_rows = [
        (portfolio.id, signal, weight)
        for signal, weight in signals.items()
    ]

    if signal_rows:
        cursor.executemany("""
            INSERT INTO portfolio_signal (portfolio_id, signal_id, weight)
            VALUES (%s, %s, %s)
        """, signal_rows)

    conn.commit()



def cache_backtest_result(
        conn: MySQLConnectionAbstract,
        portfolios: List[Portfolio],
        start_date: date,
        end_date: date,
        execution_time: timedelta):
    """
    Caches the result of a backtest into the database.

    Parameters:
    - conn: The MySQL connection object.
    - portfolios: The portfolios computed by the backtest.
    """
    today = date.today()
    cursor = conn.cursor()
    backtest_id = str(uuid.uuid1())

    # Step 1: Insert into backtest
    cursor.execute("""
        INSERT INTO backtest (id, start_date, end_date, execution_date, execution_time)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        backtest_id,
        start_date,
        end_date,
        today,
        execution_time
    ))

    # Step 2: Insert into backtest_portfolio
    cursor.executemany("""
        INSERT INTO backtest_portfolio (backtest_id, portfolio_id)
        VALUES (%s, %s)
    """, [
        (backtest_id, portfolio.id)
        for portfolio in portfolios
    ])

    conn.commit()