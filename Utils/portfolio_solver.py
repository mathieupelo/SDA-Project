from data.portfolios import Portfolio
from data.stock_snapshot import StockSnapshot
from data.stocks import *
from data.utils import *
import yfinance as yf

class PortfolioSolver:
    def __init__(self, stocks: dict[Stock, StockSnapshot], penalty_factor, max_weight_threshold, risk_aversion):
        self._stocks = stocks
        self.penalty_factor = penalty_factor
        self.max_weight_threshold = max_weight_threshold
        self.risk_aversion = risk_aversion
        
        
    def solve(self) -> Portfolio:
        pass



def solve_most_goated_portfolio(
        conn: MySQLConnectionAbstract,
        alpha_values: dict[str, float],
        price_histories: dict[str, dict[date, float]],
        start_date: date,
        end_date: date,
        fetch_database: bool = True,
        penalty_factor=0.00001,
        max_weight_threshold=0.3,
        risk_aversion=0.3
) -> Portfolio:

    # Form stocks snapshot
    stock_snapshots: dict[Stock, StockSnapshot] = { }

    for ticker, alpha_score in alpha_values.items():
        stock = None
        price_history = None

        if fetch_database:
            stock = get_stock(conn, ticker)

        if not stock:
            stock = Stock(str(uuid.uuid1()), ticker, f"{ticker}_TEST")

        if fetch_database:
            price_histories.get(ticker)

        if not price_history:
            data = yf.download([ ticker ], start=start_date, end=end_date)
            price_history = data['Close'][ticker].to_dict()

        if not price_history:
            continue

        stock_snapshots[stock] = StockSnapshot(stock, alpha_score, price_history)

    solver = PortfolioSolver(stock_snapshots, penalty_factor, max_weight_threshold, risk_aversion)
    portfolio = solver.solve()
    return portfolio
