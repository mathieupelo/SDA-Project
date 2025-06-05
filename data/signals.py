from data.stocks import Stock

class StockSignal:
    def __init__(self, stock: Stock, score: float):
        self.stock = stock
        self.score = score