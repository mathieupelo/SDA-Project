import mysql.connector
import yfinance as yf
from datetime import date as datetime
from data.utils import connect_to_database
from data.stocks import get_stocks

def fetch_prices(host: str):
    conn = connect_to_database(host)
    stocks = get_stocks(conn)

    sql = "INSERT IGNORE INTO stock_price (stock_id, date, close_price) VALUES (%s, %s, %s)"
    for stock in stocks:
        stock_id = stock[0]
        stock_symbol = stock[1]
        data = yf.download(stock_symbol, start='1800-01-01', end=datetime.today())
        close = data['Close'][stock_symbol]

        for date, price in close.items():
            with conn.cursor() as cursor:
                cursor.execute(sql, (stock_id, date, price))
                conn.commit()
