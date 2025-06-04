import mysql.connector
import yfinance as yf
from datetime import date as datetime

def fetch_prices(host):
    conn = mysql.connector.connect(
        host=host,
        user='sda_admin',
        password='qwer1234',
        database='sda'
    )

    cursor = conn.cursor()
    cursor.execute("SELECT id, ticker FROM stock")
    stocks = cursor.fetchall()
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
