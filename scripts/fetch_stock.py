import mysql.connector
import uuid
import yfinance as yf

def fetch_stocks(ticker_list, host):
    """
    Populates the MySQL database with stock data.

    Parameters:
    - ticker_list: A list of ticker symbols.
    - host: Database host string.
    """
    rows = []
    conn = mysql.connector.connect(
        host=host,
        user='sda_admin',
        password='qwer1234',
        database='sda'
    )

    for stock_symbol in ticker_list:
        stock = yf.Ticker(stock_symbol)
        stock_id = str(uuid.uuid4())
        stock_name = stock.info.get("longName")

        if not stock_name:
            print(f"⚠️  Skipping {stock_symbol} - NOT FOUND")
            continue

        rows.append((stock_id, stock_symbol, stock_name))

    with conn.cursor() as cursor:
        sql = "INSERT IGNORE INTO stock (id, ticker, name) VALUES (%s, %s, %s)"
        cursor.executemany(sql, rows)
        conn.commit()
