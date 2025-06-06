import uuid
import yfinance as yf
from typing import List
from data.database import connect_to_database

def fetch_stocks(ticker_list: List[str], host: str):

    rows = []
    for stock_symbol in ticker_list:
        stock = yf.Ticker(stock_symbol)
        stock_id = str(uuid.uuid4())
        stock_name = stock.info.get("longName")

        if not stock_name:
            print(f"⚠️  Skipping {stock_symbol} - NOT FOUND")
            continue

        rows.append((stock_id, stock_symbol, stock_name))

    conn = connect_to_database(host)
    with conn.cursor() as cursor:
        sql = "INSERT IGNORE INTO stock (id, ticker, name) VALUES (%s, %s, %s)"
        cursor.executemany(sql, rows)
        conn.commit()