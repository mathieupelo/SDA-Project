from datetime import date
from mysql.connector.abstracts import MySQLConnectionAbstract


def get_stock_price_table(conn: MySQLConnectionAbstract, stock_id: str, start: date, end: date) \
        -> dict[date, float] | None:
    """
    Fetches historical stock prices for a given stock_id and date range.

    Parameters:
        stock_id (str): UUID of the stock in the stock table.
        conn: MySQL connection object.
        start (str or datetime.date): Start date (inclusive), e.g. '2020-01-01'.
        end (str or datetime.date): End date (inclusive), e.g. '2024-12-31'.

    Returns:
        dict[date, float] | None: A dict with {date: close_price} or None if no rows are found.
    """
    sql = """
          SELECT date, close_price
          FROM stock_price
          WHERE stock_id = %s
            AND date BETWEEN %s AND %s
          ORDER BY date
          """

    cursor = conn.cursor()
    cursor.execute(sql, (stock_id, start, end))
    rows = cursor.fetchall()

    if not rows:
        return None

    return {row[0]: float(row[1]) for row in rows}


def get_stock_price(conn: MySQLConnectionAbstract, stock_id: str, when: date | str) -> float or None:
    """
    Fetches the closing price of a stock for a specific date.

    Parameters:
        conn: MySQL connection object.
        stock_id (str): UUID of the stock in the 'stock' table.
        when (str or datetime.date): The date to retrieve the price for.

    Returns:
        float or None: The closing price if found, otherwise None.
    """
    sql = """
          SELECT close_price
          FROM stock_price
          WHERE stock_id = %s \
            AND date = %s
          LIMIT 1 \
          """

    cursor = conn.cursor()
    cursor.execute(sql, (stock_id, when))
    result = cursor.fetchone()

    return result[0] if result else None
