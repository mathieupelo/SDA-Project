import datetime
from typing import List, Tuple, Iterable
from mysql.connector.abstracts import MySQLConnectionAbstract
from data.utils.database import connect_to_database
from signals.signal_base import SignalBase


def get_enabled_signals(conn: MySQLConnectionAbstract) -> set[str]:
    """
       Fetches all signal IDs from the sda.signal table where enabled is TRUE.

       Parameters:
       - conn: A MySQLConnectionAbstract connection object.

       Returns:
       - A list of signal IDs (str) that are enabled.
       """
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM sda.signal WHERE enabled = TRUE")
    rows = cursor.fetchall()
    return set([row[0] for row in rows])


def store_signal_scores_for_ticker(
    conn: MySQLConnectionAbstract,
    ticker: str,
    scores: List[Tuple[str, datetime.date, float]]
) -> None:
    """
    Stores signal scores for a single ticker.

    Parameters:
        conn: MySQL connection.
        ticker: The stock ticker associated with all scores.
        scores: List of (signal_id, date, score)
    """
    if not scores:
        return

    cursor = conn.cursor()

    # Step 1: Resolve stock_id from ticker
    cursor.execute("SELECT id FROM sda.stock WHERE ticker = %s", (ticker,))
    row = cursor.fetchone()
    if row is None:
        print(f"[WARN] Ticker '{ticker}' not found in stock table.")
        return
    stock_id = row[0]

    # Step 2: Resolve signal versions
    signal_ids = list({signal_id for signal_id, _, _ in scores})
    placeholders = ', '.join(['%s'] * len(signal_ids))
    cursor.execute(f"""
        SELECT id, version FROM sda.signal
        WHERE id IN ({placeholders})
    """, signal_ids)
    signal_versions = {row[0]: row[1] for row in cursor.fetchall()}

    # Step 3: Prepare insert values
    insert_values = []
    for signal_id, day, score in scores:
        version = signal_versions.get(signal_id)
        if version is None:
            print(f"[WARN] Missing version for signal '{signal_id}'")
            continue
        insert_values.append((stock_id, signal_id, version, day, score))

    # Step 4: Insert into DB
    if insert_values:
        cursor.executemany("""
            INSERT INTO sda.signal_score
                (stock_id, signal_id, signal_version, date, score)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE score = VALUES(score), updated_date = CURRENT_TIMESTAMP
        """, insert_values)
        conn.commit()



def get_missing_signal_scores_for_ticker(host: str, ticker: str) -> List[Tuple[str, datetime.date]]:
    """
    Returns a list of (signal_id, latest_known_date) for the given stock ticker,
    where signal scores are missing up to the most recent valid trading day
    (i.e., days with close_price IS NOT NULL).
    """
    conn = connect_to_database(host)
    cursor = conn.cursor()

    cursor.execute("""
        WITH stock_data AS (
            SELECT
                id AS stock_id,
                (
                    SELECT MIN(date)
                    FROM sda.stock_price
                    WHERE stock_id = st.id AND close_price IS NOT NULL
                ) AS min_price_date,
                (
                    SELECT MAX(date)
                    FROM sda.stock_price
                    WHERE stock_id = st.id AND close_price IS NOT NULL
                          AND date <= CURDATE() - INTERVAL 1 DAY
                ) AS last_valid_trading_day
            FROM sda.stock st
            WHERE st.ticker = %s
        )
        
        SELECT
            s.id AS signal_id,
            COALESCE(MAX(ss.date), ANY_VALUE(sd.min_price_date)) AS latest_date,
            MAX(ss.date >= sd.last_valid_trading_day) AS has_recent
        FROM
            sda.signal s
        CROSS JOIN stock_data sd
        LEFT JOIN sda.signal_score ss
            ON ss.signal_id = s.id AND ss.stock_id = sd.stock_id
        GROUP BY s.id
        HAVING has_recent = 0 OR has_recent IS NULL;
    """, (ticker,))

    rows = cursor.fetchall()
    return [(signal_id, latest_date) for signal_id, latest_date, _ in rows]


def ensure_signals_are_stored_in_db(conn: MySQLConnectionAbstract, signals: Iterable[SignalBase]):
    """
    Ensures all signals exist in the database by their id, inserting any missing ones with both id and name.

    Parameters:
    - conn: The MySQL connection object.
    - signals: An iterable of SignalBase objects.
    """
    cursor = conn.cursor()
    signals = list(signals)
    ids = [s.id for s in signals]

    if not ids:
        return

    # Step 1: Find existing signal IDs
    placeholders = ', '.join(['%s'] * len(ids))
    cursor.execute(f"SELECT id FROM sda.signal WHERE id IN ({placeholders})", ids)
    existing_ids = {row[0] for row in cursor.fetchall()}

    # Step 2: Insert missing signals (both id and name)
    to_insert = [(s.id, s.name) for s in signals if s.id not in existing_ids]

    if to_insert:
        cursor.executemany("INSERT INTO sda.signal (id, name) VALUES (%s, %s)", to_insert)
        conn.commit()
