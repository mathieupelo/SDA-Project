import mysql.connector
from mysql.connector import Error
from mysql.connector.abstracts import MySQLConnectionAbstract

USER = 'sda_admin'
PASSWORD = 'qwer1234'
DATABASE = 'sda'

def connect_to_database(host: str) -> MySQLConnectionAbstract | None:
    """
    Connects to the database at 'host' address.

    Parameters:
    - host: Address to connect to.

    Returns:
        The connection to the database.
    """
    try:
        conn = mysql.connector.connect(
            host=host,
            user=USER,
            password=PASSWORD,
            database=DATABASE
        )
        return conn

    except Error as e:
        print(f"Database connection failed: {e}")
        return None