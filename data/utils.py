import mysql.connector

USER = 'sda_admin'
PASSWORD = 'qwer1234'
DATABASE = 'sda'

def connect_to_database(host):
    """
    Connects to the database at 'host' address.

    Parameters:
    - host: Address to connect to.

    Returns:
        The connection to the database.
    """
    conn = mysql.connector.connect(
        host=host,
        user=USER,
        password=PASSWORD,
        database=DATABASE
    )

    return conn