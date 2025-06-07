from datetime import date
import pandas as pd
import time
from data.api import API
from data.stocks import get_stocks
from data.utils.database import connect_to_database

api = API('192.168.0.165')
conn = connect_to_database('192.168.0.165')
stocks = get_stocks(conn)
tickers = list({ stock.ticker for stock in stocks })

api.ensure_database_is_up_to_date()

history = api.get_price_history_for_tickers(tickers, date(2004, 1, 1), date(2024, 3, 1))
df = pd.DataFrame.from_dict(history, orient="index")
since = date(2023, 1, 1)
until = date(2024, 1, 1)

# --- DataFrame benchmark ---
start_df = time.perf_counter()
history_until_2014_df = df.loc[since : until]
end_df = time.perf_counter()


# --- Dict benchmark ---
start_dict = time.perf_counter()
history_until_2014_dict = {day: val for day, val in history.items() if since <= day <= until}
end_dict = time.perf_counter()


# --- Results ---
print("\n--- Results ---")
print(f"DataFrame filter time: {(end_df - start_df)*1000:.4f} ms")
print(f"Dict filter time:      {(end_dict - start_dict)*1000:.4f} ms")
print(f"DataFrame rows: {len(history_until_2014_df)}")
print(f"Dict rows:      {len(history_until_2014_dict)}")







