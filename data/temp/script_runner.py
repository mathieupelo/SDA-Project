from datetime import date
import pandas as pd
import time
from data.api import API

api = API('localhost')
tickers = ['AAPL', 'MSFT', 'SONY', 'AMZN', 'GOOG']
tickers = ['MSFT', 'SONY']

api.ensure_database_is_up_to_date()

history = api.get_price_history_for_tickers(tickers, date(2010, 1, 1), date(2024, 3, 1))
df = pd.DataFrame.from_dict(history, orient="index")
until = date(2014, 1, 1)

# --- DataFrame benchmark ---
start_df = time.perf_counter()
history_until_2014_df = df[df.index < until]
end_df = time.perf_counter()


# --- Dict benchmark ---
start_dict = time.perf_counter()
history_until_2014_dict = {day: val for day, val in history.items() if day < until}
end_dict = time.perf_counter()


# --- Results ---
print("\n--- Results ---")
print(f"DataFrame filter time: {(end_df - start_df)*1000:.4f} ms")
print(f"Dict filter time:      {(end_dict - start_dict)*1000:.4f} ms")
print(f"DataFrame rows: {len(history_until_2014_df)}")
print(f"Dict rows:      {len(history_until_2014_dict)}")







