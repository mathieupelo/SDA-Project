from datetime import date

from rich.jupyter import display

from data.api import API

api = API('localhost')
tickers = ['AAPL', 'MSFT', 'SONY', 'AMZN', 'GOOG']
tickers = ['MSFT', 'SONY']

api.ensure_database_is_up_to_date()

#history = api.get_price_history_for_tickers(tickers, date(2010, 1, 1), date(2024, 3, 1))
#print(history)