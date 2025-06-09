from datetime import date
from data.api import API

#from signals.signal_pipeline import run_signal_pipeline
#run_signal_pipeline('192.168.0.165')

api = API('192.168.0.165')
tickers = [ 'MSFT' ]
signals = [ 'RSI', 'MACD', 'SMA' ]
first_day = date(2020, 1, 1)
last_day = date(2024, 1, 1)

signal_scores = api.get_signal_scores_table_for_tickers(tickers, signals, first_day, last_day)
