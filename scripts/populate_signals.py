# Enhanced Backtesting Architecture for Signal Testing
import sys
import os

# Add project root (the parent of 'scripts') to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from signals.macd import MACDSignal
from signals.rsi import RSISignal
from signals.sma import SMASignal

from Utils.backtesting import *
from Utils.df_helper import *
from data.api import API

from datetime import date
from data.api import API


from signals.signal_pipeline import run_signal_pipeline


def populate_signal_scores():
    run_signal_pipeline('192.168.0.165')



if __name__ == "__main__":
    populate_signal_scores()