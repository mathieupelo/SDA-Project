
import pandas as pd
import yfinance as yf
import talib as ta
import pandas as pd
import numpy as np

def calculate_rsi_signal(data, tickers, date, period=14):
    """
    Calculate the RSI for each stock in the portfolio and return a single signal score for each company.
    
    Parameters:
    - data: A DataFrame containing the 'Close' prices for each stock (tickers as columns).
    - period: The lookback period for the RSI calculation (default is 14).
    
    Returns:
    - A list of signal scores for each stock, corresponding to the tickers.
    """
    # Convert the date to a pandas datetime object for comparison
    date = pd.to_datetime(date)

    # Initialize an empty list to hold the signal scores
    signal_scores = []

    # Loop through each stock (ticker) in the data DataFrame
    for ticker in tickers:
        # Get the closing prices for the ticker
        close_prices = data['Close'][ticker]
        
        # Calculate RSI using TA-Lib (you can adjust the time period if needed)
        rsi = ta.RSI(close_prices, timeperiod=period)
        
        # Check if the date is within the available data range
        if date in rsi.index:
            # Get the RSI value for the specific date
            signal_scores.append([ticker, rsi.loc[date]])
        else:
            # Handle the case when the date is not available in the data range
            signal_scores.append([ticker, np.nan])
            
    return signal_scores



def calculate_macd_signal(data, tickers, date):
    """
    Calculate the MACD signal for each stock in the portfolio on a specific date and return a signal score for each company.
    
    Parameters:
    - data: A DataFrame containing the 'Close' prices for each stock (tickers as columns).
    - tickers: List of stock tickers to calculate MACD for.
    - date: The date (as a string in 'YYYY-MM-DD' format) to get the MACD for.
    
    Returns:
    - A list of signal scores for each stock, corresponding to the tickers.
    """
    # Convert the date to a pandas datetime object for comparison
    date = pd.to_datetime(date)
    
    # Initialize an empty list to hold the signal scores
    signal_scores = []

    # Loop through each stock (ticker) in the data DataFrame
    for ticker in tickers:
        # Get the closing prices for the ticker
        close_prices = data['Close'][ticker]
        
        # Calculate MACD and Signal Line using TA-Lib
        macd, macdsignal, _ = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # Check if the date is within the available data range
        if date in macd.index:
            # Get the MACD Histogram value for the specific date
            macd_histogram = macd.loc[date] - macdsignal.loc[date]  # MACD Histogram = MACD - Signal Line
            signal_scores.append([ticker, macd_histogram])
        else:
            # Handle the case when the date is not available in the data range
            signal_scores.append([ticker, np.nan])
    
    return signal_scores


def calculate_sma_signal(data, tickers, date):
    """
    Calculate the SMA crossover signal for each stock in the portfolio on a specific date
    and return a signal score for each company.
    
    Parameters:
    - data: A DataFrame containing the 'Close' prices for each stock (tickers as columns).
    - tickers: List of stock tickers to calculate SMA crossover for.
    - date: The date (as a string in 'YYYY-MM-DD' format) to get the SMA crossover for.
    
    Returns:
    - A list of signal scores for each stock, corresponding to the tickers.
    """
    # Convert the date to a pandas datetime object for comparison
    date = pd.to_datetime(date)
    
    # Initialize an empty list to hold the signal scores
    signal_scores = []

    # Loop through each stock (ticker) in the data DataFrame
    for ticker in tickers:
        # Get the closing prices for the ticker
        close_prices = data['Close'][ticker]
        
        # Calculate the 50-period and 200-period Simple Moving Averages (SMA)
        sma_50 = ta.SMA(close_prices, timeperiod=50)
        sma_200 = ta.SMA(close_prices, timeperiod=200)

        # Check if the date is within the available data range
        if date in sma_50.index and date in sma_200.index:
            # Calculate the difference between the 50-period and 200-period SMAs
            sma_diff = sma_50.loc[date] - sma_200.loc[date]
            
            # Generate the score based on the SMA difference
            # Positive score for Golden Cross, negative score for Death Cross, larger the difference, larger the score
            signal_scores.append([ticker, sma_diff])  # Score is just the difference between SMAs
        else:
            # Handle the case when the date is not available in the data range
            signal_scores.append([ticker, np.nan])
    
    return signal_scores