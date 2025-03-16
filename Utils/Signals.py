import talib as ta
import pandas as pd

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