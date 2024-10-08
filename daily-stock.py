import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def get_stock_data(ticker: str):
    """
    Search for stock data for a given ticker.
    Parameters:
    ----------
    ticker : str
        ticker: stock ticker symbol (e.g., 'TSLA', 'GOOG').
        str: indicates ticker should be type string.
    Returns:
    -------
    stock_data : pandas.DataFrame
        DataFrame with stock data.
    """
 stock_data = yf.download(ticker, period="1y")
    if stock_data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    return stock_data

def prepare_data(stock_data: pd.DataFrame):
    """
    Prepares stock data for model training. The 'Close' price is shifted to predict the next day's price.
    Parameters:
    ----------
    stock_data : pandas.DataFrame
        DataFrame containing stock data with 'Close' prices.
    Returns:
    -------
    X_train, X_test, y_train, y_test : tuple
        Training and testing data.
    """
    stock_data['Prediction'] = stock_data['Close'].shift(-1)

    X = stock_data[['Close']].values
    y = stock_data['Prediction'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test
