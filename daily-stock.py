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
 stock_data = yf.download(ticker, period="12mo")
    if stock_data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    return stock_data
