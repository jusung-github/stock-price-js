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
    
def train_model(X_train: np.ndarray, y_train: np.ndarray):
    """
    Train a linear regression model on the provided data.
    Parameters:
    ----------
    X_train : np.ndarray
        Training features (closing prices).
    y_train : np.ndarray
        Training targets (next day's price).
    Returns:
    -------
    model : sklearn.linear_model.LinearRegression
        Trained linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_stock_price(model, X_test: np.ndarray):
    """
    Use trained model to predict stock price.
    Parameters:
    ----------
    model : sklearn.linear_model.LinearRegression
        Trained model.
    X_test : np.ndarray
        Testing features (closing prices).
    Returns:
    -------
    predictions : np.ndarray
        Predicted prices.
    """
    predictions = model.predict(X_test)
    return predictions
