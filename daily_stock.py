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
    stock_data = yf.download(ticker, period="6mo")
    if stock_data.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    return stock_data
    
def prepare_data(stock_data: pd.DataFrame):
    """
    Prepares stock data for model training by creating labels.
    The closing price is shifted to predict the next day's price.
    Parameters:
    ----------
    stock_data : pandas.DataFrame
        The stock data containing 'Close' prices.
    Returns:
    -------
    X_train, X_test, y_train, y_test : tuple
        Split data for training and testing.
    """
    stock_data['Next_Day_Price'] = stock_data['Close'].shift(-1)
    stock_data = stock_data.dropna()
    X = stock_data[['Close']].values  
    y = stock_data['Next_Day_Price'].values  
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

def plot_results(stock_data: pd.DataFrame, y_test: np.ndarray, predictions: np.ndarray, ticker: str):
    """
    Plot of the actual vs predicted stock prices.
    Parameters:
    ----------
    stock_data : pandas.DataFrame
        DataFrame containing stock data.
    y_test : np.ndarray
        Actual stock prices (test data).
    predictions : np.ndarray
        Predicted stock prices.
    ticker : str
        Stock ticker symbol for labeling.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data.index[-len(y_test):], y_test, label='Actual Prices')
    plt.plot(stock_data.index[-len(predictions):], predictions, label='Predicted Prices', linestyle='dashed')
    plt.title(f'{ticker} Predicted stock price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def main():
    """
    Main function to run the stock prediction model.
    """
    ticker = input("Enter the stock ticker symbol (e.g., AAPL, TSLA, GOOG): ").upper()
    stock_data = get_stock_data(ticker)
    X_train, X_test, y_train, y_test = prepare_data(stock_data)
    model = train_model(X_train, y_train)
    predictions = predict_stock_price(model, X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error for {ticker}: {mse:.2f}")
    plot_results(stock_data, y_test, predictions, ticker)

if __name__ == "__main__":
    main()
