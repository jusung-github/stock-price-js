import pytest
from daily_stock import get_stock_data, prepare_data, train_model, predict_stock_price

def test_get_stock_data_valid_ticker():
    """
    Test that get_stock_data returns a non-empty DataFrame for a valid ticker symbol.
    """
    stock_data = get_stock_data("AAPL")
    assert not stock_data.empty, "Expected non-empty DataFrame for valid ticker."

def test_get_stock_data_invalid_ticker():
    """
    Test that get_stock_data raises a ValueError when given an invalid ticker symbol.
    """
    with pytest.raises(ValueError):
        get_stock_data("INVALIDTICKER")

def test_prepare_data():
    """
    Test that prepare_data correctly splits stock data into training and test sets.
    """
    stock_data = get_stock_data("AAPL")
    X_train, X_test, y_train, y_test = prepare_data(stock_data)
    assert len(X_train) > 0 and len(y_train) > 0, "Expected non-empty training data."
    assert len(X_test) > 0 and len(y_test) > 0, "Expected non-empty testing data."

def test_train_model():
    """
    Test that train_model returns a trained model when given valid training data.
    """
    stock_data = get_stock_data("AAPL")
    X_train, X_test, y_train, y_test = prepare_data(stock_data)
    model = train_model(X_train, y_train)
    assert model is not None, "Expected trained model instance."

def test_get_stock_data_no_data():
    """
    Test get_stock_data raises a ValueError when no data is available for the ticker.
    """
    with pytest.raises(ValueError):
        get_stock_data("NODATA") 

def test_prepare_data_insufficient_data():
    """
    Test prepare_data raises an error when there are too few data points.
    """
    # Create a DataFrame with only a few rows
    stock_data = pd.DataFrame({"Close": [150, 152]})
    with pytest.raises(ValueError, match="Not enough data"):
        prepare_data(stock_data)

