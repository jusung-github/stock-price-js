import pytest
from daily_stock import get_stock_data, prepare_data, train_model, predict_stock_price

def test_get_stock_data_valid_ticker():
    stock_data = get_stock_data("AAPL")
    assert not stock_data.empty, "Expected non-empty DataFrame for valid ticker."

def test_get_stock_data_invalid_ticker():
    with pytest.raises(ValueError):
        get_stock_data("INVALIDTICKER")

def test_prepare_data():
    stock_data = get_stock_data("AAPL")
    X_train, X_test, y_train, y_test = prepare_data(stock_data)
    assert len(X_train) > 0 and len(y_train) > 0, "Expected non-empty training data."
    assert len(X_test) > 0 and len(y_test) > 0, "Expected non-empty testing data."

def test_train_model():
    stock_data = get_stock_data("AAPL")
    X_train, X_test, y_train, y_test = prepare_data(stock_data)
    model = train_model(X_train, y_train)
    assert model is not None, "Expected trained model instance."