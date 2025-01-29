from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

# Initialize the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def evaluate_strategy(predictions, labels, historical_data):
    """
    Evaluate the strategy with financial metrics and visualizations.
    """
    # Ensure predictions and labels are numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Check if historical_data is valid and contains 'close' prices
    if not historical_data or historical_data == ...:
        logger.error("Invalid historical data: Contains ellipsis or is empty.")
        return

    # Ensure historical_data is a list of dictionaries with 'close' key
    try:
        prices = np.array([float(candle['close']) for candle in historical_data])
    except (TypeError, KeyError) as e:
        logger.error(f"Error extracting 'close' prices from historical data: {e}")
        return

    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Profitability Analysis
    initial_balance = 10000  # Example starting balance
    balance = initial_balance
    profit_trades = 0
    loss_trades = 0

    for i, prediction in enumerate(predictions[:-1]):
        if prediction == 1:  # CALL
            price_change = (prices[i + 1] - prices[i]) / prices[i]
        else:  # PUT
            price_change = (prices[i] - prices[i + 1]) / prices[i]

        profit = price_change * balance  # Example: invest full balance
        if profit > 0:
            profit_trades += 1
        else:
            loss_trades += 1

        balance += profit

    win_rate = profit_trades / (profit_trades + loss_trades) if (profit_trades + loss_trades) > 0 else 0
    logger.info(f"Final Balance: ${balance:.2f}")
    logger.info(f"Win Rate: {win_rate:.2%}")
    logger.info(f"Max Drawdown: {calculate_max_drawdown(prices):.2%}")

    # Visualization
    visualize_backtest(historical_data, predictions, balance)


def calculate_max_drawdown(prices):
    """
    Calculate the maximum drawdown of the price series.
    """
    peak = -np.inf
    max_drawdown = 0
    for price in prices:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def visualize_backtest(historical_data, predictions, balance):
    """
    Plot the historical prices and model predictions.
    """
    prices = [float(candle['close']) for candle in historical_data]

    # Ensure the predictions and prices are aligned in length
    assert len(predictions) == len(prices), "The length of predictions and prices do not match!"

    plt.figure(figsize=(12, 6))

    # Plotting the price data
    plt.plot(prices, label="Price", color="blue")

    # Plotting CALL predictions
    call_predictions = [prices[i] for i, p in enumerate(predictions) if p == 1]
    call_indices = [i for i, p in enumerate(predictions) if p == 1]
    plt.scatter(call_indices, call_predictions, color="green", label="CALL Predictions", marker="^")

    # Plotting PUT predictions
    put_predictions = [prices[i] for i, p in enumerate(predictions) if p == 0]
    put_indices = [i for i, p in enumerate(predictions) if p == 0]
    plt.scatter(put_indices, put_predictions, color="red", label="PUT Predictions", marker="v")

    # Finalizing the plot
    plt.title("Backtest Visualization")
    plt.xlabel("Candlestick Index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Example usage: Replace with your actual data
    predictions = [1, 0, 1, 1, 0]  # Replace with your model's predictions
    labels = [1, 0, 1, 1, 0]  # Replace with the actual labels
    historical_data = [
        {'close': '100.00'}, {'close': '102.00'}, {'close': '101.00'},
        {'close': '103.00'}, {'close': '104.00'}
    ]  # Replace with actual historical data

    evaluate_strategy(predictions, labels, historical_data)
