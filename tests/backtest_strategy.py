from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import json
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

    # Validate historical_data
    if not isinstance(historical_data, list) or len(historical_data) == 0:
        logger.error("Invalid historical data: It must be a non-empty list.")
        return

    # Ensure 'close' prices exist
    try:
        prices = np.array([float(candle['close']) for candle in historical_data])
    except (TypeError, KeyError, ValueError) as e:
        logger.error(f"Error extracting 'close' prices from historical data: {e}")
        return

    # Ensure predictions match price data length
    if len(predictions) != len(prices):
        logger.error(f"Mismatch: {len(predictions)} predictions vs {len(prices)} prices.")
        return

    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Profitability Analysis
    initial_balance = 10000
    balance = initial_balance
    profit_trades = 0
    loss_trades = 0
    risk_per_trade = 0.02  # Risk 2% per trade

    for i, prediction in enumerate(predictions[:-1]):
        investment = balance * risk_per_trade  # Risk-based trading
        if prediction == 1:  # CALL
            price_change = (prices[i + 1] - prices[i]) / prices[i]
        else:  # PUT
            price_change = (prices[i] - prices[i + 1]) / prices[i]

        profit = price_change * investment
        if profit > 0:
            profit_trades += 1
        else:
            loss_trades += 1

        balance += profit  # Adjust balance with profit/loss

    win_rate = profit_trades / (profit_trades + loss_trades) if (profit_trades + loss_trades) > 0 else 0
    max_drawdown = calculate_max_drawdown(prices)

    logger.info(f"Final Balance: ${balance:.2f}")
    logger.info(f"Win Rate: {win_rate:.2%}")
    logger.info(f"Max Drawdown: {max_drawdown:.2%}")

    # Visualization
    visualize_backtest(historical_data, predictions, balance)


def calculate_max_drawdown(prices):
    """
    Calculate the maximum drawdown of the price series.
    """
    peak = prices[0]  # Start with the first price
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

    if len(predictions) != len(prices):
        logger.error("Mismatch in predictions and historical data length.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(prices, label="Price", color="blue")

    call_indices = [i for i, p in enumerate(predictions) if p == 1]
    put_indices = [i for i, p in enumerate(predictions) if p == 0]

    plt.scatter(call_indices, [prices[i] for i in call_indices], color="green", label="CALL", marker="^")
    plt.scatter(put_indices, [prices[i] for i in put_indices], color="red", label="PUT", marker="v")

    plt.title("Backtest Visualization")
    plt.xlabel("Candlestick Index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # Example data for testing
    predictions = [1, 0, 1, 1, 0]
    labels = [1, 0, 1, 1, 0]
    historical_data = [
        {'close': '100.00'}, {'close': '102.00'}, {'close': '101.00'},
        {'close': '103.00'}, {'close': '104.00'}
    ]

    evaluate_strategy(predictions, labels, historical_data)
