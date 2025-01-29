import logging
import json
import time
import websocket
import os
import sys
import signal
from threading import Thread
import talib
import pandas as pd
from datetime import datetime, timezone, timedelta
from config.settings import SYMBOL, INTERVAL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from queue import Queue
from data.data_processing import extract_features_and_labels as extract_features

# Graceful exit on Ctrl+C
def signal_handler(sig, frame):
    print("Exiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Ensure the directory exists
log_dir = "data/logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, "app.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# WebSocket URL for USDT Perpetual contracts (Bybit API)
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"

# Global variables
candlestick_data = Queue(maxsize=100)
last_signal = None
last_signal_time = None
signal_duration = 300  # 5 minutes lock duration

# Telegram API
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

# WebSocket callback functions
def on_message(ws, message):
    global candlestick_data
    try:
        data = json.loads(message)
        logger.debug(f"Incoming data: {data}")

        if data.get("op") == "subscribe" and data.get("success"):
            logger.info("Subscribed to candlestick data successfully.")
        elif "topic" in data and data["topic"].startswith("kline"):
            candlesticks = data.get("data", [])
            if candlesticks:
                for candlestick in candlesticks:
                    if candlestick_data.qsize() >= 100:
                        candlestick_data.get()
                    candlestick_data.put(candlestick)
            else:
                logger.warning("No candlestick data found in the message.")
    except Exception as e:
        logger.error(f"Error processing WebSocket message: {e}", exc_info=True)

def on_error(ws, error):
    logger.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
    time.sleep(5)
    if close_status_code != 1000:  # Prevent reconnect for intentional closures
        start_websocket()

def on_open(ws):
    logger.info("WebSocket connection established.")
    ws.send(json.dumps({
        "op": "subscribe",
        "args": [f"kline.{INTERVAL}.{SYMBOL}"]
    }))

def start_websocket():
    ws = websocket.WebSocketApp(
        BYBIT_WS_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = on_open
    thread = Thread(target=ws.run_forever, daemon=True)
    thread.start()

# Preprocess incoming candlestick data
def preprocess_candlestick_data(candles):
    """
    Preprocess incoming candlestick data to ensure numeric values.
    """
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric
    df.dropna(subset=numeric_columns, inplace=True)  # Drop rows with NaN in any numeric column
    logger.debug(f"Preprocessed DataFrame: {df.head()}")  # Log the first few rows of the DataFrame
    return df

def get_candlestick_data():
    """
    Retrieve candlestick data from the global WebSocket queue.
    Returns a DataFrame of the most recent candlesticks or None if insufficient data.
    """
    global candlestick_data

    try:
        # Get all candlestick data currently in the queue
        candles = list(candlestick_data.queue)
        if len(candles) < 50:  # Ensure sufficient data for analysis
            logger.warning("Insufficient candlestick data available.")
            return None
        # Preprocess data to ensure compatibility
        return preprocess_candlestick_data(candles)
    except Exception as e:
        logger.error(f"Error retrieving candlestick data: {e}", exc_info=True)
        return None

# Candlestick pattern detection functions (Bullish, Bearish, and Doji)
def is_bullish_engulfing(candles):
    if len(candles) < 2:
        return False
    prev_candle, curr_candle = candles[-2], candles[-1]
    return (
        prev_candle['close'] < prev_candle['open'] and
        curr_candle['close'] > curr_candle['open'] and
        curr_candle['close'] > prev_candle['open'] and
        curr_candle['open'] < prev_candle['close']
    )

def is_bearish_engulfing(candles):
    if len(candles) < 2:
        return False
    prev_candle, curr_candle = candles[-2], candles[-1]
    return (
        prev_candle['close'] > prev_candle['open'] and
        curr_candle['close'] < curr_candle['open'] and
        curr_candle['close'] < prev_candle['open'] and
        curr_candle['open'] > prev_candle['close']
    )

def is_doji(candles):
    if len(candles) < 1:
        return False
    curr_candle = candles[-1]
    return abs(curr_candle['close'] - curr_candle['open']) <= 0.001

# Signal generation and logging
def generate_signal(candles):
    global last_signal, last_signal_time
    if len(candles) < 50:
        logger.debug("Not enough candlestick data for analysis.")
        return None

    current_time = time.time()
    if last_signal_time and (current_time - last_signal_time < signal_duration):
        logger.debug("Signal generation skipped due to 5-minute lock.")
        return None

    features, _ = extract_features(candles)
    latest_features = features.iloc[-1] if isinstance(features, pd.DataFrame) else None
    if latest_features is None:
        logger.error("Invalid features for signal generation.")
        return None

    sma_50, rsi_14 = latest_features['SMA_50'], latest_features['RSI_14']
    close_price = candles.iloc[-1]['close']
    signal = None

    if close_price > sma_50 and rsi_14 > 50:
        signal = "CALL"
    elif close_price < sma_50 and rsi_14 < 50:
        signal = "PUT"

    if is_bullish_engulfing(candles.to_dict('records')):
        signal = "CALL"
    elif is_bearish_engulfing(candles.to_dict('records')):
        signal = "PUT"
    elif is_doji(candles.to_dict('records')):
        signal = None

    if signal and signal != last_signal:
        last_signal_time, last_signal = current_time, signal
        logger.info(f"Signal generated: {signal}")
        return signal
    return None

def log_signal(signal, valid_start, valid_end):
    logger.info(f"Generated Signal: {signal} valid from {valid_start} to {valid_end}")

# Main execution
if __name__ == "__main__":
    start_websocket()

    while True:
        try:
            candles = list(candlestick_data.queue)
            if candles:
                df = preprocess_candlestick_data(candles)
                signal = generate_signal(df)
                if signal:
                    now = datetime.now(timezone.utc)
                    valid_start = now + timedelta(minutes=5 - now.minute % 5)
                    valid_end = valid_start + timedelta(minutes=5)
                    log_signal(signal, valid_start.strftime("%H:%M"), valid_end.strftime("%H:%M"))
            else:
                logger.debug("No candlestick data available.")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Error during main loop: {e}", exc_info=True)
