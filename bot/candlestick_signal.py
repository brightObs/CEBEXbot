import logging
import json
import time
import websocket
import os
import signal
import sys
from threading import Thread, Event
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from queue import Queue
from config.settings import BYBIT_API_KEY, BYBIT_API_SECRET
from config.settings import SYMBOL, INTERVAL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from data.data_processing import extract_features_and_labels as extract_features
from models.hyperparameter_tuning import train_and_save_model  # Importing model from the hyperparameter tuning file
from threading import Lock


# Graceful exit on Ctrl+C
def signal_handler(sig, frame):
    logger.info("Exiting gracefully...")
    stop_event.set()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# Ensure log directory exists
log_dir = "data/logs"
os.makedirs(log_dir, exist_ok=True)

# Configure logging with UTF-8 encoding
logging.basicConfig(
    filename=os.path.join(log_dir, "app.log"),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8"
)

logger = logging.getLogger(__name__)

# WebSocket URL (Bybit API)
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"

# Thread-safe queue for storing candlestick data
candlestick_data = Queue(maxsize=50)

# Signal management
last_signal = None
last_signal_time = None
signal_duration = 300  # 5 minutes

# Telegram API URL
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

# WebSocket instance and stop flag
ws = None
stop_event = Event()

# Base URL for Bybit REST API
BYBIT_REST_URL = "https://api.bybit.com/v5/market/kline"

signal_lock = Lock()

# Define Nigeria's timezone (UTC+1)
NIGERIA_TZ = timezone(timedelta(hours=1))


def fetch_historical_data():
    """Fetches the last 50 candlestick data points from Bybit API."""
    url = f"{BYBIT_REST_URL}?category=linear&symbol={SYMBOL}&interval={INTERVAL}&limit=50"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure we catch any HTTP errors
        data = response.json()

        if "result" not in data or "list" not in data["result"]:
            logger.error("Invalid response structure from Bybit API")
            return None

        candles = data["result"]["list"]

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df.drop(columns=["turnover"], inplace=True)  # Remove unnecessary column

        # Convert numeric values
        numeric_columns = ["open", "high", "low", "close", "volume"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
        df.dropna(inplace=True)

        return df
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.error(f"Error fetching historical data: {e}", exc_info=True)
        return None


def on_message(ws, message):
    try:
        data = json.loads(message)
        logger.debug(f"Received data: {data}")

        if "topic" in data and "data" in data:
            candlestick = data["data"]

            if isinstance(candlestick, dict):  # Ensure correct format
                candlestick = [candlestick]

            for entry in candlestick:
                if "close" in entry:
                    if candlestick_data.full():
                        candlestick_data.get()  # Remove oldest entry
                    candlestick_data.put(entry)
                else:
                    logger.warning("Missing required fields in WebSocket data.")
        else:
            logger.warning("Unexpected WebSocket data format.")
    except Exception as e:
        logger.error(f"WebSocket message processing error: {e}", exc_info=True)


def on_error(ws, error):
    """Handles WebSocket errors."""
    logger.error(f"WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    """Handles WebSocket closure and reconnects if necessary."""
    logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
    if not stop_event.is_set():
        time.sleep(5)
        start_websocket()


def on_open(ws):
    """Handles WebSocket connection opening."""
    logger.info("WebSocket connection established.")
    ws.send(json.dumps({
        "op": "subscribe",
        "args": [f"kline.{INTERVAL}.{SYMBOL}"]
    }))


def start_websocket():
    """Starts the WebSocket connection."""
    global ws
    if ws is not None:
        ws.close()

    ws = websocket.WebSocketApp(
        BYBIT_WS_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = on_open
    thread = Thread(target=ws.run_forever, daemon=True)
    thread.start()


def preprocess_candlestick_data():
    """Converts queue data to pandas DataFrame."""
    candles = list(candlestick_data.queue)
    if len(candles) < 50:
        logger.warning("Not enough candlestick data available.")
        return None

    df = pd.DataFrame(candles)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

    return df


def generate_signal():
    """Generates a trading signal based on candlestick analysis and machine learning model."""
    global last_signal, last_signal_time

    if candlestick_data.qsize() < 50:
        logger.debug("Waiting for more candlestick data...")
        return None

    candles = preprocess_candlestick_data()
    if candles is None:
        return None

    current_time = time.time()
    if last_signal_time and (current_time - last_signal_time < signal_duration):
        logger.debug("Signal generation skipped due to lock period.")
        return None

    try:
        features, _ = extract_features(candles)  # Extracting features
    except Exception as e:
        logger.error(f"Feature extraction error: {e}", exc_info=True)
        return None

    if 'SMA_50' not in features.columns or 'RSI_14' not in features.columns:
        logger.error("Error: SMA_50 or RSI_14 is missing from feature extraction.")
        return None

    # Use machine learning model to generate the signal
    model = train_and_save_model()  # Assuming this function returns a trained model
    prediction = model.predict(features.iloc[-1].values.reshape(1, -1))  # Predict signal for the latest data point

    signal = None
    if prediction == 1:
        signal = "CALL"
    elif prediction == 0:
        signal = "PUT"

    if signal and signal != last_signal:
        last_signal_time, last_signal = current_time, signal

        # Get the latest candle's close time
        last_candle_time = candles.iloc[-1]['timestamp']

        if isinstance(last_candle_time, pd.Timestamp):
            last_candle_time = last_candle_time.timestamp()

        last_candle_datetime = datetime.fromtimestamp(last_candle_time, timezone.utc)

        # Determine the start of the next valid 5-minute candlestick window
        minutes = (last_candle_datetime.minute // 5) * 5 + 5
        next_candle_time = last_candle_datetime.replace(minute=minutes % 60, second=0, microsecond=0)

        if minutes >= 60:
            next_candle_time += timedelta(hours=1)

        # Calculate expiration time (5 minutes after next candle starts)
        end_time = next_candle_time + timedelta(minutes=5)

        # Convert times to Nigeria Timezone
        next_candle_time_nigeria = next_candle_time.astimezone(NIGERIA_TZ)
        end_time_nigeria = end_time.astimezone(NIGERIA_TZ)

        validity_period = f"{next_candle_time_nigeria.strftime('%H:%M')} - {end_time_nigeria.strftime('%H:%M')}"

        time_remaining = (next_candle_time - datetime.now(timezone.utc)).total_seconds()
        if time_remaining < 120:
            logger.debug("Signal generated too late, waiting for the next opportunity.")
            return None

        # Send signal with validity period to Telegram
        send_to_telegram(signal, validity_period)
        return signal
    return None


def send_to_telegram(signal, validity_period):
    """Sends the trading signal with validity period to Telegram."""
    message = f"📈 Trading Signal: {signal}\nValid from: {validity_period}"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}

    try:
        response = requests.post(TELEGRAM_API_URL, json=payload)
        response_data = response.json()

        if not response_data.get("ok"):
            logger.error(f"Telegram API Error: {response_data}")
        else:
            logger.info(f"Signal sent to Telegram: {message}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Telegram message: {e}", exc_info=True)


# Main execution
if __name__ == "__main__":
    fetch_historical_data()
    start_websocket()
    while not stop_event.is_set():
        generate_signal()
        time.sleep(5)
