import sys
import os
import logging
import requests
import json
import threading
import websocket
from time import sleep
from datetime import datetime, timezone, timedelta
from queue import Queue, Empty
from telegram import Bot
from telegram.ext import Application, CallbackContext, JobQueue

# Add project root to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import signal logic
from bot.candlestick_signal import get_candlestick_data, generate_signal
from config.settings import TELEGRAM_BOT_TOKEN

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), '../config/logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "scheduler.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
SIGNAL_DURATION = 300  # 5 minutes in seconds
WS_URL = "wss://stream.bybit.com/v5/public/linear"
SYMBOL = "BTCUSDT"  # Modify symbol as needed
INTERVAL = "5"  # 5-minute candlestick interval

# Thread-safe queue for candlestick data
candlestick_queue = Queue(maxsize=100)

# Lock for thread safety
data_lock = threading.Lock()


def validate_telegram_token():
    """Validates the Telegram bot token."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe"
        response = requests.get(url)
        response_data = response.json()
        if response_data.get("ok"):
            logger.info("Telegram bot token is valid.")
        else:
            raise ValueError("Invalid Telegram bot token.")
    except Exception as e:
        logger.error(f"Error validating Telegram bot token: {e}")
        raise ValueError("Error validating Telegram bot token.")


async def send_signal(context: CallbackContext):
    """Generates and sends signals to Telegram subscribers."""
    try:
        candles = list(candlestick_queue.queue)
        if len(candles) < 50:  # Ensure enough data for signal analysis
            logger.warning("Insufficient candlestick data for signal generation.")
            return

        signal = generate_signal(candles)
        if signal:
            now = datetime.utcnow().replace(tzinfo=timezone.utc)
            valid_start = (now + timedelta(minutes=5 - now.minute % 5)).replace(second=0, microsecond=0)
            valid_end = valid_start + timedelta(minutes=5)

            message = (
                f"🔹 **Signal:** {signal}\n"
                f"📅 **Valid:** {valid_start.strftime('%H:%M')} - {valid_end.strftime('%H:%M')} UTC"
            )

            for user_id in context.bot_data.get("subscribers", []):
                try:
                    await context.bot.send_message(chat_id=user_id, text=message, parse_mode="Markdown")
                    logger.info(f"Signal sent to user {user_id}: {message}")
                except Exception as e:
                    logger.error(f"Failed to send signal to user {user_id}: {e}")
        else:
            logger.debug("No valid signal generated.")
    except Exception as e:
        logger.error(f"Error in send_signal: {e}")


def schedule_signals(job_queue: JobQueue):
    """Schedules the signal-sending job."""
    job_queue.run_repeating(send_signal, interval=SIGNAL_DURATION, first=0)
    logger.info("Signal-sending job scheduled.")


def on_message(ws, message):
    """Handles incoming WebSocket messages."""
    try:
        data = json.loads(message)
        logger.debug(f"Received WebSocket data: {data}")

        if 'topic' in data and data['topic'].startswith('kline'):
            candles = data.get('data', [])
            if candles:
                with data_lock:
                    for candle in candles:
                        if candlestick_queue.full():
                            candlestick_queue.get()  # Remove oldest data if queue is full
                        candlestick_queue.put(candle)
                logger.info(f"Stored new candlestick data: {candles[0]}")
            else:
                logger.warning("No valid candlestick data in message.")
    except json.JSONDecodeError:
        logger.error("Error decoding WebSocket message.")
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")


def on_error(ws, error):
    """Handles WebSocket errors."""
    logger.error(f"WebSocket error: {error}")


def on_close(ws, close_status_code, close_msg):
    """Handles WebSocket disconnections and attempts to reconnect."""
    logger.warning(f"WebSocket closed (Code: {close_status_code}, Message: {close_msg}). Reconnecting in 5s...")
    sleep(5)
    start_websocket()


def on_open(ws):
    """Sends subscription message upon WebSocket connection."""
    logger.info("WebSocket connection established.")
    subscription_message = json.dumps({
        "op": "subscribe",
        "args": [f"kline.{INTERVAL}.{SYMBOL}"]
    })
    ws.send(subscription_message)
    logger.debug(f"Sent WebSocket subscription: {subscription_message}")


def send_ping(ws):
    """Sends periodic ping messages to keep WebSocket connection alive."""
    while True:
        try:
            ping_message = json.dumps({"op": "ping"})
            ws.send(ping_message)
            logger.debug("Sent WebSocket ping.")
            sleep(20)
        except Exception as e:
            logger.error(f"Failed to send WebSocket ping: {e}")
            break


def start_websocket():
    """Starts the WebSocket connection with automatic reconnection."""
    while True:
        try:
            logger.info("Connecting to WebSocket...")
            ws = websocket.WebSocketApp(
                WS_URL,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.on_open = on_open

            ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
            ws_thread.start()

            ping_thread = threading.Thread(target=send_ping, args=(ws,), daemon=True)
            ping_thread.start()

            ws_thread.join()  # Keep the thread alive
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}. Retrying in 5 seconds...")
            sleep(5)


def main():
    """Main function to start the WebSocket and Telegram bot scheduler."""
    try:
        validate_telegram_token()
    except ValueError as e:
        logger.error(f"Bot failed to start: {e}")
        return

    # Start WebSocket in a separate thread
    websocket_thread = threading.Thread(target=start_websocket, daemon=True)
    websocket_thread.start()

    # Initialize Telegram bot and job scheduler
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.bot_data["subscribers"] = []

    # Schedule signals
    schedule_signals(application.job_queue)

    logger.info("Starting Telegram bot and scheduler...")
    application.run_polling()


if __name__ == "__main__":
    main()
