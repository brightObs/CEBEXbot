import sys
import os
import logging
import requests
from telegram import Bot
from telegram.ext import Application, CallbackContext, JobQueue
from datetime import datetime, timezone, timedelta
import threading
from time import sleep
import json
import websocket

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import signal handling utilities
from bot.candlestick_signal import get_candlestick_data, generate_signal

# Configure logging
LOG_DIR = os.path.join(os.path.dirname(__file__), '../data/logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "scheduler.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Signal duration in seconds
SIGNAL_DURATION = 300  # 5 minutes

def validate_telegram_token(telegram_token: str):
    """
    Validates the Telegram bot token.
    """
    try:
        url = f"https://api.telegram.org/bot{telegram_token}/getMe"
        response = requests.get(url)
        response_data = response.json()
        if response_data.get("ok"):
            logger.info("Telegram bot token is valid.")
        else:
            logger.error("Invalid Telegram bot token.")
            raise ValueError("Invalid Telegram bot token.")
    except Exception as e:
        logger.error(f"Error validating Telegram bot token: {e}")
        raise ValueError("Error validating Telegram bot token.")

async def send_signal(context: CallbackContext):
    """
    Generates and sends signals to Telegram subscribers.
    """
    try:
        candles = get_candlestick_data()
        if candles:
            signal = generate_signal(candles)
            if signal:
                now = datetime.utcnow().replace(tzinfo=timezone.utc)
                valid_start = (now + timedelta(minutes=5 - now.minute % 5)).replace(second=0, microsecond=0)
                valid_end = valid_start + timedelta(minutes=5)

                message = f"Signal: {signal}\nValid: {valid_start.strftime('%H:%M')} - {valid_end.strftime('%H:%M')} UTC"
                for user_id in context.bot_data.get("subscribers", []):
                    try:
                        await context.bot.send_message(chat_id=user_id, text=message)
                        logger.info(f"Signal sent to user {user_id}: {message}")
                    except Exception as e:
                        logger.error(f"Failed to send signal to user {user_id}: {e}")
            else:
                logger.debug("No valid signal generated.")
        else:
            logger.debug("No candlestick data available for signal generation.")
    except Exception as e:
        logger.error(f"Error in send_signal: {e}")

def schedule_signals(job_queue: JobQueue, bot_data: dict):
    """
    Schedules the signal-sending job.
    """
    job_queue.run_repeating(send_signal, interval=SIGNAL_DURATION, first=0)
    logger.info("Signal-sending job scheduled.")

def on_message(ws, message):
    """
    Handles incoming WebSocket messages.
    """
    try:
        data = json.loads(message)
        if 'topic' in data and data['topic'].startswith('kline'):
            candles = data.get('data', [])
            if candles:
                logger.info(f"New candlestick data received: {candles}")
                # Store or process candlestick data here
            else:
                logger.warning("No candlestick data in the message.")
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")

def on_error(ws, error):
    """
    Handles WebSocket errors.
    """
    logger.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    """
    Handles WebSocket closure.
    """
    logger.warning("WebSocket connection closed. Reconnecting...")
    sleep(5)
    start_websocket()

def on_open(ws):
    """
    Handles WebSocket connection opening.
    """
    logger.info("WebSocket connection established.")
    ws.send(json.dumps({
        "op": "subscribe",
        "args": ["kline.5.BTCUSDT"]  # Modify symbol and interval as necessary
    }))

def start_websocket():
    """
    Starts the WebSocket connection.
    """
    try:
        ws = websocket.WebSocketApp(
            "wss://stream.bybit.com/v5/public/linear",  # Bybit WebSocket URL
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        ws.on_open = on_open
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()
        logger.info("WebSocket thread started.")
    except Exception as e:
        logger.error(f"Failed to start WebSocket: {e}")

def main():
    """
    Main function to start the scheduler and WebSocket.
    """
    from config.settings import TELEGRAM_BOT_TOKEN

    try:
        validate_telegram_token(TELEGRAM_BOT_TOKEN)
    except ValueError as e:
        logger.error(f"Bot failed to start: {e}")
        return

    start_websocket()

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.bot_data["subscribers"] = []

    schedule_signals(application.job_queue, application.bot_data)

    logger.info("Starting Telegram bot and scheduler...")
    application.run_polling()

if __name__ == "__main__":
    main()
