import json
import logging
import websocket
from time import sleep
import threading

# Initialize logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# Global variable for storing the latest candlestick data
latest_candle = {}

# WebSocket URL for Bybit's mainnet (USDT/USDC Perpetual)
WS_URL = "wss://stream.bybit.com/v5/public/linear"

# Define the WebSocket event handler functions
def on_message(ws, message):
    """
    Handle incoming messages from the WebSocket server.
    """
    try:
        data = json.loads(message)
        logger.debug(f"Incoming data: {data}")

        # Check if the data contains the expected topic and candle data
        if 'topic' in data and data['topic'].startswith('kline.'):
            candles = data.get('data', [])
            if isinstance(candles, list) and len(candles) > 0:
                latest_candle.update(candles[0])
                logger.info(f"Latest candlestick data updated: {candles[0]}")
            else:
                logger.warning("Candlestick data is empty or not in the expected format.")
        else:
            logger.warning("Received unexpected data or topic.")

    except json.JSONDecodeError:
        logger.error("Error decoding incoming message.")
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")

def on_error(ws, error):
    """
    Handle WebSocket errors.
    """
    logger.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    """
    Handle WebSocket closure.
    """
    logger.info(f"WebSocket closed with code {close_status_code} and message: {close_msg}")

def on_open(ws):
    """
    Handle WebSocket connection establishment and send subscription message.
    """
    logger.info("WebSocket connection established.")
    # Subscribe to the 5-minute candlestick data for BTCUSDT (adjust for desired pair)
    subscription_message = json.dumps({
        "op": "subscribe",
        "args": ["kline.5.BTCUSDT"]  # Update with desired pair and interval (5m)
    })
    ws.send(subscription_message)
    logger.debug(f"Subscription message sent: {subscription_message}")

def send_ping(ws):
    """
    Periodically send ping messages to keep the WebSocket connection alive.
    """
    while True:
        try:
            ping_message = json.dumps({"op": "ping"})
            ws.send(ping_message)
            logger.debug("Ping message sent.")
            sleep(20)  # Send a ping every 20 seconds
        except Exception as e:
            logger.error(f"Failed to send ping: {e}")
            break

def start_websocket():
    """
    Start and run the WebSocket connection.
    Automatically reconnects on disconnection.
    """
    while True:
        try:
            logger.info("Attempting to connect to WebSocket...")
            ws = websocket.WebSocketApp(
                WS_URL,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            ws.on_open = on_open

            # Start the WebSocket in a separate thread
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()

            # Send periodic ping messages
            ping_thread = threading.Thread(target=send_ping, args=(ws,), daemon=True)
            ping_thread.start()

            # Keep the thread alive
            ws_thread.join()
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            logger.info("Retrying in 5 seconds...")
            sleep(5)

if __name__ == "__main__":
    # Run the WebSocket in a separate thread to keep the main thread responsive
    ws_thread = threading.Thread(target=start_websocket, daemon=True)
    ws_thread.start()

    # Keep the main thread alive and display the latest candlestick data periodically
    try:
        while True:
            if latest_candle:
                logger.info(f"Current candlestick: {latest_candle}")
            sleep(10)
    except KeyboardInterrupt:
        logger.info("Shutting down WebSocket client.")
