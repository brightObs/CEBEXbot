import json
import logging
import websocket
from time import sleep
import threading
from queue import Queue
from threading import Lock

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("WebSocketHandler")

# WebSocket URL for Bybit's mainnet (USDT/USDC Perpetual)
WS_URL = "wss://stream.bybit.com/v5/public/linear"

# Thread-safe data storage
candlestick_data = Queue(maxsize=100)
data_lock = Lock()

# WebSocket connection reference
ws = None


# Define the WebSocket event handler functions
def on_message(ws, message):
    """
    Handle incoming messages from the WebSocket server.
    """
    try:
        data = json.loads(message)
        logger.debug(f"Raw incoming data: {data}")  # Log full raw message

        if isinstance(data, dict):
            topic = data.get("topic")  # Safely get topic

            # Handle subscription confirmation messages
            if data.get("op") == "subscribe" and "success" in data:
                if data["success"]:
                    logger.info("Successfully subscribed to channel.")
                else:
                    logger.warning(f"Subscription failed: {data}")
                return  # Ignore further processing for subscription confirmations

            # Ensure the topic exists and is correctly formatted
            if topic and topic.startswith("kline.") and "data" in data:
                candles = data["data"]
                if isinstance(candles, list) and candles:
                    with data_lock:
                        for candlestick in candles:
                            if candlestick_data.full():
                                candlestick_data.get()  # Remove oldest data if full
                            candlestick_data.put(candlestick)
                    logger.info(f"Updated candlestick data: {candles[0]}")
                else:
                    logger.warning("Candlestick data is empty or not in the expected format.")
            else:
                logger.debug(f"Ignoring non-candlestick message: {data}")

        else:
            logger.warning(f"Received non-dictionary data: {data}")

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
    logger.warning(f"WebSocket closed with code {close_status_code} and message: {close_msg}")
    logger.info("Attempting to reconnect in 5 seconds...")
    sleep(5)
    start_websocket()


def on_open(ws):
    """
    Handle WebSocket connection establishment and send subscription message.
    """
    logger.info("WebSocket connection established.")
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
            if ws.sock and ws.sock.connected:
                ping_message = json.dumps({"op": "ping"})
                ws.send(ping_message)
                logger.debug("Ping message sent.")
            else:
                logger.warning("WebSocket is not connected. Skipping ping.")
                break
            sleep(20)  # Send a ping every 20 seconds
        except Exception as e:
            logger.error(f"Failed to send ping: {e}")
            break


def start_websocket():
    """
    Start and run the WebSocket connection.
    Automatically reconnects on disconnection.
    """
    global ws
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
            ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
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


def get_candlestick_data():
    """
    Retrieve candlestick data from the global WebSocket queue.
    Returns a list of the most recent candlesticks or None if insufficient data.
    """
    try:
        with data_lock:
            candles = list(candlestick_data.queue)
            if len(candles) < 50:  # Ensure sufficient data for analysis
                logger.warning("Insufficient candlestick data available.")
                return None
            return candles
    except Exception as e:
        logger.error(f"Error retrieving candlestick data: {e}")
        return None


if __name__ == "__main__":
    # Run the WebSocket in a separate thread to keep the main thread responsive
    ws_thread = threading.Thread(target=start_websocket, daemon=True)
    ws_thread.start()

    # Keep the main thread alive and display the latest candlestick data periodically
    try:
        while True:
            candles = get_candlestick_data()
            if candles:
                logger.info(f"Current candlestick: {candles[-1]}")
            sleep(10)
    except KeyboardInterrupt:
        logger.info("Shutting down WebSocket client.")
