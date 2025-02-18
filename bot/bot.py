import logging
import os
import json
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from bot.candlestick_signal import generate_signal, get_candlestick_data
from bot.websocket_handler import start_websocket  # WebSocket handler module

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

# Define directories
LOG_DIR = r"C:\Users\TOSHIBA\PycharmProjects\CEBEXbot\data\logs"
SUBSCRIBERS_FILE = os.path.join(os.path.dirname(__file__), "../config/subscribers.json")

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
LOG_FILE = os.path.join(LOG_DIR, "bot.log")
logging.basicConfig(
    filename=os.path.abspath(LOG_FILE),
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler = BackgroundScheduler()


# ** Subscription Management **
def load_subscribers():
    """Load subscribers from file."""
    if os.path.exists(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_subscribers(subscribers):
    """Save subscribers to file."""
    with open(SUBSCRIBERS_FILE, "w") as f:
        json.dump(subscribers, f)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    # Ensure 'subscribers' key exists
    if "subscribers" not in context.bot_data:
        context.bot_data["subscribers"] = []

    if chat_id in context.bot_data["subscribers"]:
        await update.message.reply_text("⚠️ You are already subscribed!")
    else:
        context.bot_data["subscribers"].append(chat_id)  # Ensure modification happens
        await update.message.reply_text("✅ You have subscribed to trading signals!")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    # Ensure 'subscribers' key exists
    if "subscribers" not in context.bot_data:
        context.bot_data["subscribers"] = []

    if chat_id in context.bot_data["subscribers"]:
        context.bot_data["subscribers"].remove(chat_id)  # Properly remove user
        await update.message.reply_text("🚫 You have unsubscribed from trading signals.")
    else:
        await update.message.reply_text("⚠️ You are not subscribed!")


# ** Signal Notifications **
async def notify_signals(context: ContextTypes.DEFAULT_TYPE):
    """Notifies all subscribed users with the latest signal."""
    try:
        candles = get_candlestick_data()
        if len(candles) < 50:
            logger.debug("Insufficient candlestick data for signal generation.")
            return

        signal = generate_signal(candles)
        if signal in ["CALL", "PUT"]:
            logger.info(f"Generated signal: {signal}")

            subscribers = load_subscribers()
            for chat_id in subscribers:
                try:
                    await context.bot.send_message(chat_id=chat_id, text=f"🔹 **Signal:** {signal}")
                except Exception as e:
                    logger.error(f"Error sending signal to chat {chat_id}: {e}")
        else:
            logger.info("No valid signal generated.")
    except Exception as e:
        logger.error(f"Error in notify_signals: {e}")


# ** Main Bot Function **
def run_bot(token):
    """Main function to initialize and run the Telegram bot."""
    if not token:
        logger.error("Bot token not set! Please provide a valid token.")
        return

    # Start WebSocket for candlestick updates
    try:
        logger.info("Starting WebSocket connection...")
        start_websocket()
    except Exception as e:
        logger.error(f"Error initializing WebSocket: {e}")

    # Create the Telegram bot application
    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("stop", stop))

    # Initialize scheduler for signal notifications
    def notify_signals_job():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(notify_signals(application))
        loop.close()

    scheduler.add_job(
        notify_signals_job,
        "interval",
        minutes=5,
        id="signal_notifier",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started. Bot is running...")

    # Start polling for updates
    try:
        application.run_polling()
    except Exception as e:
        logger.error(f"Error running bot: {e}")
    finally:
        scheduler.shutdown()


# ** Run the bot if executed directly **
if __name__ == "__main__":
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.critical("TELEGRAM_BOT_TOKEN is missing from the .env file.")
    else:
        run_bot(token)
