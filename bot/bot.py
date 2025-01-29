import logging
import os
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.background import BackgroundScheduler
from bot.candlestick_signal import generate_signal, start_websocket
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

# Define the log directory and file
log_dir = r"C:\Users\TOSHIBA\PycharmProjects\CEBEXbot\data\logs"
log_file = os.path.join(log_dir, "bot.log")

# Ensure the log directory exists
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.abspath(log_file),
    filemode='a',  # Append to the log file
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,  # Set to DEBUG for detailed logging
)

logger = logging.getLogger(__name__)

# Bot command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles the /start command. Subscribes the user to signal notifications.
    """
    chat_id = update.effective_chat.id
    subscribers = context.bot_data.setdefault("subscribers", [])

    if chat_id not in subscribers:
        subscribers.append(chat_id)
        await update.message.reply_text("Signal bot activated! You will receive accurate signals.")
        logger.info(f"User {chat_id} subscribed.")
    else:
        await update.message.reply_text("You are already subscribed to signals.")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles the /stop command. Unsubscribes the user from signal notifications.
    """
    chat_id = update.effective_chat.id
    subscribers = context.bot_data.get("subscribers", [])

    if chat_id in subscribers:
        subscribers.remove(chat_id)
        await update.message.reply_text("Signal bot deactivated. You will no longer receive signals.")
        logger.info(f"User {chat_id} unsubscribed.")
    else:
        await update.message.reply_text("You are not subscribed to signals.")

async def notify_signals(context: ContextTypes.DEFAULT_TYPE):
    """
    Notifies all subscribed users with the latest signal.
    """
    try:
        # Replace with your function to fetch candlestick data
        candles = get_candlestick_data()
        if len(candles) < 50:
            logger.debug("Insufficient candlestick data for signal generation.")
            return

        signal = generate_signal(candles)
        if signal in ["CALL", "PUT"]:
            logger.info(f"Generated signal: {signal}")
            for chat_id in context.bot_data.get("subscribers", []):
                try:
                    await context.bot.send_message(chat_id=chat_id, text=f"New Signal: {signal}")
                except Exception as e:
                    logger.error(f"Error sending signal to chat {chat_id}: {e}")
        else:
            logger.info("No valid signal generated.")
    except Exception as e:
        logger.error(f"Error in notify_signals: {e}")

def run_bot(token):
    """
    Main function to initialize and run the Telegram bot.
    """
    if not token:
        logger.error("Bot token not set! Please provide a valid token.")
        return

    # Initialize WebSocket for candlestick updates
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
    scheduler = BackgroundScheduler()

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

if __name__ == "__main__":
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.critical("TELEGRAM_BOT_TOKEN is missing from the .env file.")
    else:
        run_bot(token)
