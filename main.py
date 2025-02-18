import logging
import os
import sys
import json
import asyncio
from bot.bot import run_bot  # Main bot logic
from config.settings import TELEGRAM_BOT_TOKEN
from bot.scheduler import start_websocket, schedule_signals
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Directories and file paths
LOG_DIR = "config/logs"
SUBSCRIBERS_FILE = "config/subscribers.json"

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUBSCRIBERS_FILE), exist_ok=True)

# Logging configuration
LOG_FILE = os.path.join(LOG_DIR, "bot.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Console logging for real-time monitoring
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)

# Global flag to track bot activity status
is_bot_active = False


# ** Subscription Management **
def load_subscribers():
    """Load subscribers from a JSON file."""
    if os.path.exists(SUBSCRIBERS_FILE):
        try:
            with open(SUBSCRIBERS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def save_subscribers(subscribers):
    """Save subscribers to a JSON file."""
    with open(SUBSCRIBERS_FILE, "w") as f:
        json.dump(subscribers, f)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles /start command - subscribes the user.
    """
    global is_bot_active
    chat_id = update.effective_chat.id
    subscribers = load_subscribers()

    if chat_id not in subscribers:
        subscribers.append(chat_id)
        save_subscribers(subscribers)
        await update.message.reply_text("✅ You are now subscribed to trading signals!")
        logger.info(f"User {chat_id} subscribed.")
    else:
        await update.message.reply_text("🔔 You are already subscribed.")

    is_bot_active = True


async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles /stop command - unsubscribes the user.
    """
    global is_bot_active
    chat_id = update.effective_chat.id
    subscribers = load_subscribers()

    if chat_id in subscribers:
        subscribers.remove(chat_id)
        save_subscribers(subscribers)
        await update.message.reply_text("❌ You have unsubscribed from trading signals.")
        logger.info(f"User {chat_id} unsubscribed.")
    else:
        await update.message.reply_text("⚠️ You are not subscribed.")

    is_bot_active = False


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles /status command - checks if the bot is running.
    """
    global is_bot_active
    if is_bot_active:
        await update.message.reply_text("✅ The bot is currently active and sending signals.")
    else:
        await update.message.reply_text("❌ The bot is inactive.")


async def subscribers_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles /subscribers command - shows the number of subscribers (Admin use only).
    """
    subscribers = load_subscribers()
    await update.message.reply_text(f"👥 Total Subscribers: {len(subscribers)}")


# ** Main Bot Function **
def main():
    """
    Entry point for the bot application.
    Ensures the Telegram Bot Token is set and starts the bot.
    """
    try:
        if not TELEGRAM_BOT_TOKEN:
            raise ValueError("Telegram Bot Token is missing. Please check your .env file.")

        logger.info("Initializing the Telegram bot...")

        # Initialize Telegram Bot Application
        application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        # Add command handlers for Telegram bot
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("stop", stop_command))
        application.add_handler(CommandHandler("status", status_command))
        application.add_handler(CommandHandler("subscribers", subscribers_command))

        # Store subscriber data in bot_data
        application.bot_data["subscribers"] = load_subscribers()

        # Start WebSocket for fetching candlestick data
        logger.info("Starting WebSocket service...")
        start_websocket()

        # Schedule signal sending
        logger.info("Scheduling signal generation...")
        schedule_signals(application.job_queue, application.bot_data)

        # Run the bot
        logger.info("Starting Telegram bot polling...")
        application.run_polling()

    except Exception as e:
        logger.critical(f"Application terminated due to a critical error: {e}", exc_info=True)
        sys.exit(1)  # Exit the program with an error code


# ** Run the bot if executed directly **
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application stopped manually.")
    except Exception as e:
        logger.critical(f"Unexpected termination: {e}", exc_info=True)
