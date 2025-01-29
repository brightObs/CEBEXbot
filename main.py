import logging
import os
from bot.bot import run_bot  # Ensure this contains the logic for signal generation
from config.settings import TELEGRAM_BOT_TOKEN
from bot.scheduler import start_websocket, schedule_signals
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Configure logging to file and console
log_path = 'config/logs'
os.makedirs(log_path, exist_ok=True)  # Create log directory if it doesn't exist

logging.basicConfig(
    filename=os.path.join(log_path, 'bot.log'),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add console logging for real-time feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)

# Global flag to track if the bot is active
is_bot_active = False

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles the /start command to activate the bot.
    """
    global is_bot_active

    if not is_bot_active:
        is_bot_active = True
        await update.message.reply_text("The bot is now active and functioning. You will receive signals shortly!")
        logger.info(f"Bot activated by user {update.effective_user.id}.")
    else:
        await update.message.reply_text("The bot is already active.")
        logger.info(f"Bot is already active. User {update.effective_user.id} tried to activate it again.")

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles the /stop command to deactivate the bot.
    """
    global is_bot_active

    if is_bot_active:
        is_bot_active = False
        await update.message.reply_text("The bot has been deactivated. You will no longer receive signals.")
        logger.info(f"Bot deactivated by user {update.effective_user.id}.")
    else:
        await update.message.reply_text("The bot is not active.")
        logger.info(f"User {update.effective_user.id} tried to deactivate an already inactive bot.")

def main():
    """
    Entry point for the bot application.
    Ensures the Telegram Bot Token is set and starts the bot.
    """
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("Telegram Bot Token is not set. Please ensure it's specified in the .env file.")

    logger.info("Starting the Telegram bot...")

    # Initialize Telegram Bot Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add command handlers for Telegram bot
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("stop", stop_command))

    # Store subscriber data in bot_data
    application.bot_data["subscribers"] = []  # Populate with Telegram chat IDs, e.g., [12345678]

    # Start WebSocket for fetching candlestick data
    logger.info("Starting WebSocket...")
    start_websocket()

    # Schedule signal sending
    logger.info("Scheduling signals...")
    schedule_signals(application.job_queue, application.bot_data)

    # Run the bot
    logger.info("Starting Telegram bot polling...")
    application.run_polling()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application terminated due to a critical error: {e}", exc_info=True)
