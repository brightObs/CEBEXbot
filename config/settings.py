import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Load environment variables from the .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

# Retrieve variables from the .env file
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
BYBIT_API_URL = os.getenv("BYBIT_API_URL", "https://api.bybit.com/v2/public/kline/list")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = os.getenv("INTERVAL", "5")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", None)  # Default to None if not set

# Additional settings
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "DEBUG")  # Default logging level can be adjusted
LOG_FILE_MAX_SIZE = int(os.getenv("LOG_FILE_MAX_SIZE", 10 * 1024 * 1024))  # 10MB
LOG_FILE_BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", 5))  # Default backup count

# Check for required variables and raise an exception if any are missing
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Critical environment variable TELEGRAM_BOT_TOKEN is missing. Check your .env file.")
if not BYBIT_API_KEY:
    raise ValueError("Critical environment variable BYBIT_API_KEY is missing. Check your .env file.")
if not BYBIT_API_SECRET:
    raise ValueError("Critical environment variable BYBIT_API_SECRET is missing. Check your .env file.")

# Log the loaded environment variables (excluding sensitive ones like API keys and tokens)
print(f"Loaded environment variables: SYMBOL={SYMBOL}, INTERVAL={INTERVAL}, LOGGING_LEVEL={LOGGING_LEVEL}")

# Base directory for logs (to ensure paths are absolute)
BASE_LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')

# Create the log directory if it doesn't exist
if not os.path.exists(BASE_LOG_DIR):
    os.makedirs(BASE_LOG_DIR)

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
    },
    'handlers': {
        'bot': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_LOG_DIR, 'bot.log'),
            'maxBytes': LOG_FILE_MAX_SIZE,
            'backupCount': LOG_FILE_BACKUP_COUNT,
            'formatter': 'default',
        },
        'signal': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_LOG_DIR, 'signal.log'),
            'maxBytes': LOG_FILE_MAX_SIZE,
            'backupCount': LOG_FILE_BACKUP_COUNT,
            'formatter': 'default',
        },
        'data_processing': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_LOG_DIR, 'data_processing.log'),
            'maxBytes': LOG_FILE_MAX_SIZE,
            'backupCount': LOG_FILE_BACKUP_COUNT,
            'formatter': 'default',
        },
        'model': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_LOG_DIR, 'model.log'),
            'maxBytes': LOG_FILE_MAX_SIZE,
            'backupCount': LOG_FILE_BACKUP_COUNT,
            'formatter': 'default',
        },
        'backtesting': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_LOG_DIR, 'backtesting.log'),
            'maxBytes': LOG_FILE_MAX_SIZE,
            'backupCount': LOG_FILE_BACKUP_COUNT,
            'formatter': 'default',
        },
        'errors': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_LOG_DIR, 'errors.log'),
            'maxBytes': LOG_FILE_MAX_SIZE,
            'backupCount': LOG_FILE_BACKUP_COUNT,
            'formatter': 'default',
        },
    },
    'loggers': {
        'bot': {
            'handlers': ['bot'],
            'level': LOGGING_LEVEL,
            'propagate': False,
        },
        'signal': {
            'handlers': ['signal'],
            'level': LOGGING_LEVEL,
            'propagate': False,
        },
        'data_processing': {
            'handlers': ['data_processing'],
            'level': LOGGING_LEVEL,
            'propagate': False,
        },
        'model': {
            'handlers': ['model'],
            'level': LOGGING_LEVEL,
            'propagate': False,
        },
        'backtesting': {
            'handlers': ['backtesting'],
            'level': LOGGING_LEVEL,
            'propagate': False,
        },
        'errors': {
            'handlers': ['errors'],
            'level': 'ERROR',
            'propagate': False,
        },
    },
}

# Apply logging configuration
from logging.config import dictConfig
dictConfig(LOGGING_CONFIG)

# Get loggers for modules
logger = logging.getLogger(__name__)

# Log any warnings or errors for optional configurations
if not TELEGRAM_CHAT_ID:
    logger.warning("TELEGRAM_CHAT_ID is not set in the .env file. Default value will be used.")
