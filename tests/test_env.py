from config.settings import (
    TELEGRAM_BOT_TOKEN,
    BYBIT_API_KEY,
    BYBIT_API_SECRET,
    BYBIT_API_URL,
    SYMBOL,
    INTERVAL
)

def test_environment_variables():
    # Check if all environment variables are loaded correctly
    assert TELEGRAM_BOT_TOKEN is not None, "TELEGRAM_BOT_TOKEN is not set"
    assert BYBIT_API_KEY is not None, "BYBIT_API_KEY is not set"
    assert BYBIT_API_SECRET is not None, "BYBIT_API_SECRET is not set"
    assert BYBIT_API_URL is not None, "BYBIT_API_URL is not set"
    assert SYMBOL is not None, "SYMBOL is not set"
    assert INTERVAL is not None, "INTERVAL is not set"
