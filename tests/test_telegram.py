import os
from dotenv import load_dotenv
import responses
import pytest
from bot.candlestick_signal import send_to_telegram

# Load environment variables
load_dotenv()

@pytest.fixture
def telegram_env():
    return {
        "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID")
    }

@responses.activate
def test_send_to_telegram(telegram_env):
    bot_token = telegram_env["bot_token"]
    api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    # Mock Telegram API response
    responses.add(
        responses.POST,
        api_url,
        json={"ok": True},
        status=200
    )

    # Test data
    signal = "CALL"
    start_time = "10:00"
    end_time = "10:05"

    # Call the function
    response = send_to_telegram(signal, start_time, end_time)

    # Assert the response
    assert response.status_code == 200
    assert response.json() == {"ok": True}

    print("Test passed: Message sent to Telegram successfully.")
