import unittest
from unittest.mock import AsyncMock, MagicMock
from bot.bot import start, stop
from telegram import Update
from telegram.ext import ContextTypes


class TestBot(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Set up mocks for the Telegram bot
        self.update = AsyncMock(spec=Update)
        self.context = MagicMock(spec=ContextTypes)
        self.context.bot_data = {"subscribers": []}  # Initialize bot_data

    async def test_start_command_new_user(self):
        self.update.effective_chat.id = 12345
        self.update.message.reply_text = AsyncMock()

        # Simulate /start command for a new user
        await start(self.update, self.context)

        # Check if the user was added to subscribers
        self.assertIn(12345, self.context.bot_data["subscribers"])
        self.update.message.reply_text.assert_awaited_with(
            "Signal bot activated! You will receive accurate signals."
        )

    async def test_start_command_existing_user(self):
        self.update.effective_chat.id = 12345
        self.update.message.reply_text = AsyncMock()

        # Simulate /start command for an existing user
        self.context.bot_data["subscribers"] = [12345]  # User already subscribed
        await start(self.update, self.context)

        # Ensure the user is not added again
        self.update.message.reply_text.assert_awaited_with(
            "You are already subscribed to signals."
        )

    async def test_stop_command_subscribed_user(self):
        self.update.effective_chat.id = 12345
        self.update.message.reply_text = AsyncMock()

        # Simulate /stop command for a subscribed user
        self.context.bot_data["subscribers"] = [12345]  # User already subscribed
        await stop(self.update, self.context)

        # Ensure the user was removed from subscribers
        self.assertNotIn(12345, self.context.bot_data["subscribers"])
        self.update.message.reply_text.assert_awaited_with(
            "Signal bot deactivated. You will no longer receive signals."
        )

    async def test_stop_command_unsubscribed_user(self):
        self.update.effective_chat.id = 12345
        self.update.message.reply_text = AsyncMock()

        # Simulate /stop command for an unsubscribed user
        self.context.bot_data["subscribers"] = []  # No subscribers
        await stop(self.update, self.context)

        # Ensure no removal happens
        self.update.message.reply_text.assert_awaited_with(
            "You are not subscribed to signals."
        )


if __name__ == "__main__":
    unittest.main()
