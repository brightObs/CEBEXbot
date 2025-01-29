import unittest
from unittest.mock import patch
from bot.signal import generate_signal

def generate_sample_candles(trend, count=50):
    """Generate sample candle data for testing."""
    candles = []
    for i in range(count):
        if trend == "bullish":
            candles.append({'open': str(100 + i), 'close': str(120 + i)})
        elif trend == "bearish":
            candles.append({'open': str(120 - i), 'close': str(100 - i)})
        elif trend == "sideways":
            candles.append({'open': str(100 + i), 'close': str(100 + i)})
    return candles

class TestSignal(unittest.TestCase):
    def test_generate_signal_none(self):
        candles = []
        self.assertIsNone(generate_signal(candles), "Expected None for empty candle list")

    def test_generate_signal_bullish(self):
        candles = generate_sample_candles("bullish")
        self.assertEqual(generate_signal(candles), "CALL", "Expected CALL for bullish candles")

    def test_generate_signal_bearish(self):
        candles = generate_sample_candles("bearish")
        self.assertEqual(generate_signal(candles), "PUT", "Expected PUT for bearish candles")

    def test_generate_signal_sideways(self):
        candles = generate_sample_candles("sideways")
        self.assertIsNone(generate_signal(candles), "Expected None for sideways market")

    @patch("bot.candlestick_signal.some_model.predict")
    def test_generate_signal_with_mocked_model(self, mock_predict):
        mock_predict.return_value = [1]  # Mocked output as CALL
        candles = generate_sample_candles("bullish")
        self.assertEqual(generate_signal(candles), "CALL")

if __name__ == "__main__":
    unittest.main()
