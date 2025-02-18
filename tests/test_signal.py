import unittest
from unittest.mock import patch
import pandas as pd
from bot.candlestick_signal import generate_signal
from data.data_processing import extract_features


def generate_sample_candles(trend, count=100):
    """Generate sample candles with a trend and include volume."""
    candles = []
    for i in range(count):
        if trend == "bullish":
            close = 100 + i
            high = close + 5
            low = close - 5
        elif trend == "bearish":
            close = 100 - i
            high = close + 5
            low = close - 5
        else:
            close = 100
            high = close + 2
            low = close - 2
        candles.append({
            "close": close,
            "high": high,
            "low": low,
            "open": close - 2,
            "volume": 1000  # Ensure volume is included
        })
    return candles

class TestSignal(unittest.TestCase):

    def test_generate_signal_none(self):
        """Test that an empty list of candles returns None."""
        candles = []
        self.assertIsNone(generate_signal(candles), "Expected None for empty candle list")

    def test_generate_signal_bullish(self):
        """Test that a bullish trend results in a CALL signal."""
        candles = generate_sample_candles("bullish")
        df = pd.DataFrame(candles)
        features, _ = extract_features(df)
        self.assertEqual(generate_signal(features), "CALL", "Expected CALL for bullish candles")

    def test_generate_signal_bearish(self):
        """Test that a bearish trend results in a PUT signal."""
        candles = generate_sample_candles("bearish")
        df = pd.DataFrame(candles)
        features, _ = extract_features(df)
        self.assertEqual(generate_signal(features), "PUT", "Expected PUT for bearish candles")

    def test_generate_signal_sideways(self):
        """Test that a sideways trend results in None."""
        candles = generate_sample_candles("sideways")
        df = pd.DataFrame(candles)
        features, _ = extract_features(df)
        self.assertIsNone(generate_signal(features), "Expected None for sideways market")

    def test_generate_signal_choppy(self):
        """Test that a choppy (uncertain) trend results in None."""
        candles = generate_sample_candles("sideways")
        df = pd.DataFrame(candles)
        # Add small fluctuations
        df["close"] += [(-1) ** i * 0.5 for i in range(len(df))]
        features, _ = extract_features(df)
        self.assertIsNone(generate_signal(features), "Expected None for choppy market")

    def test_generate_signal_near_sideways(self):
        """Test a market with a slight bullish trend that may still be classified as sideways."""
        candles = generate_sample_candles("sideways")
        df = pd.DataFrame(candles)
        df["close"] += [0.1 * i for i in range(len(df))]
        features, _ = extract_features(df)
        self.assertIsNone(generate_signal(features), "Expected None for near-sideways market")

    @patch("bot.candlestick_signal.extract_features")
    def test_generate_signal_with_mocked_features_call(self, mock_extract_features):
        """Test a bullish trend with a mocked feature extraction returning CALL."""
        mock_extract_features.return_value = (
            pd.DataFrame({"SMA_50": [101], "RSI_14": [55]}),  # Simulating bullish conditions
            None
        )
        candles = generate_sample_candles("bullish")
        self.assertEqual(generate_signal(candles), "CALL", "Expected CALL from mocked feature extraction")

    @patch("bot.candlestick_signal.extract_features")
    def test_generate_signal_with_mocked_features(self, mock_extract_features):
        mock_extract_features.return_value = (
            pd.DataFrame({"SMA_50": [101], "RSI_14": [55]}),
            None
        )
        candles = generate_sample_candles("bullish")
        self.assertEqual(generate_signal(candles), "CALL", "Expected CALL from mocked feature extraction")


if __name__ == "__main__":
    unittest.main()
