import ccxt
import pandas as pd
import os
from dotenv import load_dotenv
import talib
import datetime
import numpy as np
import joblib
from tenacity import retry, stop_after_attempt, wait_fixed

# Load environment variables from the .env file
load_dotenv()

# Fetch API key and secret from the environment
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')

# Ensure API keys are loaded correctly
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("API credentials are missing. Check your .env file.")

# Initialize Bybit exchange with API credentials
exchange = ccxt.bybit({
    'apiKey': BYBIT_API_KEY,
    'secret': BYBIT_API_SECRET,
    'enableRateLimit': True
})

# Set up directories
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(script_dir, "data", "raw")
processed_data_dir = os.path.join(script_dir, "data", "processed")
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)

# Function to fetch historical OHLCV data
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def fetch_historical_data(symbol, timeframe='5m', limit=1000):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        return data
    except ccxt.BaseError as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return None

# Function to preprocess data and add indicators
def add_features(df):
    """Compute technical indicators and other relevant features."""
    df.ffill(inplace=True)  # Fill missing data

    # Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values

    # Moving Averages
    df['SMA_50'] = talib.SMA(closes, timeperiod=50)
    df['SMA_200'] = talib.SMA(closes, timeperiod=200)
    df['EMA_9'] = talib.EMA(closes, timeperiod=9)
    df['EMA_21'] = talib.EMA(closes, timeperiod=21)

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_upper'] = upper
    df['BB_lower'] = lower
    df['bollinger_bandwidth'] = (upper - lower) / middle

    # RSI
    df['RSI'] = talib.RSI(closes, timeperiod=14)

    # MACD
    macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist

    # Stochastic Oscillator
    slowk, slowd = talib.STOCH(highs, lows, closes, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['stochastic_oscillator'] = slowk
    df['stochastic_signal'] = slowd

    # ADX
    df['ADX'] = talib.ADX(highs, lows, closes, timeperiod=14)

    # ATR (Average True Range)
    df['ATR'] = talib.ATR(highs, lows, closes, timeperiod=14)

    # VWAP (Volume Weighted Average Price)
    df['VWAP'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    # OBV (On-Balance Volume)
    df['OBV'] = talib.OBV(closes, volumes)

    # Lag Features
    for i in range(1, 5):
        df[f'lag_{i}'] = df['close'].shift(i)

    # Candlestick Features
    df['Bullish'] = (df['close'] > df['open']).astype(int)
    df['Bearish'] = (df['close'] < df['open']).astype(int)
    df['Doji'] = ((df['close'] - df['open']).abs() < (df['high'] - df['low']) * 0.1).astype(int)
    df['Engulfing'] = ((df['Bullish'].shift(1) == 1) & (df['Bearish'] == 1) & (df['close'] < df['open'].shift(1))).astype(int)

    df.dropna(inplace=True)
    return df

# Function to define target labels
def add_target_column(df):
    """
    Adds a target column where:
    - 1 means the price closes higher after 2 candles
    - 0 means the price closes lower after 2 candles
    - 2 means a neutral movement (within a small range)
    """
    future_close = df['close'].shift(-2)
    df['target'] = np.where(future_close > df['close'], 1, np.where(future_close < df['close'], 0, 2))
    return df

# Main execution
if __name__ == "__main__":
    # Define trading pair and timeframe
    symbol = 'BTC/USDT'
    timeframe = '5m'
    limit = 1000

    # Fetch the historical data
    df = fetch_historical_data(symbol, timeframe, limit)

    if df is not None:
        # Add technical indicators and features
        df = add_features(df)

        # Add the target column
        df = add_target_column(df)

        # Construct a timestamped filename to avoid overwriting
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"historical_data_{symbol.replace('/', '')}_{timeframe}_{timestamp}.csv"

        # Save the data as a CSV file in the raw data folder
        raw_data_path = os.path.join(raw_data_dir, file_name)
        df.to_csv(raw_data_path, index=False)
        print(f"✅ Historical data with features and target column saved: {raw_data_path}")
    else:
        print(f"❌ Failed to fetch data for {symbol}.")
