import ccxt
import pandas as pd
import os
from dotenv import load_dotenv
import talib
import datetime
from tenacity import retry, stop_after_attempt, wait_fixed

# Load environment variables from the .env file
load_dotenv()

# Fetch API key and secret from the environment
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')

# Ensure API keys are loaded correctly
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("API credentials are missing. Check your .env file.")

# Initialize the Bybit exchange with API credentials
exchange = ccxt.bybit({
    'apiKey': BYBIT_API_KEY,
    'secret': BYBIT_API_SECRET,
    'enableRateLimit': True  # Ensure rate limiting is enabled to avoid getting blocked
})

# Function to fetch historical data from Bybit for the given timeframe
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def fetch_historical_data(symbol, timeframe='5m', limit=1000):
    """
    Fetch historical OHLCV data for a given symbol and timeframe.

    Parameters:
        symbol (str): The trading pair (e.g., 'BTC/USDT').
        timeframe (str): Timeframe for the data (default '5m').
        limit (int): Number of candles to fetch (default 1000).

    Returns:
        pd.DataFrame: DataFrame containing OHLCV data.
    """
    try:
        # Fetch historical OHLCV data using Bybit's fetch_ohlcv method
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        # Convert the data into a pandas DataFrame
        data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Convert the timestamp to a readable date format
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

        return data
    except ccxt.BaseError as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return None

# Function to add additional features
def add_features(df):
    """
    Add technical indicators and datetime features to the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing OHLCV data.

    Returns:
        pd.DataFrame: DataFrame with additional features.
    """
    # Handle missing data by forward-filling or interpolating
    df.ffill(inplace=True)

    # Extract hour and day of the week from the timestamp
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Extract technical indicators
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values

    # Adding moving averages
    df['SMA_10'] = talib.SMA(closes, timeperiod=10)
    df['SMA_50'] = talib.SMA(closes, timeperiod=50)

    # Adding RSI (Relative Strength Index)
    df['RSI'] = talib.RSI(closes, timeperiod=14)

    # Adding ATR (Average True Range)
    df['ATR'] = talib.ATR(highs, lows, closes, timeperiod=14)

    # Adding Bollinger Bands (20 period, 2 std deviation)
    upperband, middleband, lowerband = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_upper'] = upperband
    df['BB_lower'] = lowerband

    return df

# Function to add a 'target' column to the data
def add_target_column(df):
    """
    Add a 'target' column: 1 if the close 2 minutes ahead is higher than the current close, else 0.

    Parameters:
        df (pd.DataFrame): DataFrame containing OHLCV data.

    Returns:
        pd.DataFrame: DataFrame with the 'target' column added.
    """
    df['target'] = (df['close'].shift(-2) > df['close']).astype(int)
    return df

# Main execution
if __name__ == "__main__":
    # Define the trading pair and timeframe
    symbol = 'BTC/USDT'  # Change this to any other pair if needed
    timeframe = '5m'
    limit = 1000

    # Fetch the historical data
    df = fetch_historical_data(symbol, timeframe, limit)

    if df is not None:
        # Add additional features
        df = add_features(df)

        # Add the target column
        df = add_target_column(df)

        # Construct a timestamp for the filename to avoid overwriting
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"historical_data_{symbol.replace('/', '')}_{timeframe}_{timestamp}.csv"

        # Save the data as a CSV file in the raw data folder
        raw_data_path = os.path.join("data", "raw", file_name)
        df.to_csv(raw_data_path, index=False)
        print(f"Historical data with features and target column saved as {raw_data_path}")
    else:
        print(f"Failed to fetch data for {symbol}.")
