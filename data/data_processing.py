import os
import pandas as pd
import numpy as np
import talib
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from config.settings import SYMBOL, INTERVAL

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)

# Define directories
script_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(script_dir, 'raw')
processed_data_dir = os.path.join(script_dir, 'processed')
os.makedirs(processed_data_dir, exist_ok=True)

# Check if DVC (Data Version Control) is enabled
dvc_enabled = os.path.exists(os.path.join(script_dir, '.dvc'))

def get_latest_file():
    """Retrieve the most recent CSV file from the raw data directory."""
    try:
        files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
        if not files:
            raise FileNotFoundError("No raw data files found.")
        latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(raw_data_dir, f)))
        return os.path.join(raw_data_dir, latest_file)
    except Exception as e:
        logging.error(f"Error retrieving latest file: {e}")
        raise

def load_data(file_path):
    """Load and validate raw data."""
    try:
        data = pd.read_csv(file_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
        if data['timestamp'].isnull().any():
            raise ValueError("Invalid timestamp values detected.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def validate_data(df):
    """Handle missing values and duplicates."""
    df.ffill(inplace=True)  # Forward-fill missing values
    df.drop_duplicates(inplace=True)  # Remove duplicate rows
    return df

def calculate_indicators(df):
    """Calculate technical indicators."""
    if 'close' not in df.columns:
        raise ValueError("Missing 'close' column in data.")

    df.dropna(inplace=True)

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
    df['Upper_Band'] = upper
    df['Lower_Band'] = lower
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

def preprocess_data(data):
    """Preprocess data and scale features."""
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("Missing required columns.")

    data = validate_data(data)
    data = calculate_indicators(data)

    # Extract time-based features
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['hour'] = data['timestamp'].dt.hour
    data.drop(columns=['timestamp'], inplace=True)

    # Define columns to scale
    scaled_columns = ['SMA_50', 'SMA_200', 'RSI', 'EMA_9', 'EMA_21', 'MACD', 'MACD_Signal', 'MACD_Hist',
                      'Upper_Band', 'Lower_Band', 'bollinger_bandwidth', 'stochastic_oscillator',
                      'ADX', 'ATR', 'VWAP', 'OBV']

    scaler = StandardScaler()
    data[scaled_columns] = scaler.fit_transform(data[scaled_columns])

    return data, scaler

def save_processed_data(df, scaler):
    """Save processed data and scaler."""
    processed_file = os.path.join(processed_data_dir, 'processed_data.csv')
    scaler_file = os.path.join(processed_data_dir, 'scaler.pkl')

    if df.empty:
        raise ValueError("Processed data is empty.")

    df.to_csv(processed_file, index=False)
    joblib.dump(scaler, scaler_file)

    if dvc_enabled:
        os.system(f"dvc add {processed_file} {scaler_file}")

    logging.info(f"Processed data saved: {processed_file}")
    logging.info(f"Scaler saved: {scaler_file}")

def main():
    """Main function to execute data processing pipeline."""
    try:
        latest_file = get_latest_file()
        raw_data = load_data(latest_file)
        processed_data, scaler = preprocess_data(raw_data)
        save_processed_data(processed_data, scaler)
        logging.info("Data processing completed successfully.")
    except Exception as e:
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
