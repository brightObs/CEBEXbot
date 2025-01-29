import os
import pandas as pd
import numpy as np
import talib
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from config.settings import SYMBOL, INTERVAL

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories
raw_data_dir = os.path.join(os.path.dirname(__file__), 'raw')
processed_data_dir = os.path.join(os.path.dirname(__file__), 'processed')
os.makedirs(processed_data_dir, exist_ok=True)

# Data Versioning using DVC
dvc_enabled = os.path.exists(os.path.join(os.path.dirname(__file__), '.dvc'))


def get_latest_file():
    files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No raw data files found.")
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(raw_data_dir, f)))
    return os.path.join(raw_data_dir, latest_file)


def load_data(file_path):
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
    if df.isnull().sum().any():
        logging.warning("Missing values detected. Filling with previous values.")
        df.fillna(method='ffill', inplace=True)
    if df.duplicated().any():
        logging.warning("Duplicate values detected. Removing duplicates.")
        df.drop_duplicates(inplace=True)
    return df


def calculate_indicators(df):
    try:
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(inplace=True)
        closes = df['close'].values
        df['SMA_50'] = talib.SMA(closes, timeperiod=50)
        df['RSI_14'] = talib.RSI(closes, timeperiod=14)
        df['EMA_12'] = talib.EMA(closes, timeperiod=12)
        df['EMA_26'] = talib.EMA(closes, timeperiod=26)
        df['MACD'], df['MACD_signal'], _ = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        raise
    return df


def preprocess_data(data):
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Data is missing required columns: {required_columns}")

    data = validate_data(data)
    data = calculate_indicators(data)

    scaler = StandardScaler()
    scaled_columns = ['SMA_50', 'RSI_14', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal']
    data[scaled_columns] = scaler.fit_transform(data[scaled_columns].fillna(0))

    return data, scaler


def extract_features_and_labels(df):
    required_columns = ['SMA_50', 'RSI_14', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal']
    features = df[required_columns]
    labels = df['close']
    return features, labels


def save_processed_data(df, scaler):
    processed_file = os.path.join(processed_data_dir, 'processed_data.csv')
    df.to_csv(processed_file, index=False)
    joblib.dump(scaler, os.path.join(processed_data_dir, 'scaler.joblib'))
    if dvc_enabled:
        os.system(f"dvc add {processed_file}")
    logging.info("Processed data and scaler saved successfully.")


def test_extract_features():
    logging.debug("Testing feature extraction...")
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2025-01-22', periods=100, freq='T'),
        'open': np.random.random(100) * 50000,
        'high': np.random.random(100) * 50000,
        'low': np.random.random(100) * 50000,
        'close': np.random.random(100) * 50000,
        'volume': np.random.random(100) * 1000
    })
    sample_data, _ = preprocess_data(sample_data)
    features, labels = extract_features_and_labels(sample_data)
    assert not features.isnull().values.any(), "Features contain NaN values."
    assert not labels.isnull().values.any(), "Labels contain NaN values."
    logging.debug("Feature extraction test passed.")


def main():
    try:
        test_extract_features()
        latest_file = get_latest_file()
        raw_data = load_data(latest_file)
        processed_data, scaler = preprocess_data(raw_data)
        save_processed_data(processed_data, scaler)
        logging.info("Data processing completed successfully.")
    except Exception as e:
        logging.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()