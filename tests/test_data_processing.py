import pytest
import logging
import pandas as pd
import numpy as np
import os
from data.data_processing import extract_features_and_labels, preprocess_data

# Ensure the directory exists
log_directory = r'C:\Users\TOSHIBA\PycharmProjects\CEBEXbot\tests\data\logs'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Set up logging configuration to log to a file
log_file_path = os.path.join(log_directory, 'test_log.log')

logging.basicConfig(
    filename=log_file_path,
    level=logging.DEBUG,  # Set the desired log level (DEBUG, INFO, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_extract_features():
    logging.debug("Testing feature extraction...")

    # Simulated sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2025-01-22', periods=100, freq='T'),
        'open': np.random.random(100) * 50000,
        'high': np.random.random(100) * 50000,
        'low': np.random.random(100) * 50000,
        'close': np.random.random(100) * 50000,
        'volume': np.random.random(100) * 1000
    })

    # Preprocess the data to calculate indicators
    sample_data, _ = preprocess_data(sample_data)  # Preprocess to calculate indicators

    # Now extract features and labels
    result = extract_features_and_labels(sample_data)

    # Check if result is None
    if result is None:
        logging.error("Feature extraction failed. Missing required columns.")
        assert False, "Feature extraction returned None"
    else:
        features, labels = result
        logging.debug(f"Features: {features.head()}")
        logging.debug(f"Labels: {labels.head()}")

        # Ensure features and labels are not empty
        assert not features.empty, "Features DataFrame is empty"
        assert not labels.empty, "Labels Series is empty"

        # Check if the required columns are present
        required_columns = ['SMA_50', 'RSI_14', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal']
        for col in required_columns:
            assert col in features.columns, f"Missing column: {col}"
