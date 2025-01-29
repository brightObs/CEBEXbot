import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from joblib import dump
import talib
from config.logger import get_logger

# Initialize logger
logger = get_logger()

# Dynamically select the most recent CSV file from the 'data/raw/' directory
data_folder = 'data/raw/'
csv_files = glob.glob(os.path.join(data_folder, 'historical_data_BTCUSDT_5m_*.csv'))
if not csv_files:
    raise FileNotFoundError("No CSV files found in the 'data/raw/' directory.")
latest_file = max(csv_files, key=os.path.getmtime)  # Select the most recent file

logger.info(f"Using data file: {latest_file}")

# Load the latest historical data
df = pd.read_csv(latest_file)

# Handle missing values (forward fill)
df.ffill(inplace=True)  # Forward fill to handle missing data
logger.info("Missing values forward-filled.")

# Feature Engineering: Add technical indicators
def add_features(df):
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

# Apply feature engineering
df = add_features(df)
logger.info("Feature engineering applied: SMA, RSI, ATR, Bollinger Bands.")

# Handle missing values after adding indicators
df.ffill(inplace=True)

# Feature Engineering: Use 'open', 'high', 'low', 'close', 'volume', and technical indicators as features
X = df[['open', 'high', 'low', 'close', 'volume', 'SMA_10', 'SMA_50', 'RSI', 'ATR', 'BB_upper', 'BB_lower']].values

# Create the target variable: 1 for Bullish (next close > current close), 0 for Bearish
# We are predicting 2 minutes ahead (based on your strategy)
df['target'] = np.where(df['close'].shift(-2) > df['close'], 1, 0)

# Drop rows with NaN targets (introduced by shifting)
df.dropna(subset=['target'], inplace=True)
y = df['target']

# Split the data into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
logger.info("Data split into training and test sets (80/20).")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model with GridSearchCV for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=3,  # 3-fold cross-validation
                           n_jobs=-1, verbose=2, scoring='accuracy')

grid_search.fit(X_train_scaled, y_train)

# Get the best model from grid search
best_rf_model = grid_search.best_estimator_
logger.info(f"Best RandomForest model found with parameters: {grid_search.best_params_}")

# Evaluate the model using cross-validation
cross_val_scores = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=5)  # 5-fold cross-validation
logger.info(f"Cross-validation accuracy: {cross_val_scores.mean() * 100:.2f}%")

# Evaluate the model on the test set
test_accuracy = best_rf_model.score(X_test_scaled, y_test)
logger.info(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Define the model and scaler saving paths
model_dir = 'models'  # Save models in a 'models' directory
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_file = os.path.join(model_dir, 'random_forest_model.joblib')
scaler_file = os.path.join(model_dir, 'scaler.joblib')

# Save the trained model and scaler
dump(best_rf_model, model_file)
dump(scaler, scaler_file)
logger.info(f"Model trained and saved as '{model_file}'.")
logger.info(f"Scaler saved as '{scaler_file}'.")
