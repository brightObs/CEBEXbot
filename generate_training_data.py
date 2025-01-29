import os
import glob
import pandas as pd
import numpy as np
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump
from collections import Counter

# Get all CSV files in the raw data folder
raw_data_path = "C:/Users/TOSHIBA/PycharmProjects/CEBEXbot/data/raw/"
csv_files = glob.glob(os.path.join(raw_data_path, "historical_data_BTCUSDT_5m_*.csv"))

# Sort files by creation time and pick the most recent one
latest_file = max(csv_files, key=os.path.getctime)

# Load the most recent file
df = pd.read_csv(latest_file)
print(f"Using the most recent data file: {latest_file}")

# Resample to ensure consistent intervals
def resample_data(df, minutes=5):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df = df.resample(f'{minutes}min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df.reset_index(inplace=True)
    return df

df = resample_data(df)

# Add features
def add_features(df):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    df['SMA_10'] = talib.SMA(closes, timeperiod=10)
    df['RSI'] = talib.RSI(closes, timeperiod=14)
    df['ATR'] = talib.ATR(highs, lows, closes, timeperiod=14)
    return df

df = add_features(df)
df.ffill(inplace=True)
df.dropna(inplace=True)

# Create target for 2-minute prediction (shift by 2 minutes instead of 1)
df['target'] = np.where(df['close'].shift(-2) > df['close'], 1, 0)

# Prepare data
features = ['open', 'high', 'low', 'close', 'volume', 'SMA_10', 'RSI', 'ATR']
X = df[features].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check class distribution
print(f"Class distribution before resampling: {Counter(y_train)}")

# Use RandomUnderSampler and SMOTE for resampling
rus = RandomUnderSampler(random_state=42)
X_train, y_train = rus.fit_resample(X_train, y_train)  # First apply under-sampling
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)  # Then apply SMOTE
print(f"Class distribution after resampling: {Counter(y_train)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],  # Fewer estimators to avoid overfitting
    'max_depth': [5, 10, 20],  # Restricting depth to avoid complex trees
    'min_samples_split': [2, 5],  # Increased minimum split to reduce overfitting
    'min_samples_leaf': [1, 2],  # Increased minimum samples per leaf
    'bootstrap': [True]  # Bootstrapping for randomness
}

# Initialize the random forest model
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Perform GridSearchCV to find best hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Get the best model from GridSearchCV
rf_best = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# Cross-validation to check model performance
cross_val_scores = cross_val_score(rf_best, X_train_scaled, y_train, cv=5)
print(f"Cross-validation accuracy: {cross_val_scores.mean() * 100:.2f}%")

# Train the final model with the best parameters
rf_best.fit(X_train_scaled, y_train)

# Evaluate model
train_score = rf_best.score(X_train_scaled, y_train)
test_score = rf_best.score(X_test_scaled, y_test)
print(f"Training accuracy: {train_score * 100:.2f}%")
print(f"Test accuracy: {test_score * 100:.2f}%")

# Save the model and scaler in the 'models/' folder
model_file = 'models/model.joblib'
scaler_file = 'models/scaler.joblib'

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Save the model and scaler
dump(rf_best, model_file)
dump(scaler, scaler_file)

print(f"Model trained and saved as '{model_file}'.")
print(f"Scaler saved as '{scaler_file}'.")
