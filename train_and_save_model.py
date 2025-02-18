# === LIBRARIES ===
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, balanced_accuracy_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime
import warnings
from collections import Counter
import talib

warnings.filterwarnings("ignore")

# === SAVE MODEL WITH VERSIONING ===
def save_model(model, model_name="trading_model"):
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_file_path = f"{model_dir}/{model_name}_v{version}.pkl"
    joblib.dump(model, model_file_path)
    print(f"Model saved at: {model_file_path}")

# === DATA LOADING & PREPROCESSING ===
# Load the processed data which already contains day_of_week and hour columns
df = pd.read_csv("data/processed/processed_data.csv")

# Create a synthetic timestamp from day_of_week and hour.
# This timestamp is used solely for sorting; the day_of_week and hour columns remain available as features.
df['timestamp'] = pd.to_datetime(
    df['day_of_week'].astype(str) + ' ' + df['hour'].astype(str),
    format='%w %H'
)
df = df.sort_values('timestamp').set_index('timestamp')

# === ADVANCED FEATURE ENGINEERING ===
def add_advanced_features(df):
    """
    Enhance the dataset with additional technical indicators and lagged features.
    This function computes missing indicators if they are not already present.
    """
    # Compute missing indicators if necessary:
    if 'SMA_10' not in df.columns:
        df['SMA_10'] = talib.SMA(df['close'].values, timeperiod=10)
    if 'RSI_14' not in df.columns:
        df['RSI_14'] = talib.RSI(df['close'].values, timeperiod=14)  # Changed from 'RSI' to 'RSI_14'
    if 'ATR' not in df.columns:
        df['ATR'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
    if 'BB_upper' not in df.columns or 'BB_lower' not in df.columns:
        if 'SMA_20' not in df.columns:
            df['SMA_20'] = talib.SMA(df['close'].values, timeperiod=20)
        upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BB_upper'] = upper
        df['BB_lower'] = lower

    # Calculate missing EMAs if they don't already exist
    if 'EMA_12' not in df.columns:
        df['EMA_12'] = talib.EMA(df['close'].values, timeperiod=12)
    if 'EMA_26' not in df.columns:
        df['EMA_26'] = talib.EMA(df['close'].values, timeperiod=26)

    # Calculate MACD and MACD signal
    if 'MACD' not in df.columns or 'MACD_signal' not in df.columns:
        macd, macd_signal, _ = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macd_signal

    # Price momentum features
    df['EMA_12_26_diff'] = df['EMA_12'] - df['EMA_26']
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['MACD_hist_change'] = df['MACD_hist'].diff()

    # Volatility features
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['SMA_10']
    df['ATR_pct'] = df['ATR'] / df['close']
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()

    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_sma_10'] = df['volume'].rolling(10).mean()

    # Price action features
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    df['price_change_20'] = df['close'].pct_change(20)

    # Lagged features
    for lag in [1, 2, 3, 5, 8, 12, 20]:
        df[f'RSI_lag_{lag}'] = df['RSI_14'].shift(lag)  # Update 'RSI' to 'RSI_14'
        df[f'MACD_lag_{lag}'] = df['MACD'].shift(lag)
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

    # Interaction features
    df['RSI_MACD_interaction'] = df['RSI_14'] * df['MACD']  # Update 'RSI' to 'RSI_14'
    df['volume_price_interaction'] = df['volume'] * df['close']

    return df.dropna()


# Apply advanced feature engineering
df = add_advanced_features(df)

# === TARGET ENGINEERING ===
# Consistent with the preprocessing, we use a prediction horizon of 3 candles ahead
df['future_price'] = df['close'].shift(-3)
df['target'] = (df['future_price'] > df['close']).astype(int)
df = df.dropna()

# === FEATURE SELECTION ===
features = [
    'open', 'high', 'low', 'close', 'volume', 'SMA_10', 'SMA_50',
    'RSI', 'ATR', 'BB_upper', 'BB_lower', 'EMA_12', 'EMA_26',
    'RSI_14', 'MACD', 'MACD_signal', 'EMA_12_26_diff', 'MACD_hist',
    'BB_width', 'ATR_pct', 'volatility_20', 'volume_change',
    'volume_sma_10', 'price_change_5', 'price_change_10',
    'price_change_20', 'RSI_MACD_interaction', 'volume_price_interaction',
    'RSI_lag_1', 'RSI_lag_2', 'MACD_lag_1', 'MACD_lag_2', 'close_lag_1',
    'close_lag_2', 'volume_lag_1', 'volume_lag_2'
]

# Ensure all required feature columns exist in the DataFrame
missing_feats = [feat for feat in features if feat not in df.columns]
if missing_feats:
    raise ValueError(f"The following required feature columns are missing: {missing_feats}")

X = df[features]
y = df['target']

# === BALANCING DATA ===
# Check class distribution
from collections import Counter
class_counts = Counter(y)
print(f"Class distribution before SMOTE: {class_counts}")

# Apply SMOTE only if there is a significant imbalance
if abs(class_counts[0] - class_counts[1]) > 50:
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"Class distribution after SMOTE: {Counter(y_resampled)}")
else:
    print("SMOTE not applied as classes are already balanced.")
    X_resampled, y_resampled = X, y

# === SPLITTING DATA ===
split_idx = int(0.8 * len(X_resampled))
X_train, X_test = X_resampled.iloc[:split_idx], X_resampled.iloc[split_idx:]
y_train, y_test = y_resampled.iloc[:split_idx], y_resampled.iloc[split_idx:]

# === MODEL ARCHITECTURE ===
# Define base learners
base_models = [
    ('lgbm', lgb.LGBMClassifier(
        num_leaves=80,
        max_depth=15,
        learning_rate=0.02,
        n_estimators=1800,
        class_weight='balanced',
        boosting_type='gbdt',
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42
    )),
    ('xgb', xgb.XGBClassifier(
        learning_rate=0.05,
        max_depth=12,
        subsample=0.9,
        scale_pos_weight=1.1,
        n_estimators=1000,
        tree_method='hist',
        colsample_bytree=0.85,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )),
    ('rf', RandomForestClassifier(
        n_estimators=900,
        class_weight='balanced_subsample',
        max_depth=18,
        min_samples_split=6,
        max_features='sqrt',
        random_state=42
    )),
    ('et', ExtraTreesClassifier(
        n_estimators=700,
        max_depth=17,
        min_samples_split=4,
        max_features='sqrt',
        random_state=42
    ))
]

# Meta-model for stacking
meta_model = LogisticRegression(class_weight='balanced', C=0.07, max_iter=2500, random_state=42)
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    stack_method='predict_proba',
    passthrough=True,
    cv=5
)

# === FINAL PIPELINE ===
# Use RobustScaler to ensure consistency during inference (matches the training pipeline)
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('model', stacking_model)
])

# === TRAINING ===
pipeline.fit(X_train, y_train)

# === EVALUATION ===
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of class 1

# Evaluate performance
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Save the final model
save_model(pipeline)

