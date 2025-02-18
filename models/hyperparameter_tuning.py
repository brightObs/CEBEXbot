import os
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, make_scorer
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Save model with versioning
def save_model(model, model_name="stacked_model"):
    model_dir = r"C:\Users\TOSHIBA\PycharmProjects\CEBEXbot\models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # Create the directory if it doesn't exist

    model_file_path = os.path.join(model_dir, f"{model_name}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
    joblib.dump(model, model_file_path)

# Load Data
df = pd.read_csv("C:/Users/TOSHIBA/PycharmProjects/CEBEXbot/data/processed/processed_data.csv")
df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

# Feature Engineering: Add missing features and enhance technical indicators
window = 20  # Period for Bollinger Bands
df['SMA'] = df['close'].rolling(window=20).mean()
df['StdDev'] = df['close'].rolling(window=20).std()
df['Upper_Band'] = df['SMA'] + (2 * df['StdDev'])
df['Lower_Band'] = df['SMA'] - (2 * df['StdDev'])
df['bollinger_bandwidth'] = (df['Upper_Band'] - df['Lower_Band']) / df['SMA']

df['SMA_200'] = df['close'].rolling(window=200).mean()

low_min = df['low'].rolling(window=14).min()
high_max = df['high'].rolling(window=14).max()
df['stochastic_oscillator'] = (df['close'] - low_min) / (high_max - low_min) * 100
df['%D'] = df['stochastic_oscillator'].rolling(window=3).mean()

# Add Moving Average Convergence Divergence (MACD)
df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Add Relative Strength Index (RSI)
delta = df['close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Lag features
for i in range(1, 6):  # Increased lag window
    df[f'lag_{i}'] = df['close'].shift(i)

# Drop rows with NaN values after feature engineering
df.dropna(inplace=True)

# Feature selection and train-test split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle class imbalance with SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Calculate class weights for better handling of imbalanced data
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Hyperparameter tuning function using optuna
def objective(trial):
    model_type = trial.suggest_categorical('model', ['RandomForest', 'XGBoost', 'ExtraTrees', 'GradientBoosting'])

    if model_type == 'RandomForest':
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
            random_state=42,
            class_weight='balanced'  # Adding class weights for imbalance
        )
    elif model_type == 'XGBoost':
        model = XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            learning_rate=trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
            random_state=42,
            scale_pos_weight=class_weights[1] / class_weights[0]  # Handle imbalance in XGBoost
        )
    elif model_type == 'ExtraTrees':
        model = ExtraTreesClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            random_state=42,
            class_weight='balanced'
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 1000),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            learning_rate=trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
            random_state=42
        )

    model.fit(X_train_res, y_train_res)
    accuracy = model.score(X_test, y_test)
    return accuracy

# Optuna study to find the best hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get best trial results and model
best_trial = study.best_trial
print(f"Best Model: {best_trial.params}")
print(f"Best Accuracy: {best_trial.value}")

# Create the best model with the chosen hyperparameters
best_model_type = best_trial.params['model']

if best_model_type == 'RandomForest':
    best_model = RandomForestClassifier(
        n_estimators=439,
        max_depth=3,
        min_samples_split=20,
        min_samples_leaf=9,
        random_state=42,
        class_weight='balanced'
    )
elif best_model_type == 'XGBoost':
    best_model = XGBClassifier(
        n_estimators=best_trial.params['n_estimators'],
        max_depth=best_trial.params['max_depth'],
        learning_rate=best_trial.params['learning_rate'],
        random_state=42,
        scale_pos_weight=class_weights[1] / class_weights[0]
    )
elif best_model_type == 'ExtraTrees':
    best_model = ExtraTreesClassifier(
        n_estimators=best_trial.params['n_estimators'],
        max_depth=best_trial.params['max_depth'],
        random_state=42,
        class_weight='balanced'
    )
else:
    best_model = GradientBoostingClassifier(
        n_estimators=best_trial.params['n_estimators'],
        max_depth=best_trial.params['max_depth'],
        learning_rate=best_trial.params['learning_rate'],
        random_state=42
    )

# Stacking Model: Combine multiple classifiers using a logistic regression meta-model
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.05, random_state=42))
]

meta_model = LogisticRegression()

stacked_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model)

# Train and evaluate the model
stacked_model.fit(X_train_res, y_train_res)

# Evaluate performance
stacked_model_accuracy = stacked_model.score(X_test, y_test)
print(f"Stacked Model Accuracy on Final Test Set: {stacked_model_accuracy}")

# Generate the classification report
y_pred = stacked_model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the best model
save_model(stacked_model, model_name="stacked_model_final")
