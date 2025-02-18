import pytest
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Test Data Path
test_data_path = r'C:\Users\TOSHIBA\PycharmProjects\CEBEXbot\data\processed\processed_data.csv'

# Function to load test data and process it
@pytest.fixture
def load_test_data():
    # Load the test data
    test_data = pd.read_csv(test_data_path)
    drop_columns = ['timestamp', 'target']  # Drop unwanted columns
    X_test = test_data.drop(columns=drop_columns)
    y_test = test_data['target']
    return X_test, y_test

# Load the scaler and model before tests
@pytest.fixture
def load_model():
    scaler = load('C:/Users/TOSHIBA/PycharmProjects/CEBEXbot/models/scaler_v1.joblib')
    model = load('C:/Users/TOSHIBA/PycharmProjects/CEBEXbot/models/randomized_search_best_model.joblib')
    return scaler, model

# Test for model predictions and evaluation
def test_model_performance(load_test_data, load_model):
    X_test, y_test = load_test_data
    scaler, model = load_model

    # Scale the features
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Assertions to ensure the metrics are valid
    assert accuracy >= 0.0 and accuracy <= 1.0, f"Accuracy is out of range: {accuracy}"
    assert precision >= 0.0 and precision <= 1.0, f"Precision is out of range: {precision}"
    assert recall >= 0.0 and recall <= 1.0, f"Recall is out of range: {recall}"
    assert f1 >= 0.0 and f1 <= 1.0, f"F1 score is out of range: {f1}"

    # Print results
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
