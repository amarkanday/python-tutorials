import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix

# Load model once for all tests
@pytest.fixture(scope="session")
def pipeline():
    return joblib.load("../data/fraud_model.pkl")

# Sample test data fixture
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "amount": [200.0, 450.5, 1000.0],
        "num_transactions": [5, 12, 20],
        "merchant_type": ["retail", "electronics", "grocery"],
        "is_fraud": [0, 1, 0]  # Optional label, not used in model input
    })

# 1. Test missing value handling
def test_missing_values_handling(pipeline, sample_data):
    df_missing = sample_data.copy()
    df_missing.loc[0, "amount"] = np.nan
    processed = pipeline.named_steps["preprocessor"].transform(df_missing.drop(columns=["is_fraud"]))

    if isinstance(processed, csr_matrix):
        processed = processed.toarray()
    assert not np.isnan(processed).any(), "Pipeline should handle missing values properly."

# 2. Test scaling of numeric features
def test_scaled_numeric_features(pipeline, sample_data):
    processed = pipeline.named_steps["preprocessor"].transform(sample_data.drop(columns=["is_fraud"]))

    if isinstance(processed, csr_matrix):
        processed = processed.toarray()

    numeric_transformer = pipeline.named_steps["preprocessor"].named_transformers_["num"]
    num_scaled = processed[:, :len(numeric_transformer.mean_)]

    assert np.all(np.abs(num_scaled) < 3), "Numeric features should be scaled properly."

# 3. Test categorical feature encoding
def test_categorical_encoding(pipeline, sample_data):
    processed = pipeline.named_steps["preprocessor"].transform(sample_data.drop(columns=["is_fraud"]))

    if isinstance(processed, csr_matrix):
        processed = processed.toarray()

    cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_encoded = processed[:, -len(cat_encoder.get_feature_names_out()):]

    assert set(np.unique(cat_encoded)).issubset({0, 1}), "Categorical features should be one-hot encoded."

# 4. Test that the model produces valid probabilities
def test_prediction_probability_range(pipeline, sample_data):
    preds = pipeline.predict_proba(sample_data.drop(columns=["is_fraud"]))[:, 1]
    assert np.all((preds >= 0) & (preds <= 1)), "Model predictions must be valid probabilities."

# 5. Test pipeline end-to-end doesn't crash
def test_pipeline_runs(pipeline, sample_data):
    try:
        pipeline.predict(sample_data.drop(columns=["is_fraud"]))
    except Exception as e:
        pytest.fail(f"Pipeline failed on end-to-end prediction: {e}")

# 6. Parametrized test for edge cases (optional)
@pytest.mark.parametrize("amount,num_transactions", [
    (0.0, 1),
    (100000.0, 999),
    (-50.0, -1)
])
def test_pipeline_with_edge_cases(pipeline, amount, num_transactions):
    test_input = pd.DataFrame([{
        "amount": amount,
        "num_transactions": num_transactions,
        "merchant_type": "grocery"
    }])
    try:
        proba = pipeline.predict_proba(test_input)[:, 1]
        assert 0.0 <= proba[0] <= 1.0
    except Exception as e:
        pytest.fail(f"Pipeline failed on edge case input: {e}")
