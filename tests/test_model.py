# in: tests/test_model.py

import pytest
import numpy as np
# We need to tell Python where to find your model.py file
import sys
sys.path.append('app/code')
from model import predict_selling_price

# Define a valid sample input for the model, based on your notebook
VALID_INPUT = {
    "engine": 1248.0,
    "owner": 'First Owner',
    "year": 2014,
    "transmission": 'Manual',
    "brand": 'Maruti'
}

def test_model_accepts_expected_input():
    """
    Test Case 1: The model takes the expected input.
    This test checks if the predict function runs without errors using a valid sample.
    """
    try:
        prediction, label = predict_selling_price(**VALID_INPUT)
        # We also check that the output types are correct (an integer and a string)
        assert isinstance(prediction, (int, np.integer))
        assert isinstance(label, str)
    except Exception as e:
        pytest.fail(f"Model prediction failed with valid input. Error: {e}")

def test_model_output_has_expected_shape():
    """
    Test Case 2: The output of the model has the expected shape.
    The prediction should be a single scalar value (not a list or array).
    """
    prediction, label = predict_selling_price(**VALID_INPUT)

    # np.isscalar checks if the output is a single value like 0, 1, 2, or 3.
    assert np.isscalar(prediction), "Prediction should be a single scalar value."