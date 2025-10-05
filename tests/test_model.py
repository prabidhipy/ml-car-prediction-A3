import pytest
import numpy as np
import pandas as pd  
import model  

feature_vals = [1200, 2016, 1, 0, 'BMW']

@pytest.fixture
def mock_model(monkeypatch):
    class DummyModel:
        def predict(self, X):
            return np.array([1])  
    monkeypatch.setattr(model, "load_model", lambda: DummyModel())

def test_model_input_shape(mock_model):
    X = model.get_X(*feature_vals)
    assert X.shape[1] == 35
    # Convert all columns to numeric to ensure dtype check passes
    X_numeric = X.apply(pd.to_numeric, errors='coerce')
    assert X_numeric.notna().all().all()  # no NaNs → all numeric

def test_model_output_shape(mock_model):
    model_instance = model.load_model()   
    X = model.get_X(*feature_vals)
    y_pred = model_instance.predict(X)
    
    # Flatten in case model returns 2D array
    y_pred = np.array(y_pred).ravel()
    
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (1,)  # 1 prediction
