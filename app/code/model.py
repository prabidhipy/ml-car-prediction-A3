import os
import time
import pickle
import numpy as np
import pandas as pd
import logging
import mlflow
import mlflow.pyfunc

logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------
# Paths for scaler and encoders
# --------------------------
SCALAR_PATH = os.path.join(BASE_DIR, "Model", "car-scaling.model")
BRAND_ENCODER_PATH = os.path.join(BASE_DIR, "Model", "car_brand_encoder.model")
TRANSMISSION_ENCODER_PATH = os.path.join(BASE_DIR, "Model", "car_transmission_encoder.model")

# Load scaler and encoders
try:
    scaler = pickle.load(open(SCALAR_PATH, "rb"))
    brand_encoder = pickle.load(open(BRAND_ENCODER_PATH, "rb"))
    transmission_encoder = pickle.load(open(TRANSMISSION_ENCODER_PATH, "rb"))
except FileNotFoundError as e:
    logging.error(f"Could not load a model file: {e}. Please ensure the 'Model' directory and its contents are correct.")
    scaler, brand_encoder, transmission_encoder = None, None, None

# --------------------------
# Prepare feature vector
# --------------------------
def get_X(engine, owner, year, transmission, brand):
    """
    Takes raw input and converts it into a processed feature vector for the model.
    """
    if not all([scaler, brand_encoder, transmission_encoder]):
        raise RuntimeError("Preprocessing models are not loaded. Cannot create feature vector.")

    # Hardcoded owner mapping from the notebook
    owner_map = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4,
    }

    # Create a DataFrame from the input to handle transformations easily
    df_input = pd.DataFrame([{
        'engine': engine,
        'owner': owner,
        'year': year,
        'transmission': transmission,
        'brand': brand
    }])

    # Apply transformations as done in the notebook
    # 1. Map Owner
    df_input['owner'] = df_input['owner'].map(owner_map)

    # 2. Label Encode Transmission
    df_input['transmission'] = transmission_encoder.transform(df_input['transmission'])

    # 3. One-Hot Encode Brand
    brand_encoded = brand_encoder.transform(df_input[['brand']])
    brand_cols = brand_encoder.get_feature_names_out(['brand'])
    brand_df = pd.DataFrame(brand_encoded.toarray(), columns=brand_cols, index=df_input.index)
    df_processed = pd.concat([df_input, brand_df], axis=1).drop('brand', axis=1)

    # 4. Scale Numerical Features
    num_cols = ['engine', 'owner', 'year']
    df_processed[num_cols] = scaler.transform(df_processed[num_cols])
    
    # 5. Ensure column order matches the training data
    # The order is based on cell [553] in your notebook
    brand_cats = list(brand_encoder.categories_[0][1:]) # Get all brand categories except the dropped first one
    final_col_order = ['engine', 'owner', 'transmission', 'year'] + brand_cats
    
    # Reindex to ensure all columns are present and in order
    X_processed = df_processed.reindex(columns=final_col_order, fill_value=0)
    
    return X_processed.to_numpy()

# --------------------------
# Load MLflow model
# --------------------------
def load_model():
    """
    Load MLflow model using credentials from environment.
    Retries up to 5 times in case of failure.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://admin:password@mlflow.ml.brain.cs.ait.ac.th/")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        logging.info(f"Using MLflow credentials for user: {username}")
    else:
        logging.warning("No MLflow credentials found in environment.")

    mlflow.set_tracking_uri(mlflow_uri)

    run_id = os.getenv("RUN_ID")
    model_name = os.getenv("MODEL_NAME", "st126380-a3-model") # Updated model name from your notebook
    model_uri = f"runs:/{run_id}/model" if run_id else f"models:/{model_name}/Production"

    for attempt in range(5):
        try:
            logging.info(f"Loading MLflow model from {model_uri} (attempt {attempt + 1}/5)")
            model = mlflow.pyfunc.load_model(model_uri)
            logging.info("✅ MLflow model loaded successfully")
            return model
        except mlflow.exceptions.MlflowException as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(3)

    raise RuntimeError(f"Failed to load MLflow model after 5 attempts. Tried URI: {model_uri}")

# Load MLflow model at import
try:
    mlflow_model = load_model()
except Exception as e:
    logging.error(f"MLflow model could not be loaded: {e}")
    mlflow_model = None  # Allow app to start but prediction will fail

# --------------------------
# Predict function
# --------------------------
def predict_selling_price(engine, owner, year, transmission, brand):
    if mlflow_model is None:
        raise RuntimeError("MLflow model is not loaded. Cannot make predictions.")

    X = get_X(engine, owner, year, transmission, brand)
    raw_pred = mlflow_model.predict(X)[0]
    
    # Class mapping based on the quantile binning in your notebook
    class_map = ["Cheap (Lowest 25%)", "Average (25-50%)", "Expensive (50-75%)", "Very Expensive (Top 25%)"]
    label = class_map[int(raw_pred)]
    
    return raw_pred, label

# --------------------------
# Example Usage
# --------------------------
if __name__ == '__main__':
    # Example input values - you can change these to test
    example_input = {
        "engine": 1248.0,
        "owner": 'First Owner',
        "year": 2014,
        "transmission": 'Manual',
        "brand": 'Maruti'
    }

    try:
        prediction, label = predict_selling_price(**example_input)
        print("\\n" + "="*30)
        print("CAR PRICE PREDICTION")
        print("="*30)
        print(f"Input Features: {example_input}")
        print(f"Predicted Class: {prediction}")
        print(f"Predicted Label: {label}")
        print("="*30)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")