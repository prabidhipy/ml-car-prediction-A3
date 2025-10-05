import os
import time
import pickle
import numpy as np
import pandas as pd
import logging
import mlflow
import mlflow.pyfunc
import matplotlib.pyplot as plt # Added for the plot function in the class

logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------------
# STEP 1: Move the custom model classes here from your notebook
# -------------------------------------------------------------------
class LogisticRegression:
    
    def __init__(self, k, n, method, alpha=0.001, max_iter=5000, regularization=None,
             epochs=None, use_penalty=False, lambda_=0.0):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.regularization = regularization
        self.epochs = epochs
        self.use_penalty = use_penalty
        self.lambda_ = lambda_
        self.W = np.random.rand(self.n, self.k) # Initialize weights
        self.losses = []
    
    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        self.losses = []
        
        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad =  self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "minibatch":
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0] - batch_size) # Fix potential out-of-bounds
                batch_X = X[ix:ix+batch_size]
                batch_Y = Y[ix:ix+batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "sto":
            start_time = time.time()
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while i in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                
                X_train_sto = X[idx, :].reshape(1, -1)
                Y_train_sto = Y[idx].reshape(1, -1)

                loss, grad = self.gradient(X_train_sto, Y_train_sto)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                
                list_of_used_ix.append(i)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
                if i % 500 == 0:
                    full_loss, _ = self.gradient(X, Y)
                    print(f"Loss at iteration {i}", full_loss)
            print(f"time taken: {time.time() - start_time}")
            
        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')
        
    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        
        loss = -np.sum(Y*np.log(h + 1e-9)) / m
        error = h - Y
        
        grad = (1/m) * self.softmax_grad(X, error)

        if self.regularization:
            loss += self.regularization(self.W)
            grad += self.regularization.derivation(self.W)
        return loss, grad
    
    def softmax(self, theta_t_x):
        theta_t_x = np.atleast_2d(theta_t_x)
        e_x = np.exp(theta_t_x - np.max(theta_t_x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return  X.T @ error

    def h_theta(self, X, W):
        return self.softmax(X @ W)
    
    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        return np.argmax(self.h_theta(X_test, self.W), axis=1)
    
    def plot(self):
        plt.plot(np.arange(len(self.losses)) , self.losses, label = "Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()
    
    def accuracy(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.sum(y_true == y_pred) / len(y_true)
    
    def precision(self, y_true, y_pred, c):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        TP = np.sum((y_pred == c) & (y_true == c))
        FP = np.sum((y_pred == c) & (y_true != c))
        return TP / (TP + FP) if (TP + FP) != 0 else 0.0

    def recall(self, y_true, y_pred, c):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        TP = np.sum((y_pred == c) & (y_true == c))
        FN = np.sum((y_pred != c) & (y_true == c))
        return TP / (TP + FN) if (TP + FN) != 0 else 0.0

    def f1_score(self, y_true, y_pred, c):
        p = self.precision(y_true, y_pred, c)
        r = self.recall(y_true, y_pred, c)
        return 2 * p * r / (p + r) if (p + r) != 0 else 0.0
    
    def macro_precision(self, y_true, y_pred, num_of_classes):
        return np.mean([self.precision(y_true, y_pred, c) for c in num_of_classes])

    def macro_recall(self, y_true, y_pred, num_of_classes):
        return np.mean([self.recall(y_true, y_pred, c) for c in num_of_classes])

    def macro_f1(self, y_true, y_pred, num_of_classes):
        return np.mean([self.f1_score(y_true, y_pred, c) for c in num_of_classes])
    
    def weighted_precision(self, y_true, y_pred, classes, weights):
        values = [self.precision(y_true, y_pred, c) for c in classes]
        return sum(p * w for p, w in zip(values, weights))

    def weighted_recall(self, y_true, y_pred, classes, weights):
        values = [self.recall(y_true, y_pred, c) for c in classes]
        return sum(r * w for r, w in zip(values, weights))

    def weighted_f1(self, y_true, y_pred, classes, weights):
        values = [self.f1_score(y_true, y_pred, c) for c in classes]
        return sum(f * w for f, w in zip(values, weights))

    def precision_recall_f1_per_class(self, y_true, y_pred):
        classes = np.unique(y_true)
        precision, recall, f1, support = {}, {}, {}, {}
        for c in classes:
            precision[c] = self.precision(y_true, y_pred, c)
            recall[c] = self.recall(y_true, y_pred, c)
            f1[c] = self.f1_score(y_true, y_pred, c)
            support[c] = np.sum(y_true == c)
        return precision, recall, f1, support
    
    def macro_avg(self, precision, recall, f1):
        classes = precision.keys()
        macro_p = np.mean([precision[c] for c in classes])
        macro_r = np.mean([recall[c] for c in classes])
        macro_f = np.mean([f1[c] for c in classes])
        return macro_p, macro_r, macro_f
    
    def weighted_avg(self, precision, recall, f1, support):
        classes = precision.keys()
        total_support = sum(support.values())
        weighted_p = sum(precision[c] * support[c] for c in classes) / total_support
        weighted_r = sum(recall[c] * support[c] for c in classes) / total_support
        weighted_f = sum(f1[c] * support[c] for c in classes) / total_support
        return weighted_p, weighted_r, weighted_f

class RidgePenalty:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.square(theta))

    def derivation(self, theta):
        return self.l * 2 * theta

class Ridge(LogisticRegression):
    def __init__(self, l, k, n, method, alpha=0.001, max_iter=5000):
        regularization = RidgePenalty(l)
        super().__init__(k=k, n=n, method=method, alpha=alpha, max_iter=max_iter, 
                         regularization=regularization, use_penalty=True, lambda_=l)

class Normal(LogisticRegression):
    def __init__(self, k, n, method, alpha=0.001, max_iter=5000):
        super().__init__(k=k, n=n, method=method, alpha=alpha, max_iter=max_iter, 
                         regularization=None, use_penalty=False)
# -------------------------------------------------------------------
# END of Class Definitions
# -------------------------------------------------------------------


# Paths for scaler and encoders
SCALAR_PATH = os.path.join(BASE_DIR, "..", "Model", "car-scaling.model")
BRAND_ENCODER_PATH = os.path.join(BASE_DIR, "..", "Model", "car_brand_encoder.model")
TRANSMISSION_ENCODER_PATH = os.path.join(BASE_DIR, "..", "Model", "car_transmission_encoder.model")

# Load scaler and encoders
try:
    scaler = pickle.load(open(SCALAR_PATH, "rb"))
    brand_encoder = pickle.load(open(BRAND_ENCODER_PATH, "rb"))
    transmission_encoder = pickle.load(open(TRANSMISSION_ENCODER_PATH, "rb"))
except FileNotFoundError as e:
    logging.error(f"Could not load a model file: {e}. Please ensure the 'Model' directory and its contents are correct.")
    scaler, brand_encoder, transmission_encoder = None, None, None

def one_hot_transform(encoder, dataframe, feature):
    """Helper function to apply one-hot encoding from the notebook."""
    encoded = encoder.transform(dataframe[[feature]])
    categories = encoder.get_feature_names_out([feature])
    feature_df = pd.DataFrame(encoded.toarray(), columns=categories)
    
    # We need to drop the original feature column and one of the new dummy columns (drop='first' logic)
    dataframe = dataframe.drop(feature, axis=1)
    # The encoder from the notebook was fitted with drop='first', so we need to replicate that.
    # The column to drop is the first category in the encoder.
    first_category_col = f"brand_{encoder.categories_[0][0]}"
    feature_df = feature_df.drop(columns=[first_category_col], errors='ignore')
    
    # Concatenate the new encoded columns
    concat_dataframe = pd.concat([dataframe.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    return concat_dataframe

# --------------------------
# Prepare feature vector
# --------------------------
def get_X(engine, owner, year, transmission, brand):
    """
    Prepare the feature vector for the ML model.

    Args:
        engine (float/int): Engine capacity
        owner (str): Owner type ('First Owner', 'Second Owner', etc.)
        year (int): Manufacturing year
        transmission (str): 'Manual' or 'Automatic'
        brand (str): Brand name

    Returns:
        pd.DataFrame: Feature vector with proper scaling and encoding
    """

    if not all([scaler, brand_encoder, transmission_encoder]):
        raise RuntimeError("Scaler or encoders are not loaded. Cannot create feature vector.")

    # 1. Map owner string to numeric
    owner_map = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4
    }
    owner_numeric = owner_map.get(owner, 1)

    # 2. Encode transmission using LabelEncoder
    try:
        transmission_numeric = transmission_encoder.transform([transmission])[0]
    except ValueError:
        logging.warning(f"Transmission '{transmission}' not found in encoder classes. Defaulting to 0.")
        transmission_numeric = 0

    # 3. Create initial DataFrame for processing
    df_input = pd.DataFrame([{
        'engine': engine,
        'owner': owner_numeric,
        'year': year,
        'transmission': transmission_numeric,
        'brand': brand
    }])

    # 4. One-hot encode brand (to match training)
    brand_encoded = brand_encoder.transform(df_input[['brand']]).toarray()
    brand_cats = list(brand_encoder.categories_[0][1:])
    df_brand = pd.DataFrame(brand_encoded, columns=brand_cats)

    # Drop raw brand column and join encoded columns
    df_input = df_input.drop(columns=['brand']).reset_index(drop=True)
    df_input = pd.concat([df_input, df_brand], axis=1)

    # 5. Scale numeric columns 
    num_cols = ['engine', 'owner', 'year']
    df_input[num_cols] = scaler.transform(df_input[num_cols])

    # 6. RENAME transmission to transmission_1 to match training data schema
    df_input.rename(columns={'transmission': 'transmission_1'}, inplace=True)
    
    # 7. ***FIX***: Convert the int (0/1) to a boolean (False/True) to match the schema.
    df_input['transmission_1'] = df_input['transmission_1'].astype(bool)
    
    # 8. Reorder columns to match the exact order from the training notebook
    final_cols = ['engine', 'owner', 'year'] + brand_cats + ['transmission_1']
    df_processed = df_input.reindex(columns=final_cols, fill_value=0)

    return df_processed

# --------------------------
# Load MLflow model
# --------------------------
def load_model():
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
    model_name = os.getenv("MODEL_NAME", "st126380-a3-model")
    # model_uri = f"runs:/{run_id}/model" if run_id else f"models:/{model_name}/Production" 
    model_uri = f"runs:/{run_id}/model" if run_id else f"models:/{model_name}@prod"

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

try:
    mlflow_model = load_model()
except Exception as e:
    logging.error(f"MLflow model could not be loaded: {e}")
    mlflow_model = None

# --------------------------
# Predict function
# --------------------------
def predict_selling_price(engine, owner, year, transmission, brand):
    if mlflow_model is None:
        raise RuntimeError("MLflow model is not loaded. Cannot make predictions.")

    X = get_X(engine, owner, year, transmission, brand)
    raw_pred = mlflow_model.predict(X)[0]
    
    class_map = ["Cheap (Lowest 25%)", "Average (25-50%)", "Expensive (50-75%)", "Very Expensive (Top 25%)"]
    label = class_map[int(raw_pred)]
    
    return raw_pred, label

# Example Usage
if __name__ == '__main__':
    example_input = {
        "engine": 1248.0,
        "owner": 'First Owner',
        "year": 2014,
        "transmission": 'Manual',
        "brand": 'Maruti'
    }

    try:
        prediction, label = predict_selling_price(**example_input)
        print("\n" + "="*30)
        print("CAR PRICE PREDICTION")
        print("="*30)
        print(f"Input Features: {example_input}")
        print(f"Predicted Class: {prediction}")
        print(f"Predicted Label: {label}")
        print("="*30)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")