from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

class LinearRegression:
    # kfold for cross-validation, default 3 folds
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)

    def __init__(self, regularization=None, lr=0.001, method='batch', init='xavier', 
                 polynomial=True, degree=2, use_momentum=True, momentum=0.9, 
                 num_epochs=500, batch_size=50, cv=kfold):
        # basic training params
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method = method  # batch, mini, sto
        self.polynomial = polynomial
        self.degree = degree
        self.init = init
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.prev_step = None  # to store previous update for momentum
        self.cv = cv
        self.regularization = regularization
        self.theta = None  # weights
        self.columns = None  # store column names for later use (feature importance etc)

    def _initialize_weights(self, n_features):
        # xavier or zeros init
        if self.init == 'xavier':
            limit = 1.0 / np.sqrt(n_features)
            return np.random.uniform(-limit, limit, size=(n_features,)).astype(np.float64)
        else:
            return np.zeros(n_features, dtype=np.float64)

    def _add_bias(self, X):
        # just tack on a column of 1s for bias
        return np.c_[np.ones((X.shape[0], 1)), X]

    def mse(self, ytrue, ypred):
        return np.mean((ytrue - ypred) ** 2)

    def r2(self, ytrue, ypred):
        ss_res = np.sum((ytrue - ypred) ** 2)
        ss_tot = np.sum((ytrue - np.mean(ytrue)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def _transform_features(self, X):
        # poly features if enabled
        if self.polynomial:
            poly = PolynomialFeatures(degree=self.degree, include_bias=False)
            return poly.fit_transform(X)
        # fallback: convert df to numpy
        return X.to_numpy() if isinstance(X, pd.DataFrame) else X

    def fit(self, X_train, y_train):
        # save column names
        self.columns = X_train.columns if isinstance(X_train, pd.DataFrame) else [f"X{i}" for i in range(X_train.shape[1])]
        X_train = self._transform_features(X_train).astype(np.float64)
        X_train = self._add_bias(X_train).astype(np.float64)
        y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
        self.kfold_scores = []
        self.kfold_r2 = []

        # loop over folds
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            self.theta = self._initialize_weights(X_tr.shape[1])
            self.prev_step = np.zeros_like(self.theta)

            # mlflow for tracking each fold
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                mlflow.log_params({"method": self.method, "lr": self.lr, "reg": str(self.regularization)})
                for epoch in range(self.num_epochs):
                    # shuffle every epoch
                    perm = np.random.permutation(X_tr.shape[0])
                    X_tr, y_tr = X_tr[perm], y_tr[perm]

                    # update weights based on method
                    if self.method == 'sto':
                        for i in range(X_tr.shape[0]):
                            self._train(X_tr[i].reshape(1, -1), y_tr[i])
                    elif self.method == 'mini':
                        for i in range(0, X_tr.shape[0], self.batch_size):
                            X_batch = X_tr[i:i+self.batch_size]
                            y_batch = y_tr[i:i+self.batch_size]
                            self._train(X_batch, y_batch)
                    else:  # batch
                        self._train(X_tr, y_tr)

                    # validate (simple dot product)
                    y_val_pred = X_val @ self.theta
                    mse_val = self.mse(y_val, y_val_pred)
                    r2_val = self.r2(y_val, y_val_pred)
                    mlflow.log_metric("val_mse", mse_val, step=epoch)
                    mlflow.log_metric("val_r2", r2_val, step=epoch)

                # save final fold metrics
                self.kfold_scores.append(mse_val)
                self.kfold_r2.append(r2_val)
                print(f"Fold {fold}: MSE={mse_val:.4f}, R2={r2_val:.4f}")

    def _train(self, X, y):
        # make sure all float64
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        y_pred = X @ self.theta
        m = X.shape[0]
        grad = (X.T @ (y_pred - y)) / m

        # regularization contribution
        if self.regularization:
            grad += self.regularization.derivation(self.theta).astype(np.float64)

        # momentum update
        if self.use_momentum:
            step = (-self.lr * grad + self.momentum * self.prev_step).astype(np.float64)
            self.theta = (self.theta + step).astype(np.float64)
            self.prev_step = step
        else:
            self.theta = (self.theta - self.lr * grad).astype(np.float64)

        return self.mse(y, y_pred)

    def predict(self, X):
        # transform + bias before prediction
        X_trans = self._transform_features(X).astype(np.float64)
        X_trans = self._add_bias(X_trans).astype(np.float64)
        return X_trans @ self.theta

    def _coef(self):
        return self.theta[1:]  # skip bias

    def _bias(self):
        return self.theta[0]

    def feature_importance(self, top_n=20, width=10, height=8):

        if self.theta is None:
            print("Model has not been trained yet. Please call fit() or fit_final().")
            return

        coefs = self._coef()

        # Get the correct feature names
        if self.polynomial and self.poly_feature_names is not None:
            feature_names = self.poly_feature_names
        else:
            feature_names = self.columns

        # Ensure feature_names and coefficients align
        if len(coefs) != len(feature_names):
            print(f"Mismatch between number of coefficients ({len(coefs)}) and feature names ({len(feature_names)}). Using generic names.")
            feature_names = [f"Feature_{i}" for i in range(len(coefs))]

        # Calculate importance and get the top N
        importance = pd.DataFrame(data=np.abs(coefs), index=feature_names, columns=['Importance'])
        top_importance = importance.sort_values(by='Importance', ascending=False).head(top_n)

        # Plotting
        plt.figure(figsize=(width, height))
        
        # Plot horizontal bar chart for better label readability
        plt.barh(top_importance.index, top_importance['Importance'])
        
        # Invert y-axis to have the most important feature on top
        plt.gca().invert_yaxis()
        
        plt.title(f"Top {top_n} Feature Importances")
        plt.xlabel("Absolute Coefficient Value (Importance)")
        plt.tight_layout() # Adjust layout to make room for labels
        plt.show()

    def fit_final(self, X_train, y_train):
        # final training on full data
        self.columns = X_train.columns if isinstance(X_train, pd.DataFrame) else [f"X{i}" for i in range(X_train.shape[1])]
        X_tr = self._transform_features(X_train).astype(np.float64)
        X_tr = self._add_bias(X_tr).astype(np.float64)
        y_tr = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

        # init weights + momentum
        self.theta = self._initialize_weights(X_tr.shape[1])
        self.prev_step = np.zeros_like(self.theta)

        print(f"Starting final training for {self.num_epochs} epochs...")
        for epoch in range(self.num_epochs):
            # shuffle data every epoch
            perm = np.random.permutation(X_tr.shape[0])
            X_tr_shuffled, y_tr_shuffled = X_tr[perm], y_tr[perm]

            # train based on method
            if self.method == 'sto':
                for i in range(X_tr_shuffled.shape[0]):
                    self._train(X_tr_shuffled[i].reshape(1, -1), y_tr_shuffled[i])
            elif self.method == 'mini':
                for i in range(0, X_tr_shuffled.shape[0], self.batch_size):
                    X_batch = X_tr_shuffled[i:i+self.batch_size]
                    y_batch = y_tr_shuffled[i:i+self.batch_size]
                    self._train(X_batch, y_batch)
            else:  # batch
                self._train(X_tr_shuffled, y_tr_shuffled)
                
        print("Final training complete.")

# ------------------ Regularization Penalty Classes ------------------
class LassoPenalty:
    # L1 penalty, pushes some weights to zero
    def __init__(self, l=0.1):
        self.l = l
        
    def __call__(self, theta):
        # compute penalty value (not used in grad)
        return self.l * np.sum(np.abs(theta))
    
    def derivation(self, theta):
        # gradient of L1 for updating weights
        return self.l * np.sign(theta)

class RidgePenalty:
    # L2 penalty, shrinks weights but doesn't zero them
    def __init__(self, l=0.1):
        self.l = l
        
    def __call__(self, theta):
        return self.l * np.sum(theta ** 2)
    
    def derivation(self, theta):
        return 2 * self.l * theta  # gradient of L2

class ElasticPenalty:
    # combo of L1 + L2, controlled by l_ratio
    def __init__(self, l=0.1, l_ratio=0.5):
        self.l = l
        self.l_ratio = l_ratio
        
    def __call__(self, theta):
        # mix L1 + 0.5*L2 (0.5 for scaling)
        return self.l * (self.l_ratio * np.sum(np.abs(theta)) + 0.5*(1-self.l_ratio)*np.sum(theta**2))
    
    def derivation(self, theta):
        # gradient mix of L1 + L2
        return self.l * (self.l_ratio * np.sign(theta) + (1-self.l_ratio) * theta)

# ------------------ Linear Regression Wrappers ------------------
class Lasso(LinearRegression):
    # wrapper to use LassoPenalty with main LinearRegression
    def __init__(self, l=0.1, **kwargs):
        super().__init__(regularization=LassoPenalty(l), **kwargs)

class Ridge(LinearRegression):
    # wrapper to use RidgePenalty
    def __init__(self, l=0.1, **kwargs):
        super().__init__(regularization=RidgePenalty(l), **kwargs)

class ElasticNet(LinearRegression):
    # wrapper to use ElasticPenalty
    def __init__(self, l=0.1, l_ratio=0.5, **kwargs):
        super().__init__(regularization=ElasticPenalty(l, l_ratio), **kwargs)

class Normal(LinearRegression):
    # plain linear regression, no regularization
    def __init__(self, **kwargs):
        super().__init__(regularization=None, **kwargs)