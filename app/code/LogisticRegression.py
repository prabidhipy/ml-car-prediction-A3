import numpy as np

class LogisticRegression:
    

    def __init__(self, regularization, k, n, method, alpha=0.001, max_iter=5000):
        self.regularization = regularization
        self.k = int(k)
        self.n = int(n)
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.W = None  # Will be initialized later
        self.losses = []

    def fit(self, X, Y):
        """
        Fit the logistic regression model to the training data.

        Args:
            X (numpy.ndarray): Training data features.
            Y (numpy.ndarray): Training data labels.

        Returns:
            None
        """
        self.W = np.random.rand(self.n, self.k)
        params = {
            "reg": type(self).__name__,
            "method": self.method,
            "k": int(self.k),
            "n": int(self.n),
            "alpha": self.alpha,
            "max_iter": self.max_iter
        }
        mlflow.log_params(params=params)

        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad = self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
                    mlflow.log_metric(key="train_loss", value=loss, step=i)
            print(f"time taken: {time.time() - start_time}")

        elif self.method == "minibatch":
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0])  # With replacement
                batch_X = X[ix:ix + batch_size]
                batch_Y = Y[ix:ix + batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
                    mlflow.log_metric(key="train_loss", value=loss, step=i)
            print(f"time taken: {time.time() - start_time}")

        elif self.method == "sto":
            start_time = time.time()
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while i in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad

                list_of_used_ix.append(i)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
                    mlflow.log_metric(key="train_loss", value=loss, step=i)
            print(f"time taken: {time.time() - start_time}")

        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')

    def gradient(self, X, Y):
        """
        Compute the gradient and loss for the logistic regression model.

        Args:
            X (numpy.ndarray): Input data features.
            Y (numpy.ndarray): Input data labels.

        Returns:
            float: Loss.
            numpy.ndarray: Gradient.
        """
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        loss = -np.sum(Y * np.log(h)) / m
        error = h - Y

        if self.regularization:
            grad = self.softmax_grad(X, error) + self.regularization.derivation(self.W)
        else:
            grad = self.softmax_grad(X, error)

        return loss, grad

    def softmax(self, theta_t_x):
        """
        Compute the softmax probabilities.

        Args:
            theta_t_x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Softmax probabilities.
        """
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        """
        Compute the gradient for softmax regression.

        Args:
            X (numpy.ndarray): Input data features.
            error (numpy.ndarray): Error.

        Returns:
            numpy.ndarray: Gradient.
        """
        return X.T @ error

    def h_theta(self, X, W):
        """
        Compute the predicted probabilities.

        Args:
            X (numpy.ndarray): Input data features.
            W (numpy.ndarray): Model weights.

        Returns:
            numpy.ndarray: Predicted probabilities.
        """
        return self.softmax(X @ W)

    def predict(self, X_test):
        """
        Predict class labels for input data.

        Args:
            X_test (numpy.ndarray): Input data for prediction.

        Returns:
            numpy.ndarray: Predicted class labels.
        """
        return np.argmax(self.h_theta(X_test, self.W), axis=1)

    def plot(self):
        """
        Plot the training losses.

        Returns:
            None
        """
        plt.plot(np.arange(len(self.losses)), self.losses, label="Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()

    def accuracy(self, y_test, y_pred):
        """
        Compute classification accuracy.

        Args:
            y_test (numpy.ndarray): True class labels.
            y_pred (numpy.ndarray): Predicted class labels.

        Returns:
            float: Accuracy.
        """
        correct_predictions = np.sum(y_test == y_pred)
        total_predictions = y_test.shape[0]
        return correct_predictions / total_predictions

    def precision(self, y_test, y_pred, c=0):
        """
        Compute precision for a specific class `c`.

        Args:
            y_test (numpy.ndarray): True class labels.
            y_pred (numpy.ndarray): Predicted class labels.

        Returns:
            float: Precision.
        """
        true_positives = np.sum((y_test == c) & (y_pred == c))
        false_positives = np.sum((y_test != c) & (y_pred == c))
        if true_positives + false_positives == 0:
            return 0
        else:
            return true_positives / (true_positives + false_positives)

    def recall(self, y_test, y_pred, c=0):
        """
        Compute recall for a specific class `c`.

        Args:
            y_test (numpy.ndarray): True class labels.
            y_pred (numpy.ndarray): Predicted class labels.

        Returns:
            float: Recall.
        """
        true_positives = np.sum((y_test == c) & (y_pred == c))
        false_negatives = np.sum((y_test == c) & (y_pred != c))
        if true_positives + false_negatives == 0:
            return 0
        else:
            return true_positives / (true_positives + false_negatives)

    def f1_score(self, y_test, y_pred, c=0):
        """
        Compute F1-score for a specific class `c`.

        Args:
            y_test (numpy.ndarray): True class labels.
            y_pred (numpy.ndarray): Predicted class labels.

        Returns:
            float: F1-score.
        """
        precision = self.precision(y_test, y_pred, c)
        recall = self.recall(y_test, y_pred, c)
        if precision + recall == 0:
            return 0
        else:
            return 2 * precision * recall / (precision + recall)

    def macro_precision(self, y_test, y_pred):
        """
        Compute macro-averaged precision.

        Args:
            y_test (numpy.ndarray): True class labels.
            y_pred (numpy.ndarray): Predicted class labels.

        Returns:
            float: Macro-averaged precision.
        """
        precisions = [self.precision(y_test, y_pred, c) for c in range(self.k)]
        return np.sum(precisions) / self.k

    def macro_recall(self, y_test, y_pred):
        """
        Compute macro-averaged recall.

        Args:
            y_test (numpy.ndarray): True class labels.
            y_pred (numpy.ndarray): Predicted class labels.

        Returns:
            float: Macro-averaged recall.
        """
        recalls = [self.recall(y_test, y_pred, c) for c in range(self.k)]
        return np.sum(recalls) / self.k

    def macro_f1(self, y_test, y_pred):
        """
        Compute macro-averaged F1-score.

        Args:
            y_test (numpy.ndarray): True class labels.
            y_pred (numpy.ndarray): Predicted class labels.

        Returns:
            float: Macro-averaged F1-score.
        """
        f1s = [self.f1_score(y_test, y_pred, c) for c in range(self.k)]
        return np.sum(f1s) / self.k

    def weighted_precision(self, y_test, y_pred):
        """
        Compute weighted precision.

        Args:
            y_test (numpy.ndarray): True class labels.
            y_pred (numpy.ndarray): Predicted class labels.

        Returns:
            float: Weighted precision.
        """
        class_counts = [np.count_nonzero(y_test == c) for c in range(self.k)]
        precisions = [class_counts[c] / len(y_test) * self.precision(y_test, y_pred, c) for c in range(self.k)]
        return np.sum(precisions)

    def weighted_recall(self, y_test, y_pred):
        """
        Compute weighted recall.

        Args:
            y_test (numpy.ndarray): True class labels.
            y_pred (numpy.ndarray): Predicted class labels.

        Returns:
            float: Weighted recall.
        """
        class_counts = [np.count_nonzero(y_test == c) for c in range(self.k)]
        recalls = [class_counts[c] / len(y_test) * self.recall(y_test, y_pred, c) for c in range(self.k)]
        return np.sum(recalls)

    def weighted_f1(self, y_test, y_pred):
        """
        Compute weighted F1-score.

        Args:
            y_test (numpy.ndarray): True class labels.
            y_pred (numpy.ndarray): Predicted class labels.

        Returns:
            float: Weighted F1-score.
        """
        class_counts = [np.count_nonzero(y_test == c) for c in range(self.k)]
        f1s = [class_counts[c] / len(y_test) * self.f1_score(y_test, y_pred, c) for c in range(self.k)]
        return np.sum(f1s)

    def classification_report(self, y_test, y_pred):
        """
        Generate a classification report.

        Args:
            y_test (numpy.ndarray): True class labels.
            y_pred (numpy.ndarray): Predicted class labels.

        Returns:
            pandas.DataFrame: Classification report.
        """
        cols = ["precision", "recall", "f1-score"]
        idx = list(range(self.k)) + ["accuracy", "macro", "weighted"]

        report = [[self.precision(y_test, y_pred, c),
                   self.recall(y_test, y_pred, c),
                   self.f1_score(y_test, y_pred, c)] for c in range(self.k)]

        report.append(["", "", self.accuracy(y_test, y_pred)])

        report.append([self.macro_precision(y_test, y_pred),
                       self.macro_recall(y_test, y_pred),
                       self.macro_f1(y_test, y_pred)])

        report.append([self.weighted_precision(y_test, y_pred),
                       self.weighted_recall(y_test, y_pred),
                       self.weighted_f1(y_test, y_pred)])

        return pd.DataFrame(report, index=idx, columns=cols)

class RidgePenalty:
    """Ridge penalty (L2 regularization) for logistic regression.

    Args:
        l (float): Regularization strength.
    """

    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        """Compute the Ridge penalty term.

        Args:
            theta (numpy.ndarray): Model parameters.

        Returns:
            float: Ridge penalty term.
        """
        return self.l * np.sum(np.square(theta))

    def derivation(self, theta):
        """Compute the derivative of the Ridge penalty.

        Args:
            theta (numpy.ndarray): Model parameters.

        Returns:
            numpy.ndarray: Derivative of the Ridge penalty.
        """
        return self.l * 2 * theta

class Ridge(LogisticRegression):
    """Logistic Regression with Ridge (L2) regularization.

    Args:
        l (float): Regularization strength.
        k (int): Number of classes.
        n (int): Number of features.
        method (str): Optimization method ('batch', 'minibatch', or 'sto').
        alpha (float, optional): Learning rate (default is 0.001).
        max_iter (int, optional): Maximum number of iterations (default is 5000).
    """

    def __init__(self, l, k, n, method, alpha=0.001, max_iter=5000):
        regularization = RidgePenalty(l)
        super().__init__(regularization, k, n, method, alpha, max_iter)

class Normal(LogisticRegression):
    """Logistic Regression without regularization.

    Args:
        k (int): Number of classes.
        n (int): Number of features.
        method (str): Optimization method ('batch', 'minibatch', or 'sto').
        alpha (float, optional): Learning rate (default is 0.001).
        max_iter (int, optional): Maximum number of iterations (default is 5000).
    """

    def __init__(self, k, n, method, alpha=0.001, max_iter=5000):
        super().__init__(regularization=None, k=k, n=n, method=method, alpha=alpha, max_iter=max_iter)