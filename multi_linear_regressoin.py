import numpy as np
from typing import List

class MyLinearRegression:
    def __init__(self):
        self.beta = None

    def fit(self, X: List[List[float]], y: List[float]) -> None:
        """
        Fits a multiple linear regression model to the provided training data.
        Parameters:
            X (List[List[float]]): 2D list or array-like of shape (n_samples, n_features) representing the input features.
            y (List[float]): 1D list or array-like of shape (n_samples,) representing the target values.
        Returns:
            None
        Notes:
            - Adds an intercept term to the feature matrix.
            - Computes the regression coefficients (beta) using the normal equation.
            - Uses the pseudo-inverse if the covariance matrix is singular.
            - The computed coefficients are stored in self.beta.
        """
        

        X = np.array(X)
        y = np.array(y)
        X_matrix = np.column_stack((np.ones((X.shape[0],)), X))  
        X_transpose = X_matrix.T
        cov_matrix = np.dot(X_transpose, X_matrix)

        # Use inverse or pseudo-inverse for singular matrices
        try:
            cov_matrix_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            cov_matrix_inv = np.linalg.pinv(cov_matrix)

        cov_with_y = np.dot(X_transpose, y)
        self.beta = np.dot(cov_matrix_inv, cov_with_y)


    def predict(self, X: List[List[float]]) -> np.ndarray:
        """
        Predicts target values using the learned linear regression coefficients.

        Parameters:
            X (List[List[float]]): A 2D list where each sublist represents the feature values for a single sample.

        Returns:
            np.ndarray: Predicted target values as a 1D numpy array.

        Raises:
            ValueError: If the model has not been fitted and coefficients are unavailable.
        """

        if self.beta is None:
            raise ValueError("Model has not been fitted yet.")
        X = np.array(X)
        X_matrix = np.column_stack((np.ones((X.shape[0],)), X))  
        return np.dot(X_matrix, self.beta)


    def cost(self, y_true: List[float], y_pred: np.ndarray) -> float:
        """
        Calculates the mean squared error cost between the true and predicted values.
        Args:
            y_true (List[float]): The list of true target values.
            y_pred (np.ndarray): The array of predicted values.
        Returns:
            float: The computed cost as the mean squared error divided by 2.
        """
        
        y_true = np.array(y_true)
        return np.sum((y_true - y_pred) ** 2) / (2 * len(y_true))

    def score(self, y_true: List[float], y_pred: np.ndarray) -> float:
        """
        Calculates the coefficient of determination (R^2 score) for the given true and predicted values.
        Args:
            y_true (List[float]): The ground truth (actual) target values.
            y_pred (np.ndarray): The predicted target values.
        Returns:
            float: The R^2 score, which indicates how well the predictions approximate the actual values. A score of 1.0 indicates perfect prediction.
        """

        y_true = np.array(y_true)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)

    def get_params(self) -> np.ndarray:
        if self.beta is None:
            raise ValueError("Model has not been fitted yet.")
        return self.beta



# Sample data
X = [
    [1.0, 1],
    [1.5, 2],
    [2.0, 3],
    [2.5, 4],
    [3.0, 5]
]
y = [150, 200, 250, 300, 350]

# Create and use the model
model = MyLinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

print("Coefficients:", model.get_params())
print("Predictions:", y_pred)
print("Cost (MSE/2):", model.cost(y, y_pred))
print("R² Score:", model.score(y, y_pred))
