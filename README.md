# Simple Linear regression
### Approach
1. We need total 4 function to implement SLR. That is .fit(), .predict(), .cost(). .score()
2. For fir function we need values of X and y and then since cost should me minimum. Therefore differentiating it with respect to m and c will give formula of m and c.
3. Therefore via that formula we get m and c. Now we return them
4. Then in predict function we use x, m, c to find y_pred then return it
5. Then in score function we calculate the score(r**2) which is given by 1-(u/v)
6. Then at last cost is give by MSE(mean squared error)

### This approach is good but it is not properly optimized
#### X and y must be of same length
#### length of X and y must not be 0
#### X should contain more than 1 unique value as if not then it will generate the 0 division error while calculating value of m in fit function. As while calculating m the denominator will become 0

------------------------------------------
SLR is failing for 

X = [5, 5, 5, 5]  
y = [30, 35, 40, 45]
------------------------------------------
#### Therefore Enhanced SLR came into picture
### Approach for EnhancedSLR
1. Added check for length of x and y
2. Added check that x must have more than 1 unique value


# Multi Linear Regression

**Detailed Key Points of the `fit` Function:**

- **Purpose:**  
    The `fit` function estimates the regression coefficients (weights) for multiple linear regression using the provided training data.

- **Input Handling:**  
    - Accepts `X` (2D list or array-like of shape `(n_samples, n_features)`) as input features.  
    - Accepts `y` (1D list or array-like of shape `(n_samples,)`) as target values.  
    - Converts `X` and `y` to NumPy arrays for efficient computation.

- **Intercept Addition:**  
    - Adds a column of ones to `X` to account for the intercept (bias) term in the regression model.

- **Normal Equation:**  
    - Uses the normal equation:  
        \[
        beta = (X^T X)^{-1} X^T y
        \]
        where \(\beta\) is the vector of regression coefficients.

- **Singular Matrix Handling:**  
    - Attempts to compute the inverse of \(X^T X\).  
    - If \(X^T X\) is singular (non-invertible), uses the pseudo-inverse to ensure the computation proceeds without error.

- **Coefficient Storage:**  
    - Stores the computed coefficients (including the intercept) in `self.beta` for later use in prediction.

- **No Return Value:**  
    - The function modifies the model in-place and does not return any value.

- **Robustness:**  
    - Handles edge cases where the feature matrix may not be invertible, making the implementation more robust for real-world data.

**Detailed Key Points of the `predict`, `score`, and `cost` Functions:**

- **`predict` Function:**
    - **Purpose:**  
        Generates predicted target values using the learned regression coefficients.
    - **Input Handling:**  
        - Accepts `X` (2D list or array-like of shape `(n_samples, n_features)`) as input features.
        - Converts `X` to a NumPy array and adds a column of ones for the intercept.
    - **Prediction Calculation:**  
        - Computes predictions by multiplying the feature matrix (with intercept) by the coefficient vector (`self.beta`).
    - **Error Handling:**  
        - Raises a `ValueError` if the model has not been fitted and coefficients are unavailable.
    - **Output:**  
        - Returns predicted values as a 1D NumPy array.

- **`cost` Function:**
    - **Purpose:**  
        Calculates the mean squared error (MSE) cost between the true and predicted values.
    - **Input Handling:**  
        - Accepts `y_true` (list or array of true target values) and `y_pred` (array of predicted values).
        - Converts `y_true` to a NumPy array for computation.
    - **Computation:**  
        - Computes the sum of squared differences between true and predicted values.
        - Divides the result by twice the number of samples to obtain the cost.
    - **Output:**  
        - Returns the computed cost as a float.

- **`score` Function:**
    - **Purpose:**  
        Computes the coefficient of determination (R² score) to evaluate model performance.
    - **Input Handling:**  
        - Accepts `y_true` (actual target values) and `y_pred` (predicted values).
        - Converts `y_true` to a NumPy array.
    - **Computation:**  
        - Calculates the residual sum of squares (`ss_res`) and the total sum of squares (`ss_tot`).
        - Computes the R² score as \(1 - \frac{ss\_res}{ss\_tot}\).
    - **Output:**  
        - Returns the R² score as a float, where 1.0 indicates perfect prediction.

