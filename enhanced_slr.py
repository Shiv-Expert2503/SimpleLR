import numpy as np

#Calculatoin values of m and c
def fit(X,y):
    X, y = np.array(X), np.array(y)

    #Len of x and y should not be 0
    if len(X) == 0 or len(y) == 0:
        raise ValueError("X and y cannot be empty")

    #Len of x and y should be equal
    if len(X) != len(y):
        raise ValueError("X and y must have same length")

    # X must have more than 1 unique value as it will create a 0 division error while calculating value of m
    if len(set(X)) == 1:
        raise ValueError("X must contain more than 1 unique value")

    n = len(X)

    m = (n * np.sum(X*y) - np.sum(X) * np.sum(y)) / (n * np.sum(X**2) - (np.sum(X))**2)
    c = (np.sum(y) - m*np.sum(X))/ n

    return m, c

#Calculating y_pred
def predict (X, m, c):
    X = np.array(X)
    return m * X + c

#Calculating cod(r2)
def score(y, y_pred):
    u = np.sum((y - y_pred)**2)
    v = np.sum((y - np.mean(y))**2)
    return 1 - (u/v)

#Calculating MSE
def cost(y, y_pred):
    n = len(y)
    return np.sum((y - y_pred)**2) / n
