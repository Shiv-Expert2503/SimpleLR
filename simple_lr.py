import numpy as np

#Calculatoin values of m and c
def fit(X,y):
    X, y = np.array(X), np.array(y)
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
