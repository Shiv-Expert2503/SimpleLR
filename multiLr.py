import numpy as np

def fit(X, y):
    '''
    :param X: independent variables (2D array, shape [n_sample, n_features])
    :param y: dependent variables (1D array, shape [n_sample])
    :return: beta: np-array Coefficients of linear regression
    '''

    x = np.array(X)
    y = np.array(y)
    # print(x.shape)

    # to ensure that x is 2D and y is 1D
    if x.ndim != 2:
        raise ValueError('X must be a 2D array')
    if y.ndim != 1:
        raise ValueError('y must be a 1D array')

    X_matrix = np.column_stack((np.ones((x.shape[0])), x))
    X_trans_matrix = X_matrix.transpose()

    cov_matrix = np.dot(X_trans_matrix, X_matrix)

    try:
        cov_matrix_inv = np.linalg.inv(cov_matrix)

    except np.linalg.LinAlgError:
        cov_matrix_inv = np.linalg.pinv(cov_matrix) # Using pseudo inverse in case of singularity

    cov_bw_dep_indep_matrix = np.dot(X_trans_matrix, y)

    beta = np.dot(cov_matrix_inv, cov_bw_dep_indep_matrix)
    return beta


def predict(X, beta):
