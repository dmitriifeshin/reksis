import numpy as np
import scipy.sparse as sp


class EASE:
    def __init__(self, reg=500.0):
        self.reg = reg

    def fit(self, X, G=None):
        X = sp.csr_matrix(X, dtype=np.float32)
        X.data[:] = 1.0
        self.X = X
        if G is None:
            G = gram(X)
        P = G.copy()
        P[np.diag_indices_from(P)] += self.reg
        P = np.linalg.inv(P)
        B = P / (-np.diag(P))
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)
        return self

    def score(self, users):
        return self.X[users] @ self.B


def gram(X):
    X = sp.csr_matrix(X, dtype=np.float64)
    X.data[:] = 1.0
    return (X.T @ X).toarray()
