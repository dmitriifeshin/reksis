import numpy as np
import scipy.sparse as sp


class ItemKNN:
    def __init__(self, k=100, shrink=10.0, chunk=2048):
        self.k = k
        self.shrink = shrink
        self.chunk = chunk

    def fit(self, X):
        X = sp.csr_matrix(X, dtype=np.float32)
        X.data[:] = 1.0
        self.X = X
        n_items = X.shape[1]
        Xc = X.tocsc()
        norms = np.sqrt(np.asarray(X.sum(axis=0)).ravel())
        k = min(self.k, n_items - 1)
        rows, cols, vals = [], [], []
        for s in range(0, n_items, self.chunk):
            e = min(s + self.chunk, n_items)
            co = (Xc.T @ Xc[:, s:e]).toarray()
            sim = co / (np.outer(norms, norms[s:e]) + self.shrink + 1e-9)
            sim[np.arange(s, e), np.arange(e - s)] = 0.0
            top = np.argpartition(-sim, k, axis=0)[:k]
            v = np.take_along_axis(sim, top, axis=0)
            mask = v > 0
            rows.append(top[mask])
            cols.append(np.broadcast_to(np.arange(s, e), top.shape)[mask])
            vals.append(v[mask])
        self.W = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_items, n_items),
            dtype=np.float32,
        )
        return self

    def score(self, users):
        return (self.X[users] @ self.W).toarray()
