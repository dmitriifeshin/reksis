import numpy as np
import scipy.sparse as sp


class _Side:
    def __init__(self, rows, cols, w, b, n_rows, n_cols):
        order = np.lexsort((cols, rows))
        self.rows = rows[order]
        self.cols = cols[order]
        self.w = w[order].astype(np.float32)
        self.b = b[order].astype(np.float32)
        self.indptr = np.concatenate([[0], np.cumsum(np.bincount(self.rows, minlength=n_rows))])
        self.shape = (n_rows, n_cols)

    def matrix(self, values):
        return sp.csr_matrix((values, self.cols, self.indptr), shape=self.shape)


class IALS:
    def __init__(self, factors=64, reg=1.0, alpha=10.0, iterations=15, cg_steps=3, seed=42):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.cg_steps = cg_steps
        self.seed = seed

    def fit(self, uid, iid, n_users, n_items, pref=None, weight=None):
        uid = np.asarray(uid)
        iid = np.asarray(iid)
        pref = np.ones(len(uid), dtype=np.float32) if pref is None else np.asarray(pref, dtype=np.float32)
        weight = np.ones(len(uid), dtype=np.float32) if weight is None else np.asarray(weight, dtype=np.float32)
        conf = 1.0 + self.alpha * weight
        users = _Side(uid, iid, conf - 1.0, conf * pref, n_users, n_items)
        items = _Side(iid, uid, conf - 1.0, conf * pref, n_items, n_users)
        rng = np.random.default_rng(self.seed)
        self.user_factors = (rng.standard_normal((n_users, self.factors)) * 0.01).astype(np.float32)
        self.item_factors = (rng.standard_normal((n_items, self.factors)) * 0.01).astype(np.float32)
        for _ in range(self.iterations):
            self.user_factors = self._cg(self.user_factors, self.item_factors, users)
            self.item_factors = self._cg(self.item_factors, self.user_factors, items)
        return self

    def _cg(self, X, Y, side):
        A0 = Y.T @ Y + self.reg * np.eye(self.factors, dtype=np.float32)
        Yc = Y[side.cols]

        def mul(V):
            t = np.einsum("ij,ij->i", V[side.rows], Yc) * side.w
            return V @ A0 + side.matrix(t) @ Y

        X = X.copy()
        R = side.matrix(side.b) @ Y - mul(X)
        P = R.copy()
        rs = (R * R).sum(1)
        for _ in range(self.cg_steps):
            AP = mul(P)
            a = rs / np.maximum((P * AP).sum(1), 1e-12)
            X += a[:, None] * P
            R -= a[:, None] * AP
            rs_new = (R * R).sum(1)
            P = R + (rs_new / np.maximum(rs, 1e-12))[:, None] * P
            rs = rs_new
        return X

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
