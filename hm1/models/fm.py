import copy

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def _normalize_rows(X):
    X = sp.csr_matrix(X, dtype=np.float32)
    s = np.asarray(X.sum(axis=1)).ravel()
    s[s == 0] = 1.0
    return sp.diags(1.0 / s) @ X


class QueryEncoder:
    def __init__(self, n_users, item_profile=None, years=None, use_user_id=True):
        self.n_users = n_users if use_user_id else 0
        self.item_profile = None if item_profile is None else sp.csr_matrix(item_profile, dtype=np.float32)
        self.years = None if years is None else np.asarray(years)
        self.n_years = 0 if years is None else len(self.years)
        self.n_profile = 0 if item_profile is None else self.item_profile.shape[1]
        self.n_features = self.n_users + self.n_years + self.n_profile

    def _year_block(self, ts):
        y = pd.DatetimeIndex(ts).year.values
        idx = np.clip(np.searchsorted(self.years, y), 0, self.n_years - 1)
        return sp.csr_matrix(
            (np.ones(len(idx), dtype=np.float32), (np.arange(len(idx)), idx)),
            shape=(len(idx), self.n_years),
        )

    def _user_block(self, uid):
        return sp.csr_matrix(
            (np.ones(len(uid), dtype=np.float32), (np.arange(len(uid)), uid)),
            shape=(len(uid), self.n_users),
        )

    def _assemble(self, uid, ts, profile):
        blocks = []
        if self.n_users:
            blocks.append(self._user_block(uid))
        if self.n_years:
            blocks.append(self._year_block(ts))
        if self.n_profile:
            blocks.append(profile)
        return sp.hstack(blocks, format="csr", dtype=np.float32)

    def train_queries(self, df):
        df = df.reset_index(drop=True)
        profile = None
        if self.n_profile:
            order = np.lexsort((df["timestamp"].values, df["uid"].values))
            uid = df["uid"].values[order]
            C = self.item_profile[df["iid"].values[order]].toarray()
            prev = np.cumsum(C, axis=0, dtype=np.float32)
            prev -= C
            del C
            starts = np.r_[0, np.flatnonzero(np.diff(uid)) + 1]
            group = np.repeat(starts, np.diff(np.r_[starts, len(uid)]))
            prev -= prev[group]
            P = np.empty_like(prev)
            P[order] = prev
            del prev
            profile = _normalize_rows(sp.csr_matrix(P))
        return self._assemble(df["uid"].values, df["timestamp"].values, profile)

    def inference_queries(self, history, n_users, t_ref):
        profile = None
        if self.n_profile:
            H = sp.csr_matrix(
                (np.ones(len(history), dtype=np.float32), (history["uid"].values, history["iid"].values)),
                shape=(n_users, self.item_profile.shape[0]),
            )
            profile = _normalize_rows(H @ self.item_profile)
        uid = np.arange(n_users)
        ts = np.full(n_users, np.datetime64(pd.Timestamp(t_ref)))
        return self._assemble(uid, ts, profile)


class _FM(torch.nn.Module):
    def __init__(self, n_query, n_item, dim):
        super().__init__()
        self.q_v = torch.nn.Embedding(n_query, dim)
        self.i_w = torch.nn.Embedding(n_item, 1)
        self.i_v = torch.nn.Embedding(n_item, dim)
        torch.nn.init.normal_(self.q_v.weight, std=0.01)
        torch.nn.init.normal_(self.i_v.weight, std=0.01)
        torch.nn.init.zeros_(self.i_w.weight)

    @staticmethod
    def _bag(weight, X):
        return F.embedding_bag(X[0], weight, X[1], mode="sum", per_sample_weights=X[2])

    def query(self, X):
        v = self._bag(self.q_v.weight, X)
        sq = F.embedding_bag(X[0], self.q_v.weight ** 2, X[1], mode="sum", per_sample_weights=X[2] ** 2)
        return v, sq

    def items(self, X):
        v = self._bag(self.i_v.weight, X)
        sq = F.embedding_bag(X[0], self.i_v.weight ** 2, X[1], mode="sum", per_sample_weights=X[2] ** 2)
        w = self._bag(self.i_w.weight, X).squeeze(1)
        bias = w + 0.5 * ((v ** 2).sum(1) - sq.sum(1))
        return v, bias, sq


def _to_bag(X):
    X = sp.csr_matrix(X)
    return (
        torch.from_numpy(X.indices.astype(np.int64)),
        torch.from_numpy(X.indptr[:-1].astype(np.int64)),
        torch.from_numpy(X.data.astype(np.float32)),
    )


class FMRecommender:
    def __init__(self, dim=64, lr=1e-3, reg=1e-4, epochs=10, batch_size=2048, n_neg=4, seed=42):
        self.dim = dim
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_neg = n_neg
        self.seed = seed

    def fit(self, Q, pos_items, item_X, candidates, callback=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Q = sp.csr_matrix(Q, dtype=np.float32)
        pos_items = np.asarray(pos_items)
        cand = np.flatnonzero(candidates)
        self.item_bag = _to_bag(item_X)
        self.model = _FM(Q.shape[1], item_X.shape[1], self.dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.history = []
        best, best_state = -np.inf, None
        n = Q.shape[0]
        for epoch in range(self.epochs):
            self.model.train()
            perm = rng.permutation(n)
            total = 0.0
            for s in range(0, n, self.batch_size):
                b = perm[s:s + self.batch_size]
                qv, qsq = self.model.query(_to_bag(Q[b]))
                iv, ib, isq = self.model.items(self.item_bag)
                pos = torch.from_numpy(pos_items[b])
                neg = torch.from_numpy(cand[rng.integers(0, len(cand), size=(len(b), self.n_neg))])
                sp_ = (qv * iv[pos]).sum(1) + ib[pos]
                sn = (qv[:, None, :] * iv[neg]).sum(2) + ib[neg]
                loss = -F.logsigmoid(sp_[:, None] - sn).mean()
                l2 = qsq.sum() + isq[pos].sum() + isq[neg].sum() / self.n_neg
                loss = loss + self.reg * l2 / len(b)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            record = {"epoch": epoch + 1, "loss": total / n}
            if callback is not None:
                self._cache_items()
                metric = callback(self)
                record["metric"] = metric
                if metric > best:
                    best, best_state = metric, copy.deepcopy(self.model.state_dict())
                    self.best_epoch = epoch + 1
            self.history.append(record)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._cache_items()
        return self

    def _cache_items(self):
        self.model.eval()
        with torch.no_grad():
            iv, ib, _ = self.model.items(self.item_bag)
        self.item_v = iv.numpy()
        self.item_b = ib.numpy()

    def set_queries(self, Q):
        self.Q_inf = sp.csr_matrix(Q, dtype=np.float32)
        return self

    def score(self, users):
        with torch.no_grad():
            qv, _ = self.model.query(_to_bag(self.Q_inf[users]))
        return qv.numpy() @ self.item_v.T + self.item_b
