import numpy as np


def recommend(model, users, seen, candidates, k=10, batch_size=2048):
    users = np.asarray(users)
    recs = np.empty((len(users), k), dtype=np.int64)
    for s in range(0, len(users), batch_size):
        u = users[s:s + batch_size]
        scores = np.array(model.score(u), dtype=np.float32, copy=True)
        scores[:, ~candidates] = -np.inf
        rows, cols = seen[u].nonzero()
        scores[rows, cols] = -np.inf
        top = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
        recs[s:s + len(u)] = np.take_along_axis(top, order, axis=1)
    return recs


def ndcg_at_k(recs, truth, k=10):
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    res = np.zeros(len(recs))
    for n, (r, t) in enumerate(zip(recs, truth)):
        rel = np.isin(r[:k], t)
        idcg = discounts[:min(len(t), k)].sum()
        res[n] = (rel * discounts).sum() / idcg
    return res


def recall_at_k(recs, truth, k=10):
    return np.array([np.isin(r[:k], t).sum() / min(len(t), k) for r, t in zip(recs, truth)])


def hitrate_at_k(recs, truth, k=10):
    return np.array([np.isin(r[:k], t).any() for r, t in zip(recs, truth)], dtype=float)


def coverage_at_k(recs, n_candidates, k=10):
    return len(np.unique(recs[:, :k])) / n_candidates


def evaluate(recs, truth, n_candidates, k=10):
    return {
        f"NDCG@{k}": ndcg_at_k(recs, truth, k).mean(),
        f"Recall@{k}": recall_at_k(recs, truth, k).mean(),
        f"HitRate@{k}": hitrate_at_k(recs, truth, k).mean(),
        f"Coverage@{k}": coverage_at_k(recs, n_candidates, k),
    }


def evaluate_model(model, users, truth, seen, candidates, k=10):
    recs = recommend(model, users, seen, candidates, k)
    return evaluate(recs, truth, candidates.sum(), k), recs
