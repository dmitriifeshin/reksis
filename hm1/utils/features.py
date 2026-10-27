import numpy as np
import pandas as pd
import scipy.sparse as sp


def _clean(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    x = str(x).strip()
    return x or None


def item_tokens(meta, item2idx, top_stores=300, price_bins=8, min_cat_count=20):
    m = meta.drop_duplicates("item_id").set_index("item_id").reindex(item2idx.index)
    tokens = [[] for _ in range(len(m))]

    cats = m["categories"].apply(lambda c: [] if c is None else [x for x in c if x != "Video Games"])
    cat_counts = pd.Series([c for cs in cats for c in set(cs)]).value_counts()
    good_cats = set(cat_counts[cat_counts >= min_cat_count].index)
    for n, cs in enumerate(cats):
        cs = [c for c in dict.fromkeys(cs) if c in good_cats]
        tokens[n] += ["cat=" + c for c in cs] if cs else ["cat=NA"]

    mains = m["main_category"].map(_clean)
    top_main = set(mains.value_counts().index[:10])
    for n, x in enumerate(mains):
        tokens[n].append("main=" + ("NA" if x is None else x if x in top_main else "other"))

    stores = m["store"].map(_clean)
    top = set(stores.value_counts().index[:top_stores])
    for n, x in enumerate(stores):
        tokens[n].append("store=" + ("NA" if x is None else x if x in top else "other"))

    price = m["price"].astype(float)
    q = pd.qcut(price, price_bins, labels=False, duplicates="drop")
    for n, x in enumerate(q):
        tokens[n].append("price=" + ("NA" if np.isnan(x) else f"q{int(x)}"))

    year = pd.to_datetime(m["release_date"]).dt.year
    for n, y in enumerate(year):
        if np.isnan(y):
            t = "NA"
        elif y <= 2005:
            t = "<=2005"
        else:
            t = str(int(y))
        tokens[n].append("release=" + t)

    esrb = m["esrb"].map(_clean)
    top_esrb = set(esrb.value_counts().index[:8])
    for n, x in enumerate(esrb):
        tokens[n].append("esrb=" + ("NA" if x is None else x if x in top_esrb else "other"))

    return tokens


def tokens_to_csr(tokens, prefix=None, vocab=None):
    if prefix is not None:
        tokens = [[t for t in ts if t.startswith(prefix)] for ts in tokens]
    if vocab is None:
        vocab = sorted({t for ts in tokens for t in ts})
    index = {t: n for n, t in enumerate(vocab)}
    rows, cols = [], []
    for r, ts in enumerate(tokens):
        for t in ts:
            if t in index:
                rows.append(r)
                cols.append(index[t])
    X = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(tokens), len(vocab)),
    )
    X.data[:] = 1.0
    return X, vocab
