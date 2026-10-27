import json
import os
import urllib.request

import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE_URL = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main"
RATINGS_URL = f"{BASE_URL}/benchmark/5core/rating_only/Video_Games.csv"
META_URL = f"{BASE_URL}/raw/meta_categories/meta_Video_Games.jsonl"


def download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def load_ratings(data_dir="data"):
    path = download(RATINGS_URL, os.path.join(data_dir, "Video_Games.csv"))
    df = pd.read_csv(path)
    df = df.rename(columns={"parent_asin": "item_id"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def _parse_price(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_meta(data_dir="data", item_ids=None):
    cache = os.path.join(data_dir, "meta_Video_Games.parquet")
    if not os.path.exists(cache):
        path = download(META_URL, os.path.join(data_dir, "meta_Video_Games.jsonl"))
        keep = None if item_ids is None else set(item_ids)
        rows = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if keep is not None and r["parent_asin"] not in keep:
                    continue
                d = r.get("details") or {}
                rows.append({
                    "item_id": r["parent_asin"],
                    "title": r.get("title"),
                    "main_category": r.get("main_category"),
                    "categories": list(r.get("categories") or []),
                    "store": r.get("store"),
                    "price": _parse_price(r.get("price")),
                    "release_date": d.get("Release date") or d.get("Date First Available"),
                    "esrb": d.get("Rated"),
                    "n_features": len(r.get("features") or []),
                    "n_description": len(r.get("description") or []),
                    "average_rating": r.get("average_rating"),
                    "rating_number": r.get("rating_number"),
                })
        meta = pd.DataFrame(rows)
        meta["release_date"] = pd.to_datetime(meta["release_date"], errors="coerce", format="mixed")
        meta.to_parquet(cache, index=False)
    return pd.read_parquet(cache)


def kcore(df, min_user=5, min_item=10):
    while True:
        uc = df["user_id"].map(df["user_id"].value_counts())
        ic = df["item_id"].map(df["item_id"].value_counts())
        mask = (uc >= min_user) & (ic >= min_item)
        if mask.all():
            return df.reset_index(drop=True)
        df = df[mask]


def encode(df):
    users = np.sort(df["user_id"].unique())
    items = np.sort(df["item_id"].unique())
    user2idx = pd.Series(np.arange(len(users)), index=users)
    item2idx = pd.Series(np.arange(len(items)), index=items)
    df = df.copy()
    df["uid"] = user2idx.loc[df["user_id"]].values
    df["iid"] = item2idx.loc[df["item_id"]].values
    return df, user2idx, item2idx


def temporal_split(df, val_start, test_start):
    val_start, test_start = pd.Timestamp(val_start), pd.Timestamp(test_start)
    train = df[df["timestamp"] < val_start].reset_index(drop=True)
    val = df[(df["timestamp"] >= val_start) & (df["timestamp"] < test_start)].reset_index(drop=True)
    test = df[df["timestamp"] >= test_start].reset_index(drop=True)
    return train, val, test


def to_csr(df, n_users, n_items, values=None):
    data = np.ones(len(df), dtype=np.float32) if values is None else np.asarray(values, dtype=np.float32)
    return sp.csr_matrix((data, (df["uid"].values, df["iid"].values)), shape=(n_users, n_items))


def make_ground_truth(history, target, threshold=4.0):
    known_users = np.unique(history["uid"].values)
    known_items = np.unique(history["iid"].values)
    t = target[target["rating"] >= threshold]
    t = t[np.isin(t["uid"].values, known_users) & np.isin(t["iid"].values, known_items)]
    seen = set(zip(history["uid"].values, history["iid"].values))
    t = t[[(u, i) not in seen for u, i in zip(t["uid"].values, t["iid"].values)]]
    gt = t.groupby("uid")["iid"].apply(lambda x: np.unique(x.values))
    return gt.index.values, list(gt.values)


def candidate_mask(history, n_items):
    mask = np.zeros(n_items, dtype=bool)
    mask[np.unique(history["iid"].values)] = True
    return mask
