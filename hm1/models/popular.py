import numpy as np
import pandas as pd


class TopPopular:
    def __init__(self, window_days=None):
        self.window_days = window_days

    def fit(self, df, n_items):
        if self.window_days is not None:
            df = df[df["timestamp"] >= df["timestamp"].max() - pd.Timedelta(days=self.window_days)]
        self.pop = np.bincount(df["iid"].values, minlength=n_items).astype(np.float32)
        return self

    def score(self, users):
        return np.broadcast_to(self.pop, (len(users), len(self.pop)))
