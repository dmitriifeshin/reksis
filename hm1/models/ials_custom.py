import numpy as np
import pandas as pd

from models.ials import IALS


class TimeRatingIALS(IALS):
    def __init__(self, factors=64, reg=1.0, alpha=10.0, iterations=15, cg_steps=3, seed=42,
                 gamma=1.0, tau=2.0, eta=0.3):
        super().__init__(factors, reg, alpha, iterations, cg_steps, seed)
        self.gamma = gamma
        self.tau = tau
        self.eta = eta

    def target(self, rating, timestamp, t_ref):
        rating = np.asarray(rating, dtype=np.float64)
        s = ((rating - 1.0) / 4.0) ** self.gamma
        age = (pd.Timestamp(t_ref) - pd.to_datetime(timestamp)) / pd.Timedelta(days=365.25)
        age = np.clip(np.asarray(age, dtype=np.float64), 0.0, None)
        if self.tau is None or np.isinf(self.tau):
            d = np.ones_like(age)
        else:
            d = self.eta + (1.0 - self.eta) * np.exp(-age / self.tau)
        return (s * d).astype(np.float32)

    def fit_df(self, df, n_users, n_items, t_ref=None):
        t_ref = df["timestamp"].max() if t_ref is None else t_ref
        self.pref = self.target(df["rating"].values, df["timestamp"].values, t_ref)
        return self.fit(df["uid"].values, df["iid"].values, n_users, n_items, pref=self.pref)
