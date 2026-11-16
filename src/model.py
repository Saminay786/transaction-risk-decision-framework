"""LightGBM fraud model + probability calibration."""
import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

BASE_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "metric": "binary_logloss",
    "verbose": -1,
}


def train_lgbm(X_tr, y_tr, X_es, y_es, scale_pos_weight: float, seed: int):
    params = dict(BASE_PARAMS, seed=seed, scale_pos_weight=scale_pos_weight,
                  deterministic=True, force_row_wise=True, num_threads=4)
    dtr = lgb.Dataset(X_tr, y_tr)
    des = lgb.Dataset(X_es, y_es, reference=dtr)
    return lgb.train(params, dtr, num_boost_round=3000, valid_sets=[des],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def score(booster, X) -> np.ndarray:
    return booster.predict(X, num_iteration=booster.best_iteration)


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Calibrator:
    """Isotonic or Platt (logistic on the raw-score logit)."""

    def __init__(self, method: str):
        assert method in ("isotonic", "platt")
        self.method = method

    def fit(self, raw, y):
        if self.method == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, y)
        else:
            self.m = LogisticRegression(C=1e6).fit(_logit(raw), y)
        return self

    def predict(self, raw):
        if self.method == "isotonic":
            return self.m.predict(raw)
        return self.m.predict_proba(_logit(raw))[:, 1]


def metrics(y, p) -> dict:
    return {
        "auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
    }
