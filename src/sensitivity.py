"""Review-cost sweep and loss-vs-friction frontier."""
import numpy as np
import pandas as pd

from .config import CostConfig
from .costs import APPROVE, DECLINE, REVIEW, realised_costs
from .policies import capacity, fit_p2, p1_fixed, p2_thresholds, p3_argmin


def sweep_review_cost(val, test, cfg: CostConfig, values=range(1, 21)) -> pd.DataFrame:
    """val/test: (p, y, A). Re-fit P2 on validation for each c_r; evaluate on test.
    Only changes c_r in memory; config/costs.yaml is never edited."""
    pv, yv, Av = val
    pt, yt, At = test
    cap_t = capacity(len(pt), cfg)
    rows = []
    for c_r in values:
        c = cfg.with_(c_r=float(c_r))
        t_r, t_d, _ = fit_p2(pv, yv, Av, c)
        a2 = p2_thresholds(pt, t_r, t_d, cap_t)
        a3 = p3_argmin(pt, At, c, cap=cap_t)
        n = len(pt)
        rows.append({
            "c_r": c_r, "t_r": t_r, "t_d": t_d,
            "cost_per_1k_P1": 1000 * realised_costs(yt, At, p1_fixed(pt), c).sum() / n,
            "cost_per_1k_P2": 1000 * realised_costs(yt, At, a2, c).sum() / n,
            "cost_per_1k_P3": 1000 * realised_costs(yt, At, a3, c).sum() / n,
            "reviews_P2": int((a2 == REVIEW).sum()),
            "reviews_P3": int((a3 == REVIEW).sum()),
        })
    return pd.DataFrame(rows)


def friction_and_loss(y, A, a, rho):
    """x = legit txns reviewed/declined per 1k; y = residual fraud EUR per 1k."""
    n = len(y)
    legit_hit = ((y == 0) & (a != APPROVE)).sum()
    resid = A[(y == 1) & (a == APPROVE)].sum() + (1 - rho) * A[(y == 1) & (a == REVIEW)].sum()
    return 1000 * legit_hit / n, 1000 * resid / n


def frontier_curves(p, y, A, cfg: CostConfig) -> dict:
    """Two traced curves on test: decline-only threshold sweep, and review-only
    threshold sweep (capacity-bound, highest p first)."""
    qs = np.unique(np.concatenate([np.quantile(p, np.linspace(0.5, 1, 400)), [1.0]]))
    cap = capacity(len(p), cfg)
    dec, rev = [], []
    for t in qs[::-1]:
        dec.append(friction_and_loss(y, A, np.where(p > t, DECLINE, APPROVE), cfg.rho))
        rev.append(friction_and_loss(y, A, p2_thresholds(p, t, 1.0, cap), cfg.rho))
    return {"Decline-only threshold sweep": tuple(np.array(dec).T),
            "Review-only sweep (capacity K)": tuple(np.array(rev).T)}
