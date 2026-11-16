"""Decision policies: P0 approve-all, P1 fixed 0.5 cut-off, P2 global thresholds,
P3 per-transaction argmin of expected cost. P2 and P3 respect review capacity K."""
import numpy as np

from .config import CostConfig
from .costs import APPROVE, DECLINE, REVIEW, expected_costs, realised_cost_matrix


def capacity(n: int, cfg: CostConfig) -> int:
    return int(np.floor(cfg.K * n))


def p0_approve_all(p) -> np.ndarray:
    return np.zeros(len(p), dtype=int)


def p1_fixed(p, threshold: float = 0.5) -> np.ndarray:
    return np.where(np.asarray(p) > threshold, DECLINE, APPROVE).astype(int)


def p2_thresholds(p, t_r: float, t_d: float, cap: int) -> np.ndarray:
    """Review if t_r < p <= t_d, decline if p > t_d. If the review band holds more
    than `cap` transactions, only the `cap` highest-p are reviewed; the rest approved."""
    p = np.asarray(p, dtype=float)
    a = np.full(len(p), APPROVE, dtype=int)
    a[p > t_d] = DECLINE
    band = np.flatnonzero((p > t_r) & (p <= t_d))
    if len(band) > cap:
        order = band[np.argsort(-p[band], kind="stable")]
        band = order[:cap]
    a[band] = REVIEW
    return a


def default_grid() -> np.ndarray:
    return np.unique(np.concatenate([[0.0], np.logspace(-4, 0, 49)]))


def fit_p2(p, y, A, cfg: CostConfig, grid=None):
    """Grid-search (t_r, t_d), t_r <= t_d, minimising realised cost on the given data.
    t_r == t_d means no review band. Returns (t_r, t_d, cost)."""
    grid = default_grid() if grid is None else np.asarray(grid)
    M = realised_cost_matrix(y, A, cfg)
    cap = capacity(len(p), cfg)
    idx = np.arange(len(p))
    best = (None, None, np.inf)
    for j, t_d in enumerate(grid):
        for t_r in grid[: j + 1]:
            a = p2_thresholds(p, t_r, t_d, cap)
            c = M[idx, a].sum()
            if c < best[2] - 1e-9:
                best = (float(t_r), float(t_d), float(c))
    return best


def p3_argmin(p, A, cfg: CostConfig, cap: int | None = None) -> np.ndarray:
    """Per-transaction argmin of expected cost. If more than `cap` transactions want
    review, keep those with the largest expected saving of review over the best
    non-review action; the rest take their best of approve/decline."""
    E = expected_costs(p, A, cfg)
    a = np.argmin(E, axis=1).astype(int)
    if cap is None:
        cap = capacity(len(a), cfg)
    rev = np.flatnonzero(a == REVIEW)
    if len(rev) > cap:
        fallback = np.where(E[:, APPROVE] <= E[:, DECLINE], APPROVE, DECLINE)
        saving = np.minimum(E[rev, APPROVE], E[rev, DECLINE]) - E[rev, REVIEW]
        keep = rev[np.argsort(-saving, kind="stable")[:cap]]
        drop = np.setdiff1d(rev, keep)
        a[drop] = fallback[drop]
    return a
