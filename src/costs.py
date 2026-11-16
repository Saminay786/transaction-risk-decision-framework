"""Expected and realised cost of each action.

Actions: 0 = approve, 1 = review, 2 = decline.

E[C|approve] = p (A + phi)
E[C|review]  = c_r + p (1 - rho)(A + phi) + (1 - p) delta
E[C|decline] = (1 - p)(m A + kappa)

Realised cost uses the true label y in place of p. For reviews we take the
expectation over the review outcome (caught with probability rho), so realised
cost is deterministic given y.
"""
import numpy as np

from .config import CostConfig

APPROVE, REVIEW, DECLINE = 0, 1, 2
ACTION_NAMES = {APPROVE: "approve", REVIEW: "review", DECLINE: "decline"}


def expected_costs(p, A, cfg: CostConfig) -> np.ndarray:
    """Return an (n, 3) array of expected costs [approve, review, decline]."""
    p = np.asarray(p, dtype=float)
    A = np.asarray(A, dtype=float)
    loss = A + cfg.phi
    approve = p * loss
    review = cfg.c_r + p * (1 - cfg.rho) * loss + (1 - p) * cfg.delta
    decline = (1 - p) * (cfg.m * A + cfg.kappa)
    return np.column_stack([approve, review, decline])


def realised_cost_matrix(y, A, cfg: CostConfig) -> np.ndarray:
    """(n, 3) realised cost of each action given the true label."""
    return expected_costs(np.asarray(y, dtype=float), A, cfg)


def realised_costs(y, A, actions, cfg: CostConfig) -> np.ndarray:
    """Per-transaction realised cost of the chosen actions."""
    M = realised_cost_matrix(y, A, cfg)
    actions = np.asarray(actions, dtype=int)
    return M[np.arange(len(actions)), actions]
