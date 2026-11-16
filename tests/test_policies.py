import numpy as np

from src.config import CostConfig
from src.costs import APPROVE, DECLINE, REVIEW, expected_costs, realised_costs
from src.kpis import bootstrap, kpi_table
from src.policies import capacity, fit_p2, p1_fixed, p2_thresholds, p3_argmin

CFG = CostConfig(c_r=5, rho=0.9, phi=15, m=0.03, kappa=10, delta=1, K=0.01)


def _data(n=5000, seed=1):
    rng = np.random.default_rng(seed)
    p = rng.beta(0.3, 20, n)
    y = (rng.uniform(size=n) < p).astype(int)
    A = rng.lognormal(3.5, 1.2, n)
    return p, y, A


def test_p1():
    assert list(p1_fixed([0.1, 0.5, 0.51])) == [APPROVE, APPROVE, DECLINE]


def test_p3_is_argmin_without_capacity_pressure():
    p, _, A = _data()
    a = p3_argmin(p, A, CFG, cap=len(p))
    assert np.array_equal(a, np.argmin(expected_costs(p, A, CFG), axis=1))


def test_p3_capacity_never_exceeded():
    p, _, A = _data()
    for cap in (0, 5, 20, 50):
        a = p3_argmin(p, A, CFG, cap=cap)
        assert (a == REVIEW).sum() <= cap


def test_p3_capacity_keeps_highest_saving():
    p = np.array([0.3, 0.3, 0.3])
    A = np.array([50.0, 500.0, 200.0])
    a = p3_argmin(p, A, CFG, cap=1)
    E = expected_costs(p, A, CFG)
    saving = np.minimum(E[:, 0], E[:, 2]) - E[:, 1]
    assert a[np.argmax(saving)] == REVIEW
    assert (a == REVIEW).sum() == 1


def test_p2_capacity_and_bands():
    p = np.array([0.001, 0.02, 0.05, 0.1, 0.9])
    a = p2_thresholds(p, t_r=0.01, t_d=0.5, cap=2)
    assert a[-1] == DECLINE and a[0] == APPROVE
    assert (a == REVIEW).sum() == 2
    assert a[3] == REVIEW and a[2] == REVIEW and a[1] == APPROVE  # highest-p reviewed


def test_fit_p2_returns_ordered_thresholds_and_beats_p1():
    p, y, A = _data()
    t_r, t_d, cost = fit_p2(p, y, A, CFG)
    assert t_r <= t_d
    assert cost <= realised_costs(y, A, p1_fixed(p), CFG).sum() + 1e-9
    assert capacity(len(p), CFG) >= (p2_thresholds(p, t_r, t_d, capacity(len(p), CFG)) == REVIEW).sum()


def test_kpis_and_bootstrap_shapes():
    p, y, A = _data()
    acts = {"P0": np.zeros(len(p), int), "P1": p1_fixed(p), "P3": p3_argmin(p, A, CFG)}
    k = kpi_table(y, A, acts, CFG)
    assert k.loc["P0", "savings_vs_approve_all_eur"] == 0
    assert k.loc["P0", "fraud_eur_caught_rate"] == 0
    bs = bootstrap(y, A, acts, CFG, B=50)
    assert (bs["cost_per_1k_lo"] <= bs["cost_per_1k_hi"]).all()
