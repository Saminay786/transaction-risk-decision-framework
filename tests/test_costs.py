import numpy as np
import pytest

from src.config import CostConfig
from src.costs import APPROVE, DECLINE, REVIEW, expected_costs, realised_costs

CFG = CostConfig(c_r=5, rho=0.9, phi=15, m=0.03, kappa=10, delta=1, K=0.01)


def test_expected_costs_hand_computed():
    # p=0.2, A=100: approve = 0.2*115 = 23
    # review = 5 + 0.2*0.1*115 + 0.8*1 = 5 + 2.3 + 0.8 = 8.1
    # decline = 0.8*(3 + 10) = 10.4
    E = expected_costs([0.2], [100.0], CFG)
    assert E[0] == pytest.approx([23.0, 8.1, 10.4])


def test_p_zero_and_one():
    E = expected_costs([0.0, 1.0], [50.0, 50.0], CFG)
    # p=0: approve free, review = c_r + delta, decline = m*A + kappa
    assert E[0] == pytest.approx([0.0, 6.0, 11.5])
    # p=1: approve = A+phi, review = c_r + (1-rho)(A+phi), decline free
    assert E[1] == pytest.approx([65.0, 5 + 6.5, 0.0])


def test_zero_amount():
    E = expected_costs([0.5], [0.0], CFG)
    assert E[0] == pytest.approx([7.5, 5 + 0.75 + 0.5, 5.0])


def test_realised_costs_use_labels():
    y = np.array([1, 0, 1, 0])
    A = np.array([100.0, 100.0, 100.0, 100.0])
    a = np.array([APPROVE, APPROVE, REVIEW, DECLINE])
    c = realised_costs(y, A, a, CFG)
    assert c == pytest.approx([115.0, 0.0, 5 + 0.1 * 115, 13.0])


def test_realised_equals_expected_at_labels():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 50)
    A = rng.uniform(0, 500, 50)
    for act in (APPROVE, REVIEW, DECLINE):
        a = np.full(50, act)
        assert realised_costs(y, A, a, CFG) == pytest.approx(expected_costs(y, A, CFG)[:, act])
