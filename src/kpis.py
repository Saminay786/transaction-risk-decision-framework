"""Business KPIs per policy and paired bootstrap CIs."""
import numpy as np
import pandas as pd

from .config import CostConfig
from .costs import APPROVE, DECLINE, REVIEW, realised_costs


def policy_kpis(y, A, a, cfg: CostConfig) -> dict:
    y = np.asarray(y)
    A = np.asarray(A, dtype=float)
    a = np.asarray(a)
    n = len(y)
    fraud, legit = y == 1, y == 0
    rev, dec, app = a == REVIEW, a == DECLINE, a == APPROVE
    fraud_eur = A[fraud].sum()
    caught_eur = A[fraud & dec].sum() + cfg.rho * A[fraud & rev].sum()
    residual_eur = A[fraud & app].sum() + (1 - cfg.rho) * A[fraud & rev].sum()
    cost = realised_costs(y, A, a, cfg).sum()
    n_rev = int(rev.sum())
    # Net value of reviews vs approving those same transactions:
    # loss prevented on reviewed fraud minus review cost and legit-customer friction.
    prevented = cfg.rho * (A[fraud & rev] + cfg.phi).sum()
    review_cost = cfg.c_r * n_rev + cfg.delta * (legit & rev).sum()
    return {
        "n_txns": n,
        "n_frauds": int(fraud.sum()),
        "fraud_eur_total": fraud_eur,
        "fraud_eur_caught_rate": caught_eur / fraud_eur if fraud_eur > 0 else np.nan,
        "residual_exposure_eur": residual_eur,
        "false_positive_rate": (legit & (rev | dec)).sum() / legit.sum(),
        "false_decline_rate": (legit & dec).sum() / legit.sum(),
        "review_rate": n_rev / n,
        "decline_rate": dec.sum() / n,
        "n_reviews": n_rev,
        "n_declines": int(dec.sum()),
        "total_cost_eur": cost,
        "cost_per_1k_eur": 1000 * cost / n,
        "roi_per_review_eur": (prevented - review_cost) / n_rev if n_rev else np.nan,
    }


def kpi_table(y, A, actions: dict, cfg: CostConfig, ref_p1="P1", ref_p0="P0") -> pd.DataFrame:
    rows = {name: policy_kpis(y, A, a, cfg) for name, a in actions.items()}
    df = pd.DataFrame(rows).T
    for ref, col in [(ref_p1, "vs_P1"), (ref_p0, "vs_approve_all")]:
        base = df.loc[ref, "total_cost_eur"]
        df[f"savings_{col}_eur"] = base - df["total_cost_eur"]
        df[f"savings_{col}_pct"] = 100 * (base - df["total_cost_eur"]) / base
    df.index.name = "policy"
    return df


def bootstrap(y, A, actions: dict, cfg: CostConfig, B: int = 1000, seed: int = 42,
              refs=("P1", "P0")) -> pd.DataFrame:
    """Paired bootstrap over test rows (decisions held fixed). Returns 95% CIs for
    cost per 1k txns and savings vs each reference, per policy."""
    rng = np.random.default_rng(seed)
    n = len(y)
    costs = {k: realised_costs(y, A, a, cfg) for k, a in actions.items()}
    names = list(actions)
    C = np.column_stack([costs[k] for k in names])  # (n, P)
    sums = np.empty((B, len(names)))
    for b in range(B):
        idx = rng.integers(0, n, n)
        sums[b] = C[idx].sum(axis=0)
    point = C.sum(axis=0)
    rows = []
    for j, k in enumerate(names):
        r = {"policy": k,
             "cost_per_1k_eur": 1000 * point[j] / n,
             "cost_per_1k_lo": np.percentile(1000 * sums[:, j] / n, 2.5),
             "cost_per_1k_hi": np.percentile(1000 * sums[:, j] / n, 97.5)}
        for ref in refs:
            i = names.index(ref)
            s = sums[:, i] - sums[:, j]
            tag = "vs_P1" if ref == "P1" else "vs_approve_all"
            r[f"savings_{tag}_eur"] = point[i] - point[j]
            r[f"savings_{tag}_lo"] = np.percentile(s, 2.5)
            r[f"savings_{tag}_hi"] = np.percentile(s, 97.5)
            r[f"p_savings_{tag}_gt0"] = float((s > 0).mean())
        rows.append(r)
    return pd.DataFrame(rows).set_index("policy")
