# PLAN — Transaction Risk Decision Framework

Status: **DRAFT, waiting for approval.** No code will be written until this is approved.

Goal: turn a calibrated fraud score into **approve / review / decline** decisions that minimise
expected € cost, and report business KPIs (with bootstrap CIs). Model AUC is secondary.

---

## 0. Inputs & setup

| Item | Decision |
|---|---|
| Data | `creditcard.csv` already on disk at `Downloads\archive (1)\` (284,807 rows, 31 cols). I'll **copy** it to `data/raw/creditcard.csv` (no Kaggle download needed). `data/` is gitignored. |
| Python | 3.12.6. Already installed: lightgbm 4.7.0, scikit-learn 1.9.1, pandas 3.0.6, pyyaml, matplotlib. `requirements.txt` pins these exact versions (+ numpy, scipy, pytest). |
| Repo | Not a git repo yet. I'll add `.gitignore` but **won't** `git init` / push unless you ask. |
| Determinism | Single `seed: 42` in config, used for LightGBM, bootstrap, and any sampling. |

### ⚠️ `config/costs.yaml` — needs your input
"The same `config/costs.yaml` from before" **does not exist anywhere under Downloads** (I searched).
Below are proposed placeholder values. Tell me to use these, or paste your previous file.

```yaml
# config/costs.yaml  (all money in EUR)
c_r: 5.0      # cost of one manual review (analyst time)
rho: 0.90     # probability a review catches a fraud that is really fraud
phi: 15.0     # extra cost per approved fraud on top of the amount (chargeback fee + ops)
m: 0.03       # margin lost on a falsely declined legit txn, as a fraction of Amount
kappa: 10.0   # fixed cost of a false decline (customer friction / churn risk)
delta: 1.0    # friction cost of reviewing a legit txn (delay, customer contact)
K: 0.01       # review capacity as a FRACTION of transactions (1% → ~570 reviews on test)
seed: 42
```
(K as a fraction rather than an absolute count, so it scales between validation and test — say if you'd prefer an absolute number.)

---

## 1. Data & EDA — `src/data.py`, `src/eda.py`
- Load CSV, sort by `Time`, split by row order: **60% train / 20% validation / 20% test** (no shuffling).
- Report rows / frauds / fraud rate per split → `reports/split_summary.csv`.
  Expected ≈ 80–100 frauds in test; the data covers only ~2 days, so the fraud rate can drift between splits — I'll state it explicitly.
- EDA: overall fraud rate; Amount distribution fraud vs legit (log-scale histogram + quantile table) → `reports/eda_amount.csv`, `reports/figures/amount_fraud_vs_legit.png`.
- Features: `V1–V28`, `Amount`, plus `hour = (Time // 3600) % 24`. Raw `Time` is **dropped** (it's monotonic and would just encode the split).

## 2. Model — `src/model.py`
- LightGBM binary, early stopping on validation log-loss (fixed max rounds, seed from config).
- Compare **(a) no weighting vs (b) `scale_pos_weight = neg/pos`** on validation PR-AUC; pick the better one, record both → `reports/model_comparison.csv`.
- Calibration on validation: **isotonic vs Platt (sigmoid)**; pick by validation Brier.
- Report on test for raw + calibrated: AUC, PR-AUC, Brier → `reports/model_metrics.csv`; reliability plot → `reports/figures/reliability.png`.

**Known caveat (validation reuse):** validation is used for early stopping, calibration *and* policy tuning. To limit optimism I'll split validation by time into **val-A (first half: early stopping + calibration)** and **val-B (second half: P2 threshold grid search + choosing calibrator)**. Test stays untouched until final evaluation. Say if you'd rather keep a single validation block as written in the original spec.

## 3. Decision layer — `src/costs.py`, `src/policies.py`
Symbols: p_i = calibrated P(fraud), A_i = Amount, others from config.
```
E[C|approve] = p_i (A_i + φ)
E[C|review]  = c_r + p_i (1−ρ)(A_i + φ) + (1−p_i) δ
E[C|decline] = (1−p_i)(m·A_i + κ)
```
Realised cost (true label y_i ∈ {0,1}) — expected cost with p_i replaced by y_i:
- approve: y(A+φ); decline: (1−y)(mA+κ)
- review: c_r + (1−y)δ + y·(1−ρ)(A+φ) — **expected over the review outcome** (deterministic, no simulation noise). Alternative: simulate Bernoulli(ρ) catches with the seed. I'll use the expectation unless you say otherwise.

Policies (all fitted on validation only, applied to test):
- **P0 approve-all** — baseline for "savings vs approve-all".
- **P1** decline if p > 0.5, else approve.
- **P2** global thresholds (t_r, t_d), t_r ≤ t_d: review if t_r < p ≤ t_d, decline if p > t_d. Grid on a log-spaced p grid, minimise realised validation cost. **Capacity K enforced**: if more than K·n fall in the review band, only the K·n highest-p ones are reviewed and the rest are approved, so P2 and P3 are compared at equal review budget.
- **P3** per-transaction argmin of expected cost, then if reviews exceed K·n, keep the K·n reviews with the largest **expected saving of review vs the best non-review action**; the rest fall back to their best of approve/decline.

Unit tests (`tests/test_costs.py`, `tests/test_policies.py`): hand-computed cost values, edge cases p=0 / p=1, A=0, argmin correctness, capacity never exceeded, t_r ≤ t_d.

## 4. KPIs — `src/kpis.py` → `reports/kpis.csv`
Per policy (P0–P3), on test:
- fraud € caught rate = € of fraud declined or caught by review / total fraud €
- residual exposure = € fraud approved (+ missed in review)
- false-positive rate = legit declined or reviewed / legit
- review rate, decline rate
- total cost per 1,000 txns
- ROI per review = (fraud € prevented by reviews − review costs) / #reviews
- savings vs P1 and vs approve-all (€ and %)

**Bootstrap 95% CIs**: 1,000 resamples of the test set (rows with replacement, policy decisions fixed as fitted on validation) for total cost per 1k and both savings figures → `reports/kpis_bootstrap.csv`. Savings CIs use *paired* resamples (same rows for both policies).

## 5. Sensitivity — `src/sensitivity.py`
- Sweep **c_r = 1..20** (step 1): re-fit P2 thresholds on validation per c_r, evaluate on test → plot optimal t_r, t_d and test cost vs c_r → `reports/sensitivity_cr.csv`, `reports/figures/sensitivity_cr.png`. (Sweep only changes c_r in memory; `costs.yaml` is never edited.)
- **Loss-vs-friction frontier**: x = customer friction (# legit txns reviewed or declined per 1k, or € friction cost), y = residual fraud € per 1k; trace by sweeping thresholds; mark P1, P2, P3 → `reports/figures/frontier.png`.

## 6. Deliverables
```
transaction-risk-decision-framework/
  PLAN.md  README.md  run_all.py  requirements.txt  .gitignore
  config/costs.yaml
  data/raw/creditcard.csv            (gitignored)
  src/  data.py eda.py model.py costs.py policies.py kpis.py sensitivity.py plots.py
  tests/ test_costs.py test_policies.py
  reports/ *.csv  metrics.json  decision_memo.md  figures/*.png
```
- `run_all.py`: one command runs everything end-to-end and writes all of `reports/` (target: < 2 min on the laptop).
- **README**: problem, formulation with every symbol defined, assumptions table, results table (auto-filled from `reports/kpis.csv`), 3 charts (reliability, sensitivity, frontier), limitations (anonymised features, ~2 days of data, few frauds → wide CIs, costs are assumptions, validation reuse, next step: IEEE-CIS with user-level leakage controls).
- **`reports/decision_memo.md`**: one page for a Trust & Risk leader — recommendation, € impact with CIs, review headcount implication, risks.
- **Rule:** every number in README/memo is generated from files in `reports/` (I'll render them from the CSVs in `run_all.py` rather than typing numbers by hand).

---

## Decisions needed from you
1. **costs.yaml** — use the proposed values above, or paste your previous file?
2. **Validation split** into val-A / val-B (my recommendation), or a single validation block?
3. **Review outcome** — expected value (recommended) or simulated Bernoulli(ρ)?
4. **Location** — this folder (`Downloads\transaction-risk-decision-framework`) OK?
