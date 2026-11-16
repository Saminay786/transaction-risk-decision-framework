### Data split (time-ordered)

| Split | Rows | Frauds | Fraud rate | Fraud € |
|---|---:|---:|---:|---:|
| train | 170,884 | 360 | 0.211% | €40,168 |
| val_a | 28,480 | 24 | 0.084% | €5,394 |
| val_b | 28,481 | 33 | 0.116% | €6,837 |
| test | 56,962 | 75 | 0.132% | €7,729 |

### Model (test set)

Weighting chosen on val_b PR-AUC: **unweighted** (PR-AUC 0.841 unweighted vs 0.807 with scale_pos_weight). Calibrator chosen on val_b Brier: **platt**.

| Scores | AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| raw | 0.9865 | 0.7895 | 0.000444 |
| isotonic | 0.9591 | 0.7673 | 0.000467 |
| platt (used) | 0.9865 | 0.7895 | 0.000444 |

### Policies on test

Test set: 56,962 txns, 75 frauds, €7,729 fraud amount. Review capacity K = 569 reviews. P2 thresholds fitted on val_b: t_r = 0.562, t_d = 0.562 (equal, so P2 uses **no review band**).

| Policy | Cost / 1k txns [95% CI] | Fraud € caught | Residual exposure | FPR | Review rate | Decline rate | ROI / review | Savings vs P1 [95% CI] | Savings vs approve-all [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| P0 approve-all | €155.4 [€81.4, €247.6] | 0.0% | €7,729 | 0.000% | 0.000% | 0.000% | — | −€3,737 [−€7,769, −€538] | — |
| P1 decline if p>0.5 | €89.8 [€34.5, €161.4] | 49.0% | €3,941 | 0.011% | 0.000% | 0.102% | — | — | €3,737 [€538, €7,769] |
| P2 global thresholds | €91.1 [€35.1, €161.8] | 48.4% | €3,989 | 0.007% | 0.000% | 0.093% | — | −€72 [−€247, €20] | €3,665 [€491, €7,759] |
| P3 per-txn expected cost | €69.4 [€25.0, €127.0] | 68.2% | €2,455 | 0.076% | 0.058% | 0.118% | €9.42 | €1,162 [−€352, €3,493] | €4,899 [€1,307, €9,362] |

Bootstrap probability that P3 is cheaper than P1: **84.9%**; than approve-all: **99.7%** (1,000 paired resamples).

### Review-cost sensitivity (P2 re-fitted on val_b per c_r; evaluated on test)

- P2 uses a review band only for c_r ≤ €1; above that, its optimal t_r = t_d.
- P3 reviews on test fall from 133 at c_r = €1 to 4 at c_r = €20; P3 cost per 1k stays within €67.3–€69.8.
- Full table: `reports/sensitivity_cr.csv`.
