# Decision memo: replace the fixed 0.5 fraud cut-off with expected-cost decisions

**To:** Head of Trust & Risk · **Re:** card-transaction approve / review / decline policy
**Basis:** back-test on the last 56,962 transactions (~8 hours) of the ULB card dataset; all figures from `reports/`.

## Recommendation
Move from "decline if fraud score > 0.5" (P1) to **per-transaction expected-cost decisions (P3)**:
approve, send to review, or decline according to which has the lowest expected € cost, given the
calibrated fraud probability **and the transaction amount**. Pilot it in shadow mode before switching.

## What it changes (test period)
| | Today-like rule (P1) | Proposed (P3) |
|---|---:|---:|
| Total cost per 1,000 txns | €89.8 | €69.4 |
| Share of fraud € stopped | 49.0% | 68.2% |
| Fraud € still getting through | €3,941 | €2,455 |
| Legit customers declined | 6 | 12 |
| Manual reviews | 0 | 33 (capacity 569) |

Savings vs P1: **€1,162** over the test period, 95% CI [−€352, €3,493]
(P3 cheaper in 85% of bootstrap resamples).
Savings vs approving everything: **€4,899**, 95% CI [€1,307, €9,362].

## Why it works
A flat probability cut-off treats a €2 and a €2,000 transaction the same. P3 declines or reviews
medium-risk transactions only when the amount at stake justifies the review cost or customer
friction. It needs no threshold tuning on fraud labels, only a calibrated score, which matters
when there are few frauds to tune on (the global-threshold policy P2, tuned on 33 validation frauds, did not beat P1 on test).

## Risks and caveats
- **Small sample:** 75 test frauds. The saving vs P1 is directionally positive but its CI
  includes zero. Treat it as a hypothesis for a shadow pilot, not a booked number.
- **Cost assumptions drive the answer:** review €5, catch rate 90%, chargeback overhead €15,
  false-decline cost 3% of amount + €10. Finance/Ops should confirm these first.
- **Review headcount:** P3 used 33 reviews (~0.58 per 1,000 txns) against a capacity of
  569, so no extra analysts are needed at these costs.
- **Data:** two days of anonymised European card data; behaviour on our own traffic must be validated.

## Ask
Approve a 4-week shadow run of P3 next to the current rule, with Finance-confirmed cost inputs,
and a go/no-go on whether the measured saving's CI excludes zero.
