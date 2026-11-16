"""Render README results block and the decision memo from files in reports/.
No result number is typed by hand: everything below is read from saved outputs."""
import json
import re

import pandas as pd

from .config import ROOT

REPORTS = ROOT / "reports"
README = ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
NAMES = {"P0": "P0 approve-all", "P1": "P1 decline if p>0.5", "P2": "P2 global thresholds",
         "P3": "P3 per-txn expected cost"}


def _eur(x, d=0):
    s = f"€{abs(x):,.{d}f}"
    return f"−{s}" if round(x, d) < 0 else s


def _pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def _ci(lo, hi, d=0):
    return f"[{_eur(lo, d)}, {_eur(hi, d)}]"


def load():
    k = pd.read_csv(REPORTS / "kpis.csv", index_col="policy")
    bs = pd.read_csv(REPORTS / "kpis_bootstrap.csv", index_col="policy")
    mm = pd.read_csv(REPORTS / "model_metrics.csv")
    mc = pd.read_csv(REPORTS / "model_comparison.csv")
    sp = pd.read_csv(REPORTS / "split_summary.csv", index_col="split")
    sens = pd.read_csv(REPORTS / "sensitivity_cr.csv")
    meta = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    return k, bs, mm, mc, sp, sens, meta


def results_block() -> str:
    k, bs, mm, mc, sp, sens, meta = load()
    chosen = mm[mm["chosen"]].iloc[0]
    raw = mm[mm["scores"] == "raw"].iloc[0]
    L = []
    L.append("### Data split (time-ordered)\n")
    L.append("| Split | Rows | Frauds | Fraud rate | Fraud € |")
    L.append("|---|---:|---:|---:|---:|")
    for s, r in sp.iterrows():
        L.append(f"| {s} | {int(r.rows):,} | {int(r.frauds)} | {r.fraud_rate_pct:.3f}% | {_eur(r.fraud_amount_eur)} |")
    L.append("")
    L.append("### Model (test set)\n")
    L.append(f"Weighting chosen on val_b PR-AUC: **{meta['model_variant']}** "
             f"(PR-AUC {mc.set_index('variant').loc['unweighted','val_b_pr_auc']:.3f} unweighted vs "
             f"{mc.set_index('variant').loc['scale_pos_weight','val_b_pr_auc']:.3f} with scale_pos_weight). "
             f"Calibrator chosen on val_b Brier: **{meta['calibrator']}**.\n")
    L.append("| Scores | AUC | PR-AUC | Brier |")
    L.append("|---|---:|---:|---:|")
    for _, r in mm.iterrows():
        tag = " (used)" if r.chosen else ""
        L.append(f"| {r.scores}{tag} | {r.auc:.4f} | {r.pr_auc:.4f} | {r.brier:.6f} |")
    L.append("")
    t = meta["p2_thresholds"]
    L.append("### Policies on test\n")
    L.append(f"Test set: {int(k.loc['P0','n_txns']):,} txns, {int(k.loc['P0','n_frauds'])} frauds, "
             f"{_eur(k.loc['P0','fraud_eur_total'])} fraud amount. Review capacity K = "
             f"{meta['test_review_capacity']} reviews. P2 thresholds fitted on val_b: "
             f"t_r = {t['t_r']:.3g}, t_d = {t['t_d']:.3g}"
             + (" (equal, so P2 uses **no review band**)." if t['t_r'] == t['t_d'] else ".") + "\n")
    L.append("| Policy | Cost / 1k txns [95% CI] | Fraud € caught | Residual exposure | FPR | Review rate | Decline rate | ROI / review | Savings vs P1 [95% CI] | Savings vs approve-all [95% CI] |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for p in k.index:
        r, b = k.loc[p], bs.loc[p]
        roi = "—" if pd.isna(r.roi_per_review_eur) else _eur(r.roi_per_review_eur, 2)
        s1 = "—" if p == "P1" else f"{_eur(b.savings_vs_P1_eur)} {_ci(b.savings_vs_P1_lo, b.savings_vs_P1_hi)}"
        s0 = "—" if p == "P0" else (f"{_eur(b.savings_vs_approve_all_eur)} "
                                    f"{_ci(b.savings_vs_approve_all_lo, b.savings_vs_approve_all_hi)}")
        L.append(f"| {NAMES[p]} | {_eur(r.cost_per_1k_eur, 1)} {_ci(b.cost_per_1k_lo, b.cost_per_1k_hi, 1)} "
                 f"| {_pct(r.fraud_eur_caught_rate)} | {_eur(r.residual_exposure_eur)} "
                 f"| {_pct(r.false_positive_rate, 3)} | {_pct(r.review_rate, 3)} | {_pct(r.decline_rate, 3)} "
                 f"| {roi} | {s1} | {s0} |")
    L.append("")
    p3 = bs.loc["P3"]
    L.append(f"Bootstrap probability that P3 is cheaper than P1: **{p3.p_savings_vs_P1_gt0:.1%}**; "
             f"than approve-all: **{p3.p_savings_vs_approve_all_gt0:.1%}** (1,000 paired resamples).\n")
    L.append("### Review-cost sensitivity (P2 re-fitted on val_b per c_r; evaluated on test)\n")
    L.append(f"- P2 uses a review band only for c_r ≤ {_eur(sens.loc[sens.reviews_P2 > 0, 'c_r'].max()) if (sens.reviews_P2 > 0).any() else '— (never)'}; "
             f"above that, its optimal t_r = t_d.")
    L.append(f"- P3 reviews on test fall from {int(sens.reviews_P3.iloc[0])} at c_r = {_eur(sens.c_r.iloc[0])} "
             f"to {int(sens.reviews_P3.iloc[-1])} at c_r = {_eur(sens.c_r.iloc[-1])}; P3 cost per 1k stays within "
             f"{_eur(sens.cost_per_1k_P3.min(), 1)}–{_eur(sens.cost_per_1k_P3.max(), 1)}.")
    L.append(f"- Full table: `reports/sensitivity_cr.csv`.")
    return "\n".join(L)


def memo() -> str:
    k, bs, mm, mc, sp, sens, meta = load()
    p1, p3 = k.loc["P1"], k.loc["P3"]
    b3 = bs.loc["P3"]
    cfg = meta["config"]
    n = int(k.loc["P0", "n_txns"])
    hours = sp.loc["test", "time_end_h"] - sp.loc["test", "time_start_h"]
    p2 = k.loc["P2"]
    p2_note = ("the global-threshold policy P2, tuned on "
               f"{int(sp.loc['val_b', 'frauds'])} validation frauds, "
               + ("did not beat P1 on test" if p2.total_cost_eur >= p1.total_cost_eur
                  else "beat P1 on test but by less than P3"
                  if p2.total_cost_eur > p3.total_cost_eur else "did as well as P3 on test"))
    cap_note = (", so no extra analysts are needed at these costs"
                if p3.n_reviews < 0.5 * meta["test_review_capacity"] else
                "; capacity is close to binding, so headcount should be planned")
    return f"""# Decision memo: replace the fixed 0.5 fraud cut-off with expected-cost decisions

**To:** Head of Trust & Risk · **Re:** card-transaction approve / review / decline policy
**Basis:** back-test on the last {n:,} transactions (~{hours:.0f} hours) of the ULB card dataset; all figures from `reports/`.

## Recommendation
Move from "decline if fraud score > 0.5" (P1) to **per-transaction expected-cost decisions (P3)**:
approve, send to review, or decline according to which has the lowest expected € cost, given the
calibrated fraud probability **and the transaction amount**. Pilot it in shadow mode before switching.

## What it changes (test period)
| | Today-like rule (P1) | Proposed (P3) |
|---|---:|---:|
| Total cost per 1,000 txns | {_eur(p1.cost_per_1k_eur, 1)} | {_eur(p3.cost_per_1k_eur, 1)} |
| Share of fraud € stopped | {_pct(p1.fraud_eur_caught_rate)} | {_pct(p3.fraud_eur_caught_rate)} |
| Fraud € still getting through | {_eur(p1.residual_exposure_eur)} | {_eur(p3.residual_exposure_eur)} |
| Legit customers declined | {int(round(p1.false_decline_rate * (n - k.loc['P0','n_frauds'])))} | {int(round(p3.false_decline_rate * (n - k.loc['P0','n_frauds'])))} |
| Manual reviews | {int(p1.n_reviews)} | {int(p3.n_reviews)} (capacity {meta['test_review_capacity']}) |

Savings vs P1: **{_eur(b3.savings_vs_P1_eur)}** over the test period, 95% CI {_ci(b3.savings_vs_P1_lo, b3.savings_vs_P1_hi)}
(P3 cheaper in {b3.p_savings_vs_P1_gt0:.0%} of bootstrap resamples).
Savings vs approving everything: **{_eur(b3.savings_vs_approve_all_eur)}**, 95% CI {_ci(b3.savings_vs_approve_all_lo, b3.savings_vs_approve_all_hi)}.

## Why it works
A flat probability cut-off treats a €2 and a €2,000 transaction the same. P3 declines or reviews
medium-risk transactions only when the amount at stake justifies the review cost or customer
friction. It needs no threshold tuning on fraud labels, only a calibrated score, which matters
when there are few frauds to tune on ({p2_note}).

## Risks and caveats
- **Small sample:** {int(k.loc['P0','n_frauds'])} test frauds. The saving vs P1 is directionally positive but its CI
  {'includes' if b3.savings_vs_P1_lo < 0 else 'excludes'} zero. Treat it as a hypothesis for a shadow pilot, not a booked number.
- **Cost assumptions drive the answer:** review €{cfg['c_r']:g}, catch rate {cfg['rho']:.0%}, chargeback overhead €{cfg['phi']:g},
  false-decline cost {cfg['m']:.0%} of amount + €{cfg['kappa']:g}. Finance/Ops should confirm these first.
- **Review headcount:** P3 used {int(p3.n_reviews)} reviews (~{1000 * p3.review_rate:.2f} per 1,000 txns) against a capacity of
  {meta['test_review_capacity']}{cap_note}.
- **Data:** two days of anonymised European card data; behaviour on our own traffic must be validated.

## Ask
Approve a 4-week shadow run of P3 next to the current rule, with Finance-confirmed cost inputs,
and a go/no-go on whether the measured saving's CI excludes zero.
"""


def render_all():
    block = results_block()
    (REPORTS / "results.md").write_text(block + "\n", encoding="utf-8")
    (REPORTS / "decision_memo.md").write_text(memo(), encoding="utf-8")
    if README.exists():
        txt = README.read_text(encoding="utf-8")
        if START in txt and END in txt:
            txt = re.sub(re.escape(START) + r".*?" + re.escape(END),
                         lambda _: f"{START}\n{block}\n{END}", txt, flags=re.S)
            README.write_text(txt, encoding="utf-8")
