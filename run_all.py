"""End-to-end pipeline: data -> model -> calibration -> policies -> KPIs -> sensitivity
-> figures -> README results + decision memo. Every reported number is written to reports/."""
import json
import time

import numpy as np
import pandas as pd

from src import plots
from src.config import ROOT, load_config
from src.costs import REVIEW
from src.data import TARGET, load_raw, split_summary, time_split, xy
from src.kpis import bootstrap, kpi_table
from src.model import Calibrator, metrics, score, train_lgbm
from src.policies import capacity, fit_p2, p0_approve_all, p1_fixed, p2_thresholds, p3_argmin
from src.report import render_all
from src.sensitivity import friction_and_loss, frontier_curves, sweep_review_cost

REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"


def reliability_bins(y, p, edges=(0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0001)):
    """Log-spaced bins: almost all scores are tiny, so quantile bins collapse."""
    b = np.digitize(p, edges[1:-1])
    mp, fp, cnt = [], [], []
    for i in np.unique(b):
        s = b == i
        mp.append(p[s].mean())
        fp.append(y[s].mean())
        cnt.append(int(s.sum()))
    return np.array(mp), np.array(fp), np.array(cnt)


def main():
    t0 = time.time()
    FIGS.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    out = {"config": cfg.__dict__}

    # 1. Data + EDA ---------------------------------------------------------
    df = load_raw()
    splits = time_split(df)
    split_summary(splits).to_csv(REPORTS / "split_summary.csv", index=False)
    q = [0.25, 0.5, 0.75, 0.9, 0.99]
    eda = (df.groupby(TARGET)["Amount"]
             .describe(percentiles=q).rename(index={0: "legit", 1: "fraud"}))
    eda.to_csv(REPORTS / "eda_amount.csv")
    out["eda"] = {"rows": len(df), "frauds": int(df[TARGET].sum()),
                  "fraud_rate_pct": 100 * df[TARGET].mean(),
                  "hours_covered": df["Time"].max() / 3600}
    plots.amount_hist(df["Amount"].to_numpy(), df[TARGET].to_numpy(),
                      FIGS / "amount_fraud_vs_legit.png")

    Xtr, ytr, _ = xy(splits["train"])
    Xa, ya, _ = xy(splits["val_a"])
    Xb, yb, Ab = xy(splits["val_b"])
    Xt, yt, At = xy(splits["test"])

    # 2. Model: weighting comparison ---------------------------------------
    spw_bal = (ytr == 0).sum() / (ytr == 1).sum()
    cands, rows = {}, []
    for name, spw in [("unweighted", 1.0), ("scale_pos_weight", spw_bal)]:
        bst = train_lgbm(Xtr, ytr, Xa, ya, spw, cfg.seed)
        m = metrics(yb, score(bst, Xb))
        rows.append({"variant": name, "scale_pos_weight": spw, "best_iteration": bst.best_iteration,
                     **{f"val_b_{k}": v for k, v in m.items()}})
        cands[name] = bst
    comp = pd.DataFrame(rows)
    comp.to_csv(REPORTS / "model_comparison.csv", index=False)
    chosen = comp.sort_values("val_b_pr_auc", ascending=False).iloc[0]["variant"]
    bst = cands[chosen]
    out["model_variant"] = chosen

    # Calibration: fit on val_a, choose on val_b Brier ----------------------
    raw_a, raw_b, raw_t = score(bst, Xa), score(bst, Xb), score(bst, Xt)
    cals = {m: Calibrator(m).fit(raw_a, ya) for m in ("isotonic", "platt")}
    briers = {m: metrics(yb, c.predict(raw_b))["brier"] for m, c in cals.items()}
    cal_name = min(briers, key=briers.get)
    cal = cals[cal_name]
    out["calibrator"] = cal_name
    out["val_b_brier"] = briers

    mrows, curves = [], {}
    for name, p in [("raw", raw_t)] + [(m, c.predict(raw_t)) for m, c in cals.items()]:
        mrows.append({"scores": name, "chosen": name == cal_name, **metrics(yt, p)})
        curves[f"{'Raw LightGBM' if name == 'raw' else name.title()}"] = reliability_bins(yt, p)
    pd.DataFrame(mrows).to_csv(REPORTS / "model_metrics.csv", index=False)
    plots.reliability(curves, FIGS / "reliability.png")

    pb = cal.predict(raw_b)
    pt = cal.predict(raw_t)

    # 3. Policies (fit on val_b only) ---------------------------------------
    t_r, t_d, val_cost = fit_p2(pb, yb, Ab, cfg)
    out["p2_thresholds"] = {"t_r": t_r, "t_d": t_d, "val_b_cost": val_cost}
    cap = capacity(len(pt), cfg)
    actions = {
        "P0": p0_approve_all(pt),
        "P1": p1_fixed(pt),
        "P2": p2_thresholds(pt, t_r, t_d, cap),
        "P3": p3_argmin(pt, At, cfg, cap=cap),
    }
    out["test_review_capacity"] = cap

    # 4. KPIs + bootstrap ---------------------------------------------------
    kpis = kpi_table(yt, At, actions, cfg)
    kpis.to_csv(REPORTS / "kpis.csv")
    bs = bootstrap(yt, At, actions, cfg, B=1000, seed=cfg.seed)
    bs.to_csv(REPORTS / "kpis_bootstrap.csv")
    plots.cost_bars(bs, FIGS / "cost_by_policy.png")

    # 5. Sensitivity + frontier ---------------------------------------------
    sens = sweep_review_cost((pb, yb, Ab), (pt, yt, At), cfg)
    sens.to_csv(REPORTS / "sensitivity_cr.csv", index=False)
    plots.sensitivity(sens, FIGS / "sensitivity_cr.png")
    fc = frontier_curves(pt, yt, At, cfg)
    pts = {k: friction_and_loss(yt, At, actions[k], cfg.rho) for k in ("P1", "P2", "P3")}
    pd.DataFrame([{"policy": k, "legit_hit_per_1k": x, "residual_eur_per_1k": y}
                  for k, (x, y) in pts.items()]).to_csv(REPORTS / "frontier_points.csv", index=False)
    plots.frontier(fc, pts, FIGS / "frontier.png")

    out["runtime_s"] = round(time.time() - t0, 1)
    with open(REPORTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)

    # 6. Render README results + memo from saved outputs --------------------
    render_all()
    print(f"Done in {time.time() - t0:.1f}s. Outputs in {REPORTS}")


if __name__ == "__main__":
    main()
