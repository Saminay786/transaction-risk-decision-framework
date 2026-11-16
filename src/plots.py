"""Static matplotlib figures (light theme, fixed categorical order per policy)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# Fixed slot order: entity -> colour never changes between charts.
POLICY_COLORS = {"P0": "#52514e", "P1": "#eb6834", "P2": "#1baf7a", "P3": "#2a78d6"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def _style(ax, title, xlabel, ylabel):
    fig = ax.figure
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def amount_hist(amount, y, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(-1, np.log10(max(amount.max(), 10)), 50)
    a = np.clip(amount, 0.1, None)
    ax.hist(a[y == 0], bins=bins, density=True, histtype="step", lw=2, color=SERIES[0], label="Legit")
    ax.hist(a[y == 1], bins=bins, density=True, histtype="step", lw=2, color=SERIES[1], label="Fraud")
    ax.set_xscale("log")
    _style(ax, "Amount distribution: fraud vs legit (density)", "Amount (EUR, log scale; 0 shown at 0.1)", "Density")
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def reliability(curves: dict, path):
    """curves: name -> (mean_pred, frac_pos, counts)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([1e-6, 1], [1e-6, 1], ls="--", color=INK_2, lw=1, label="Perfect calibration")
    for (name, (mp, fp, cnt)), c in zip(curves.items(), SERIES):
        ax.plot(mp, fp, marker="o", ms=6, lw=2, color=c, label=name,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlim(1e-6, 1)
    ax.set_ylim(0, 1.05)
    _style(ax, "Reliability on test (log-spaced bins)", "Mean predicted P(fraud)", "Observed fraud rate")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sensitivity(df, path):
    """Two panels sharing x (c_r): thresholds and test cost — never a dual axis."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4))
    a3.plot(df["c_r"], df["reviews_P2"], marker="o", ms=5, lw=2, color=POLICY_COLORS["P2"], label="P2")
    a3.plot(df["c_r"], df["reviews_P3"], marker="o", ms=5, lw=2, color=POLICY_COLORS["P3"], label="P3")
    _style(a3, "Reviews used on test vs review cost", "Review cost c_r (EUR)", "Number of reviews")
    a3.legend(frameon=False, labelcolor=INK, fontsize=9)
    a1.plot(df["c_r"], df["t_r"], marker="o", ms=5, lw=2, color=SERIES[0], label="Review threshold t_r")
    a1.plot(df["c_r"], df["t_d"], marker="s", ms=5, lw=2, color=SERIES[1], label="Decline threshold t_d")
    a1.set_yscale("log")
    _style(a1, "P2 optimal thresholds vs review cost", "Review cost c_r (EUR)", "Threshold on calibrated p")
    a1.legend(frameon=False, labelcolor=INK, fontsize=9)
    for col, c, lab in [("cost_per_1k_P1", POLICY_COLORS["P1"], "P1"),
                        ("cost_per_1k_P2", POLICY_COLORS["P2"], "P2"),
                        ("cost_per_1k_P3", POLICY_COLORS["P3"], "P3")]:
        a2.plot(df["c_r"], df[col], marker="o", ms=5, lw=2, color=c, label=lab)
    _style(a2, "Test cost per 1,000 txns vs review cost", "Review cost c_r (EUR)", "EUR per 1,000 txns")
    a2.legend(frameon=False, labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def frontier(curves: dict, points: dict, path):
    """curves: name -> (x, y); points: policy -> (x, y)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for (name, (x, y)), c in zip(curves.items(), [SERIES[3], "#6250d6"]):
        ax.plot(x, y, lw=2, color=c, label=name)
    offsets = {"P1": (8, -14), "P2": (-24, 8), "P3": (8, 6)}
    for k, (x, y) in points.items():
        ax.scatter([x], [y], s=90, color=POLICY_COLORS[k], edgecolor=SURFACE, linewidth=2, zorder=5)
        ax.annotate(k, (x, y), xytext=offsets.get(k, (7, 5)), textcoords="offset points", color=INK,
                    fontsize=10, fontweight="bold")
    ax.set_xscale("symlog", linthresh=1)
    _style(ax, "Loss vs friction frontier (test)", "Legit customers reviewed or declined per 1,000 txns",
           "Residual fraud exposure (EUR per 1,000 txns)")
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cost_bars(bs, path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = list(bs.index)
    y = np.arange(len(names))[::-1]
    for yi, k in zip(y, names):
        r = bs.loc[k]
        ax.barh(yi, r["cost_per_1k_eur"], height=0.55, color=POLICY_COLORS[k])
        ax.errorbar(r["cost_per_1k_eur"], yi, xerr=[[r["cost_per_1k_eur"] - r["cost_per_1k_lo"]],
                    [r["cost_per_1k_hi"] - r["cost_per_1k_eur"]]], color=INK, capsize=4, lw=1.2)
        ax.text(r["cost_per_1k_hi"] + 3, yi, f"{r['cost_per_1k_eur']:.0f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, names)
    _style(ax, "Total cost per 1,000 test txns (95% bootstrap CI)", "EUR per 1,000 txns", "")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
