"""Summary of the results: draws every chart used in the paper and the stakeholder slides.

Run after the two notebooks have been executed (it only READS the files they saved in outputs/final/;
nothing is trained or tested again and no database is needed):

    python summarize_results.py

Output: report/figures/*.png
    pipeline_compact.png      icon-and-arrow pipeline (paper Figure 2, slide "From data to action")
    eda.png                   distribution of the target and share of volume per product
    batch_r2.png              out-of-sample R2 for each training week
    ladder.png                R2 as each feature enters the equation (paper Figure 1)
    coefficients.png          standardised weights of the final features
    product_scatter.png       predicted vs actual mean demand per product (validation week 9)
    inventory.png             forecast error as inventory impact (paper Figure 3, slide)
    forecast_vs_history.png   weeks 3-9 actual and weeks 10-11 forecast per product

Needs: numpy, pandas, matplotlib (already in requirements.txt).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parent
OUT, FIG = ROOT / "outputs" / "final", ROOT / "report" / "figures"
NAVY, BLUE, GREEN, ORANGE, RED, GREY, LIGHT = "#1f3a5f", "#2f6db5", "#2e8b57", "#e8833a", "#c0392b", "#6b7280", "#f3f6fa"


def savefig(fig, path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", **kw)
    plt.close(fig)
    return path


def load_results(r2_fallback):
    R = dict(meta=None, metrics=None, m2=None, mcf=None, base=None, inv=None, hyper=None, weeks=None, scope=None,
             r2=None, r2_source="n/a", log=None, prod=None, dist=None, eda=None, test=None, coefs=None, std_w=None,
             intercept=None, feat_names=None, formula=None, ladder=None, cand=None, removed=None, train_gd=None, n_features=None)
    p = OUT / "metrics_final.json"
    if p.exists():
        j = json.loads(p.read_text(encoding="utf-8"))
        R.update(meta=j, metrics=j["validation_1_week_ahead"], m2=j.get("validation_2_weeks_ahead"), mcf=j.get("closed_form"),
                 base=j.get("baseline"), inv=j.get("inventory"), hyper=j.get("hyperparameters"), weeks=j.get("weeks"),
                 scope=j.get("scope"), formula=j.get("formula"), ladder=j.get("ladder"), cand=j.get("candidates"), removed=j.get("removed_features"), train_gd=j.get("train_gd"))
        R["r2"], R["r2_source"] = R["metrics"]["R2_log"], "measured"
    elif r2_fallback is not None:
        R["r2"], R["r2_source"] = r2_fallback, "reported by the author"
    for key, fn in [("log", "training_log.csv"), ("prod", "product_results_week9.csv"), ("dist", "val_distribution.csv")]:
        if (OUT / fn).exists():
            R[key] = pd.read_csv(OUT / fn)
    for key, fn in [("eda", "eda_summary.json"), ("test", "test_summary.json")]:
        if (OUT / fn).exists():
            R[key] = json.loads((OUT / fn).read_text(encoding="utf-8"))
    cp = OUT / "coefficients_final.csv"
    if cp.exists():
        c = pd.read_csv(cp)
        R.update(coefs=c["weight (original scale)"].to_numpy(), std_w=c["weight (standardised)"].to_numpy(),
                 feat_names=list(c["feature"]), n_features=len(c))
    if R["n_features"] is None:
        R["n_features"] = 4
    return R


def fig_eda(R):
    d, p = R["dist"], R["prod"]
    if d is None and p is None:
        return None
    n = (d is not None) + (p is not None)
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n, 3.7))
    axes = np.atleast_1d(axes); k = 0
    if d is not None:
        ax = axes[k]; k += 1
        ax.fill_between(d.bin_left, d.actual, color=BLUE, alpha=.6)
        ax.set_title("Distribution of log1p(Adjusted demand)\n(validation week)", fontsize=9.5)
        ax.set_xlabel("log1p(units)"); ax.set_ylabel("Density")
    if p is not None:
        ax = axes[k]
        s = p.sort_values("Actual_Total", ascending=False)["Actual_Total"].to_numpy()
        share = np.cumsum(s) / s.sum() * 100
        ax.plot(np.arange(1, len(s) + 1) / len(s) * 100, share, color=ORANGE, lw=2)
        ax.plot([0, 100], [0, 100], "--", color=GREY, lw=1)
        ax.set_title("Few products drive most of the volume\n(cumulative share of demand)", fontsize=9.5)
        ax.set_xlabel("% of products (largest first)"); ax.set_ylabel("% of total demand")
    for a in axes:
        a.grid(alpha=.3)
    fig.tight_layout()
    return savefig(fig, FIG / "eda.png", dpi=180)


def fig_batch_r2(R):
    L = R["log"]
    if L is None:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(L["week"].astype(str), L["R2_before_learning"], "o-", color=BLUE, lw=2, label="Training weeks (before learning)")
    if R["r2"] is not None and R["weeks"]:
        ax.plot([str(R["weeks"]["val"])], [R["r2"]], "*", color=RED, ms=16, label="Hold-out validation week")
    ax.set_xlabel("Week (one week = one batch)"); ax.set_ylabel("R\u00b2 on log(1 + demand)")
    ax.set_title("Out-of-sample R\u00b2 improves as history grows", fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=8)
    return savefig(fig, FIG / "batch_r2.png", dpi=180)


def fig_product_scatter(R):
    p = R["prod"]
    if p is None:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    q = p[(p.Actual_Mean > 0) & (p.Predicted_Mean > 0)]
    ax.scatter(q.Actual_Mean, q.Predicted_Mean, s=np.clip(q.n_rows / q.n_rows.max() * 120, 6, 120), alpha=.45, color=BLUE)
    lim = [min(q.Actual_Mean.min(), q.Predicted_Mean.min()), max(q.Actual_Mean.max(), q.Predicted_Mean.max())]
    ax.plot(lim, lim, "r--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Actual mean demand per order line"); ax.set_ylabel("Predicted mean demand")
    ax.set_title(f"Each dot is one product ({len(q):,} products)", fontsize=10); ax.grid(alpha=.3)
    return savefig(fig, FIG / "product_scatter.png", dpi=180)


def fig_inventory(R):
    inv = R["inv"]
    if inv is None:
        return None
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.bar(["Stockout risk\n(prediction < actual)", "Excess-stock risk\n(prediction > actual)"],
           [inv["under_forecast_pct_of_demand"], inv["over_forecast_pct_of_demand"]], color=[RED, ORANGE])
    ax.set_ylim(0, max(inv["under_forecast_pct_of_demand"], inv["over_forecast_pct_of_demand"]) * 1.2)
    ax.set_ylabel("% of actual demand"); ax.set_title("Forecast error as inventory impact", fontsize=10)
    for i, v in enumerate([inv["under_forecast_pct_of_demand"], inv["over_forecast_pct_of_demand"]]):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    return savefig(fig, FIG / "inventory.png", dpi=180)


def fig_ladder(R):
    if not R.get("ladder"):
        return None
    L = pd.DataFrame(R["ladder"])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(L["step"], L["R2_train"], "o-", color="#2f6db5", label="training")
    ax.plot(L["step"], L["R2_valid"], "s--", color="#e8833a", label="validation")
    k = R["n_features"]
    ax.axvline(k, color="#2e8b57", ls=":"); ax.text(k + .15, ax.get_ylim()[0] + .02, f"final: {k} features", color="#2e8b57", fontsize=8)
    ax.set_xlabel("number of features in the equation"); ax.set_ylabel("R\u00b2 on log(1 + demand)"); ax.grid(alpha=.3); ax.legend(frameon=False)
    ax.set_title("R\u00b2 as each feature enters the equation", fontsize=10)
    return savefig(fig, FIG / "ladder.png", dpi=180)


def fig_coefs(R):
    if R["std_w"] is None:
        return None
    s = pd.Series(R["std_w"], index=R["feat_names"]); s = s.reindex(s.abs().sort_values(ascending=False).index)[:10][::-1]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.barh(s.index, s.values, color=[GREEN if v > 0 else RED for v in s.values])
    ax.axvline(0, color="k", lw=.8); ax.set_title("Top 10 features (standardised weights)", fontsize=10)
    ax.set_xlabel("Weight (effect on log demand per 1 std of the feature)")
    return savefig(fig, FIG / "coefficients.png", dpi=180)


def fig_pipeline_compact(R):
    """Compact data-to-action pipeline with icons, arrows and a feedback loop. Solid green = completed, dashed orange = planned."""
    nf = R["n_features"]
    nodes = [("Data origin", "Kaggle files\n(train, test)", "db", True),
             ("SQL clean\n+ filter", "rename, NULL checks,\n10 best-selling products", "funnel", True),
             ("Weekly chunks\n+ EDA", "all rows, never more\nthan 1M at a time", "stack", True),
             ("History\nfeatures", f"{nf} features,\nno leakage", "grid", True),
             ("Gradient\nDescent", "linear model of\nlog1p(demand)", "curve", True),
             ("Evaluate", "week 9 hold-out:\nR\u00b2, RMSLE", "check", True),
             ("Save, reload\n+ API", "frozen model served\nby a local API", "disk", True),
             ("Decision", "delivery quantity\nper store", "arrow", False),
             ("Feedback", "track stockouts\nand returns", "loop", False)]
    fig, ax = plt.subplots(figsize=(15, 5.0)); ax.set_xlim(-0.7, 8.7); ax.set_ylim(-0.75, 2.35); ax.axis("off"); ax.set_aspect("equal", adjustable="datalim")
    yc = 1.05

    def icon(kind, cx, cy, col):
        if kind == "db":
            ax.add_patch(Rectangle((cx - .17, cy - .14), .34, .27, fc=col, ec="none"))
            ax.add_patch(Ellipse((cx, cy - .14), .34, .12, fc=col, ec="none")); ax.add_patch(Ellipse((cx, cy + .13), .34, .12, fc=col, ec="white", lw=1))
        elif kind == "funnel":
            ax.add_patch(Polygon([(cx - .2, cy + .17), (cx + .2, cy + .17), (cx + .04, cy - .02), (cx + .04, cy - .2), (cx - .04, cy - .2), (cx - .04, cy - .02)], fc=col, ec="none"))
        elif kind == "stack":
            for k in range(3):
                ax.add_patch(Rectangle((cx - .17 + k * .04, cy - .17 + k * .1), .26, .09, fc=col, ec="white", lw=.8))
        elif kind == "grid":
            for a in range(3):
                for b in range(3):
                    ax.add_patch(Rectangle((cx - .19 + a * .13, cy - .19 + b * .13), .1, .1, fc=col, ec="none"))
        elif kind == "curve":
            xs = np.linspace(-.2, .2, 40); ax.plot(cx + xs, cy - .15 + (xs / .2) ** 2 * .32, color=col, lw=2.4)
            ax.plot([cx - .12], [cy - .15 + (.12 / .2) ** 2 * .32], "o", color=col, ms=5); ax.plot([cx], [cy - .15], "o", color=ORANGE, ms=5)
        elif kind == "check":
            ax.plot([cx - .13, cx - .03, cx + .15], [cy, cy - .1, cy + .13], color=col, lw=3.4, solid_capstyle="round")
        elif kind == "disk":
            ax.add_patch(Rectangle((cx - .15, cy - .15), .3, .3, fc=col, ec="none")); ax.add_patch(Rectangle((cx - .08, cy + .02), .16, .13, fc="white", ec="none"))
        elif kind == "arrow":
            ax.add_patch(FancyArrowPatch((cx - .2, cy), (cx + .2, cy), arrowstyle="-|>", mutation_scale=20, color=col, lw=3))
        elif kind == "loop":
            ax.add_patch(Arc((cx, cy), .32, .32, theta1=40, theta2=320, color=col, lw=3))
            ax.add_patch(FancyArrowPatch((cx + .11, cy + .13), (cx + .17, cy + .06), arrowstyle="-|>", mutation_scale=14, color=col, lw=2))

    for i, (t, sub, kind, done) in enumerate(nodes):
        col = GREEN if done else ORANGE
        ax.add_patch(Circle((i, yc), .38, fc="white", ec=col, lw=2.8, ls="-" if done else (0, (4, 3)), zorder=2))
        n_p, n_l = len(ax.patches), len(ax.lines)
        icon(kind, i, yc, BLUE if done else ORANGE)
        for art in ax.patches[n_p:] + ax.lines[n_l:]:
            art.set_zorder(3)
        ax.text(i, yc + .52, t, ha="center", va="bottom", fontsize=9.6, fontweight="bold", color=NAVY)
        ax.text(i, yc - .5, sub, ha="center", va="top", fontsize=7.8, color="#444", zorder=5, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
        ax.text(i - .31, yc + .31, str(i + 1), ha="center", va="center", fontsize=7.5, color="white", fontweight="bold",
                bbox=dict(boxstyle="circle,pad=.18", fc=col, ec="none"), zorder=4)
        if i < len(nodes) - 1:
            ax.add_patch(FancyArrowPatch((i + .41, yc), (i + .59, yc), arrowstyle="-|>", mutation_scale=14, color=GREY, lw=1.8, zorder=1))
    fy = -0.28                                                    # feedback loop: from node 9 back to node 3
    ax.plot([8, 8, 2], [yc - .38, fy, fy], color=ORANGE, lw=2.2, ls=(0, (4, 3)), zorder=1)
    ax.add_patch(FancyArrowPatch((2, fy), (2, yc - .38), arrowstyle="-|>", mutation_scale=14, color=ORANGE, lw=2.2, ls=(0, (4, 3))))
    ax.text(5, fy - .06, "each new week of data is added to the history, the model is retrained and the forecast is reviewed",
            ha="center", va="top", fontsize=8.2, color=ORANGE, style="italic")
    ax.plot([6.6, 6.95], [2.22, 2.22], color=GREEN, lw=3); ax.text(7.0, 2.22, "completed", va="center", fontsize=8.5)
    ax.plot([7.65, 8.0], [2.22, 2.22], color=ORANGE, lw=3, ls=(0, (3, 2))); ax.text(8.05, 2.22, "planned", va="center", fontsize=8.5)
    return savefig(fig, FIG / "pipeline_compact.png", dpi=200)


def fig_forecast_vs_history(R):
    wp, tp = OUT / "weekly_product_summary.csv", OUT / "test_predictions_by_product_week.csv"
    if not (wp.exists() and tp.exists()):
        print("  (forecast_vs_history skipped: run notebooks/test_full_filtered.ipynb first)")
        return None
    hist, g = pd.read_csv(wp), pd.read_csv(tp)
    prods = list(g.groupby("Product_ID").Predicted_Mean.mean().sort_values(ascending=False).index)
    n = len(prods); ncols = 5; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), sharex=True, squeeze=False)
    for ax, pid in zip(axes.ravel(), prods):
        h = hist[hist.Product_ID == pid]; f = g[g.Product_ID == pid]
        ax.plot(h.Week, h["mean"], "o-", color=BLUE, label="actual")
        ax.plot(f.Week, f.Predicted_Mean, "s--", color=ORANGE, label="forecast")
        if len(h) and len(f):
            ax.plot([h.Week.iloc[-1], f.Week.iloc[0]], [h["mean"].iloc[-1], f.Predicted_Mean.iloc[0]], ":", color="gray")
        ax.set_title(f"Product {pid}", fontsize=10); ax.grid(alpha=.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Mean demand per order line: actual (weeks 3-9) and forecast (weeks 10-11)", y=1.0)
    fig.tight_layout()
    return savefig(fig, FIG / "forecast_vs_history.png", dpi=180)


def main():
    if not (OUT / "metrics_final.json").exists():
        sys.exit("outputs/final/metrics_final.json not found. Run notebooks/train_full_filtered.ipynb first.")
    FIG.mkdir(parents=True, exist_ok=True)
    R = load_results(None)
    steps = [fig_pipeline_compact, fig_eda, fig_batch_r2, fig_ladder, fig_coefs, fig_product_scatter, fig_inventory, fig_forecast_vs_history]
    for fn in steps:
        p = fn(R)
        print(f"{'saved ' + str(p.relative_to(ROOT)) if p else 'skipped'}  <- {fn.__name__}")
    print("Done: figures are in", FIG.relative_to(ROOT))


if __name__ == "__main__":
    main()
