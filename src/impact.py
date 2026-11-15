"""Calibration check, dollar impact and active-window sensitivity (task 12).

Run `python -m src.impact` to fit the tabular model, print everything below and
save the tables and the calibration chart to results/.

1. Calibration: do predicted probabilities match how often restaurants close?
   The test set is split into tenths by predicted risk; each tenth's average
   prediction is compared with its actual closure rate.

2. Dollar impact, for a lender with PORTFOLIO_LOANS restaurant loans that
   reviews the riskiest REVIEW_SHARE the model flags each year.
   A closure is not a default: owners sell, repay or close debt-free. SBA data
   (src/sba.py) gives the annual charge-off rate on restaurant loans, so
       expected defaults a year = PORTFOLIO_LOANS * ANNUAL_CHARGE_OFF_RATE
       loss per default         = LOAN_SIZE * LENDER_LOSS_RATE
   Assuming defaults are spread evenly across closures, the share of losses in
   the flagged group equals the share of closures in it (recall), so
       loss avoided = recall * expected defaults * loss per default.
   "Avoided" assumes reviewing a flagged loan prevents its loss (declined,
   priced or restructured); it is the most the review can be worth. Flagged
   restaurants that do not close are false alarms; breakeven_per_false_alarm is
   the most the lender can give up on each before the review stops paying.

3. Active-window sensitivity: labels, features and model rebuilt with a 3, 6
   and 12 month active window (config.ACTIVE_WINDOW_MONTHS is 6).
"""

import numpy as np
import pandas as pd

from src import config
from src.evaluate import evaluate
from src.features import add_rating_features
from src.labels import train_test_split
from src.tabular import (DEFAULT_GROUPS, NO_RECENCY_GROUPS, build_features, fit_model,
                         load_inputs, predict)

RESULTS_DIR = config.PROJECT_ROOT / "results"

# ---- Lender assumptions -----------------------------------------------------
# From `python -m src.sba` (SBA 7(a) restaurant loans; rerun it to check).
LOAN_SIZE = 175_000             # median restaurant loan, FY2010-14
LENDER_LOSS_RATE = 0.19         # lender's loss on a charged-off loan after the SBA
                                # guarantee, as a share of the loan
TOTAL_LOSS_RATE = 0.66          # the same before the guarantee (an unguaranteed loan)
ANNUAL_CHARGE_OFF_RATE = 0.0126 # charge-offs per loan-year, restaurant loans, 2012-2019
PORTFOLIO_LOANS = 1_000
REVIEW_SHARE = 0.10

WINDOWS = (3, 6, 12)
LABELS = ("closed", "exclude")


# ---- Calibration --------------------------------------------------------------

def calibration_table(y_true, p, bins=10):
    """Average predicted probability vs actual closure rate in each tenth of risk."""
    df = pd.DataFrame({"closed": np.asarray(y_true), "predicted": np.asarray(p)})
    df["risk_group"] = pd.qcut(df["predicted"].rank(method="first"), bins, labels=range(1, bins + 1))
    return df.groupby("risk_group", observed=True).agg(
        restaurants=("closed", "size"),
        predicted=("predicted", "mean"),
        actual=("closed", "mean"),
    )


def plot_calibration(table, path, title):
    """Predicted vs actual closure rate per risk tenth, with the perfect-calibration line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, blue = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    top = max(table["predicted"].max(), table["actual"].max()) * 1.1
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot([0, top], [0, top], color=muted, lw=1, ls="--", zorder=1)
    ax.text(top * 0.62, top * 0.55, "perfectly calibrated", color=muted, ha="left", fontsize=9)
    ax.plot(table["predicted"], table["actual"], color=blue, lw=2, marker="o", ms=8,
            markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=2)
    ax.set(xlim=(0, top), ylim=(0, top))
    ax.set_xlabel("Predicted closure probability (average per tenth of restaurants)", color=muted)
    ax.set_ylabel("Share that actually closed in 2019", color=muted)
    ax.set_title(title, color=ink, loc="left", fontsize=11)
    step = 0.05 if top > 0.2 else 0.02
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.MultipleLocator(step))
        axis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(color=grid, lw=0.8)
    ax.tick_params(colors=muted)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---- Dollar impact --------------------------------------------------------------

def dollar_impact(recall, precision, review_share=REVIEW_SHARE, portfolio=PORTFOLIO_LOANS,
                  annual_charge_off_rate=ANNUAL_CHARGE_OFF_RATE, loan_size=LOAN_SIZE,
                  loss_rate=LENDER_LOSS_RATE):
    """Yearly losses avoided by reviewing the flagged share of a loan portfolio (see module docstring)."""
    flagged = portfolio * review_share
    expected_loss = portfolio * annual_charge_off_rate * loan_size * loss_rate
    avoided = recall * expected_loss
    false_alarms = flagged * (1 - precision)
    return {
        "loans_reviewed": flagged,
        "expected_loss": expected_loss,
        "loss_avoided": avoided,
        "share_of_loss_avoided": recall,
        "false_alarms": false_alarms,
        "breakeven_per_false_alarm": avoided / false_alarms if false_alarms else np.nan,
    }


def impact_table(evaluations, review_share=REVIEW_SHARE, **assumptions):
    """dollar_impact() for each evaluate() result, plus a random-review row."""
    pct = f"{review_share * 100:g}"
    rows = {}
    for r in evaluations:
        rows[r["model"]] = dollar_impact(r[f"recall_top{pct}"], r[f"precision_top{pct}"],
                                         review_share, **assumptions)
    rate = evaluations[0]["closure_rate"]
    rows["Random review"] = dollar_impact(review_share, rate, review_share, **assumptions)
    return pd.DataFrame(rows).T


def assumption_sensitivity(recall, precision):
    """Loss avoided by the model when one assumption at a time is changed."""
    base = dict(review_share=REVIEW_SHARE, annual_charge_off_rate=ANNUAL_CHARGE_OFF_RATE,
                loan_size=LOAN_SIZE, loss_rate=LENDER_LOSS_RATE)
    cases = {
        "Base case": {},
        "Charge-off rate 1.0% a year": {"annual_charge_off_rate": 0.010},
        "Charge-off rate 2.0% a year": {"annual_charge_off_rate": 0.020},
        "Average loan ($314k) not median": {"loan_size": 314_000},
        "No SBA guarantee (loss 66% of loan)": {"loss_rate": TOTAL_LOSS_RATE},
    }
    rows = {name: dollar_impact(recall, precision, **{**base, **change})
            for name, change in cases.items()}
    return pd.DataFrame(rows).T[["expected_loss", "loss_avoided", "breakeven_per_false_alarm"]]


# ---- Fitting --------------------------------------------------------------------------

def fit_and_score(restaurants, reviews, window, label, groups_list):
    """Train on the training year, score the test year; returns (test labels, {name: scores})."""
    dates = reviews[["business_id", "date"]]
    train, test = train_test_split(active_window_months=window, pre_cutoff_closures=label,
                                   restaurants=restaurants, reviews=dates)
    X_train = build_features(train, restaurants, reviews)
    X_test = build_features(test, restaurants, reviews)
    scores = {}
    for name, groups in groups_list.items():
        model = fit_model(X_train, train["closed"], groups=groups)
        scores[name] = pd.Series(predict(model, X_test), index=test["business_id"].to_numpy())
    return test, scores


def star_scores(test, reviews):
    """Star-rating rule: minus the average rating as of the cutoff (higher = riskier)."""
    rated = add_rating_features(test, reviews[["business_id", "date", "stars"]])
    return pd.Series(-rated["avg_stars"].to_numpy(), index=rated["business_id"].to_numpy())


def _money(df, cols):
    out = df.copy().astype(object)
    for c in cols:
        out[c] = df[c].map(lambda v: f"${v:,.0f}")
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    RESULTS_DIR.mkdir(exist_ok=True)
    restaurants, reviews = load_inputs()
    models = {"Tabular model": DEFAULT_GROUPS, "Tabular, no recency": NO_RECENCY_GROUPS}

    window_rows, impact_tables = [], {}
    for window in WINDOWS:
        for label in LABELS:
            test, scores = fit_and_score(restaurants, reviews, window, label,
                                         models if window == config.ACTIVE_WINDOW_MONTHS
                                         else {"Tabular model": DEFAULT_GROUPS})
            results = [evaluate(test, s, name) for name, s in scores.items()]
            r = results[0]
            window_rows.append({"window_months": window, "label": label, "restaurants": r["n"],
                                "closure_rate": r["closure_rate"],
                                "avg_precision": r["avg_precision"], "roc_auc": r["roc_auc"],
                                "lift_top10": r["lift_top10"], "brier": r["brier"]})
            if window != config.ACTIVE_WINDOW_MONTHS:
                continue

            # Calibration of the headline model.
            p = scores["Tabular model"].reindex(test["business_id"]).to_numpy()
            cal = calibration_table(test["closed"], p)
            cal.to_csv(RESULTS_DIR / f"calibration_{label}.csv")
            plot_calibration(cal, RESULTS_DIR / f"calibration_{label}.png",
                             f"Tabular model calibration, 2019 test set ({label!r} label)")
            print(f"\n=== Calibration, {label!r} label "
                  f"(predicted {p.mean():.1%} closures on average, actual {test['closed'].mean():.1%})")
            print(cal.round(3).to_string())

            # Dollar impact.
            results.append(evaluate(test, star_scores(test, reviews), "Star-rating rule",
                                    probabilities=False))
            table = impact_table(results)
            impact_tables[label] = table
            table.to_csv(RESULTS_DIR / f"impact_{label}.csv")
            print(f"\n=== Yearly impact per {PORTFOLIO_LOANS:,} restaurant loans, reviewing the "
                  f"riskiest {REVIEW_SHARE:.0%} ({label!r} label)")
            print(_money(table, ["expected_loss", "loss_avoided", "breakeven_per_false_alarm"])
                  .to_string(formatters={"share_of_loss_avoided": "{:.0%}".format}))

            head = table.loc["Tabular model"]
            pct = f"{REVIEW_SHARE * 100:g}"
            sens = assumption_sensitivity(r[f"recall_top{pct}"], r[f"precision_top{pct}"])
            sens.to_csv(RESULTS_DIR / f"impact_assumptions_{label}.csv")
            print(f"\nTabular model, one assumption changed at a time ({label!r} label):")
            print(_money(sens, sens.columns).to_string())

    windows = pd.DataFrame(window_rows)
    windows.to_csv(RESULTS_DIR / "window_sensitivity.csv", index=False)
    print("\n=== Active-window sensitivity, tabular model on the 2019 test set")
    print(windows.round(3).to_string(index=False))
