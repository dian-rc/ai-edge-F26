"""Combined text-plus-tabular model and the model comparison (task 11).

The combined model is the tabular gradient boosting model (src.tabular) with
one extra feature, `text_score`: the review-text model's predicted closure
probability (src.text). This "stacking" design won a 5-fold cross-validation on
the training year (average precision: tabular 0.307, + text score 0.333,
+ 64 embedding components 0.325, + all 384 embedding numbers 0.322), and it
keeps the model explainable: the text enters as a single risk score.

To keep the training rows honest, their text_score is out-of-fold: each
training restaurant is scored by a text model fitted on the other four fifths
of the training year, never on itself. Test rows are scored by a text model
fitted on the whole training year.

Run `python -m src.combined` to compare, on the 2019 test set and for both labels:
    random review, star rating, recency, text, tabular, tabular + text
It prints the tables and saves to results/:
    model_comparison.csv          evaluate() metrics for every model and label
    model_comparison.png          average precision chart for the slides
    combined_vs_tabular.csv       bootstrap check that the gain from text is real
    impact_combined_<label>.csv   dollar impact (src.impact) including the combined model
    calibration_combined_<label>.csv / .png
                                  calibration of the combined model (src.impact's check)

Run time: the first run embeds about 1.4M reviews and caches them in models/
(about 12 minutes on an Apple-silicon GPU, 1 to 2 hours on a laptop CPU); after
that, about 1 to 5 minutes depending on the laptop.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.baseline import star_rating_scores
from src.evaluate import evaluate
from src.impact import calibration_table, impact_table, plot_calibration
from src.labels import train_test_split
from src.tabular import (DEFAULT_GROUPS, HGB_PARAMS, build_features, columns_for, fit_model,
                         load_inputs)
from src.tabular import predict as predict_tabular
from src.text import build_text_features, load_text_reviews

RESULTS_DIR = config.PROJECT_ROOT / "results"
# Regularisation of the text model, as chosen by cross-validation in task 9
# (src.text.fit_text_model picked C = 0.01 for every setting tried).
TEXT_C = 0.01
N_FOLDS = 5
N_BOOTSTRAP = 1000
LABELS = ("closed", "exclude")
MODEL_ORDER = ["Random review", "Star rating", "Recency", "Text", "Tabular", "Tabular + text"]


# ---- Text score -------------------------------------------------------------------

def _text_model():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=TEXT_C, max_iter=3000))


def text_scores(X_text_train, y_train, X_text_test, n_folds=N_FOLDS):
    """Text-model closure probabilities: out-of-fold for training rows, full fit for test rows."""
    y_train = np.asarray(y_train)
    train_scores = np.zeros(len(X_text_train))
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=config.RANDOM_SEED)
    for fit_rows, score_rows in folds.split(X_text_train, y_train):
        model = _text_model().fit(X_text_train.iloc[fit_rows], y_train[fit_rows])
        train_scores[score_rows] = model.predict_proba(X_text_train.iloc[score_rows])[:, 1]
    test_scores = _text_model().fit(X_text_train, y_train).predict_proba(X_text_test)[:, 1]
    return train_scores, test_scores


# ---- Combined model -------------------------------------------------------------------

def fit_combined(X_tab, text_score, y, groups=DEFAULT_GROUPS):
    """Tabular gradient boosting (same settings as src.tabular) plus the text_score feature."""
    X = X_tab[columns_for(X_tab, groups)].assign(text_score=np.asarray(text_score))
    model = HistGradientBoostingClassifier(categorical_features="from_dtype", **HGB_PARAMS)
    model.fit(X, y)
    model.feature_columns_ = list(X.columns)
    model.groups_ = groups
    return model


def predict_combined(model, X_tab, text_score):
    """Predicted probability of closure."""
    X = X_tab[columns_for(X_tab, model.groups_)].assign(text_score=np.asarray(text_score))
    return model.predict_proba(X[model.feature_columns_])[:, 1]


# ---- Comparison -------------------------------------------------------------------------

def score_all(label, restaurants, tab_reviews, text_reviews):
    """Test table and closure-risk scores (Series by business_id) for every model in MODEL_ORDER."""
    stars = tab_reviews[["business_id", "date", "stars"]]
    train, test = train_test_split(pre_cutoff_closures=label, restaurants=restaurants,
                                   reviews=tab_reviews[["business_id", "date"]])
    X_tab_train = build_features(train, restaurants, tab_reviews)
    X_tab_test = build_features(test, restaurants, tab_reviews)
    X_text_train = build_text_features(train, text_reviews)
    X_text_test = build_text_features(test, text_reviews)
    y = train["closed"].to_numpy()

    text_train, text_test = text_scores(X_text_train, y, X_text_test)
    tabular = fit_model(X_tab_train, y, groups=DEFAULT_GROUPS)
    combined = fit_combined(X_tab_train, text_train, y)

    ids = test["business_id"].to_numpy()
    rng = np.random.default_rng(config.RANDOM_SEED)
    scores = {
        "Random review": pd.Series(rng.random(len(test)), index=ids),
        "Star rating": star_rating_scores(test, stars),
        "Recency": pd.Series(X_tab_test["days_since_last"].to_numpy(), index=ids),
        "Text": pd.Series(text_test, index=ids),
        "Tabular": pd.Series(predict_tabular(tabular, X_tab_test), index=ids),
        "Tabular + text": pd.Series(predict_combined(combined, X_tab_test, text_test), index=ids),
    }
    return test, scores


def bootstrap_gain(test, scores_a, scores_b, n=N_BOOTSTRAP):
    """Average-precision gain of scores_b over scores_a, with a paired bootstrap over test restaurants.

    Each resample draws test restaurants with replacement and scores both models
    on the same draw, so the spread reflects which restaurants happened to be in
    the test year.
    """
    y = test["closed"].to_numpy()
    a = scores_a.reindex(test["business_id"]).to_numpy()
    b = scores_b.reindex(test["business_id"]).to_numpy()
    rng = np.random.default_rng(config.RANDOM_SEED)
    gains = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        gains[i] = average_precision_score(y[idx], b[idx]) - average_precision_score(y[idx], a[idx])
    return {
        "gain": average_precision_score(y, b) - average_precision_score(y, a),
        "ci_low": np.percentile(gains, 2.5),
        "ci_high": np.percentile(gains, 97.5),
        "share_of_resamples_positive": (gains > 0).mean(),
    }


def plot_comparison(table, path):
    """Average precision per model, one panel per label, combined model highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    accent, base, faint = "#2a78d6", "#a3a29d", "#d6d5d0"
    # Headline label ("exclude") first; see README.
    titles = {"exclude": "Headline: excluding likely pre-cutoff closures ('exclude')",
              "closed": "For reference: all closures ('closed')"}
    top = table["avg_precision"].max() * 1.18
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, facecolor=surface)
    for ax, label in zip(axes, titles):
        sub = table[table["label"] == label].set_index("model").loc[MODEL_ORDER]
        colors = [accent if m == "Tabular + text" else faint if m == "Random review" else base
                  for m in sub.index]
        y = np.arange(len(sub))[::-1]
        ax.set_facecolor(surface)
        ax.barh(y, sub["avg_precision"], height=0.55, color=colors, zorder=2)
        for yi, (model, v) in zip(y, sub["avg_precision"].items()):
            weight = "bold" if model == "Tabular + text" else "normal"
            ax.text(v + top * 0.012, yi, f"{v:.3f}", va="center", fontsize=9, color=ink,
                    fontweight=weight)
        ax.set_yticks(y, sub.index, color=ink, fontsize=10)
        ax.set_xlim(0, top)
        ax.set_title(titles[label], loc="left", fontsize=11, color=ink)
        ax.grid(axis="x", color=grid, lw=0.8, zorder=0)
        ax.tick_params(axis="x", colors=muted)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.set_xlabel("Average precision on 2019 closures (higher is better)", color=muted)
    fig.suptitle("Adding review text to the tabular model gives the best ranking of risky restaurants",
                 x=0.01, y=0.97, ha="left", fontsize=12, color=ink)
    fig.text(0.01, 0.875, "Random review scores about the closure rate (one random draw). "
             "2019 test set; models trained on 2018 closures.", fontsize=9, color=muted)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=200, facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    RESULTS_DIR.mkdir(exist_ok=True)
    restaurants, tab_reviews = load_inputs()
    # src.tabular's review columns plus stars for the star rule (already included).
    text_reviews = load_text_reviews()

    rows, gains = [], []
    for label in LABELS:
        test, scores = score_all(label, restaurants, tab_reviews, text_reviews)
        probabilistic = {"Text", "Tabular", "Tabular + text"}
        results = [evaluate(test, scores[m], m, probabilities=m in probabilistic)
                   for m in MODEL_ORDER]
        rows += [{**r, "label": label} for r in results]
        gains.append({"label": label, "comparison": "Tabular + text vs Tabular",
                      **bootstrap_gain(test, scores["Tabular"], scores["Tabular + text"])})

        impact = impact_table([r for r in results if r["model"] != "Random review"])
        impact.to_csv(RESULTS_DIR / f"impact_combined_{label}.csv")

        # Calibration of the combined model, with the same check as the tabular model (src.impact).
        p = scores["Tabular + text"].reindex(test["business_id"]).to_numpy()
        cal = calibration_table(test["closed"], p)
        cal.to_csv(RESULTS_DIR / f"calibration_combined_{label}.csv")
        plot_calibration(cal, RESULTS_DIR / f"calibration_combined_{label}.png",
                         f"Tabular + text model calibration, 2019 test set ({label!r} label)")
        print(f"\n=== Calibration of Tabular + text, {label!r} label "
              f"(predicted {p.mean():.1%} closures on average, actual {test['closed'].mean():.1%})")
        print(cal.round(3).to_string())
        print(f"\n=== {label!r} label: dollar impact per 1,000 loans, reviewing the riskiest 10%")
        print(impact[["loss_avoided", "share_of_loss_avoided", "false_alarms",
                      "breakeven_per_false_alarm"]].round(3).to_string())

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(gains).to_csv(RESULTS_DIR / "combined_vs_tabular.csv", index=False)
    plot_comparison(table, RESULTS_DIR / "model_comparison.png")

    print("\n=== Model comparison, 2019 test set")
    print(table.pivot(index="model", columns="label",
                      values=["avg_precision", "roc_auc", "lift_top10"])
          .loc[MODEL_ORDER].round(3).to_string())
    print("\n=== Gain from adding text (paired bootstrap, 95% interval)")
    print(pd.DataFrame(gains).round(4).to_string(index=False))
