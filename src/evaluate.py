"""Shared evaluation function: scores any model on the same test set (task 6).

Every model (star-rating rule, tabular, text, combined) is scored with
evaluate(), so the numbers in the slides are comparable.

Typical use:
    from src.evaluate import evaluate, compare
    from src.labels import train_test_split

    train, test = train_test_split()
    scores = pd.Series(predict(model, X_test), index=test["business_id"])
    results = [
        evaluate(test, scores, name="Gradient boosting"),
        evaluate(test, -avg_stars, name="Star-rating rule", probabilities=False),
    ]
    compare(results)

Scores: higher means more likely to close. Pass either
- a Series indexed by business_id (safest: matched to the test set by id, and
  extra ids are ignored, so one set of scores can be checked on both labels), or
- an array in the same row order as `labels`.

The test set is whatever `labels` you pass, normally the test table from
src.labels.train_test_split(). To score the "exclude" label, pass the test table
built with pre_cutoff_closures="exclude"; its rows are a subset of the default
test set with the same labels, so the same Series of scores works for both.

Metrics (one dict per model):
    avg_precision   headline metric. Closures are rare, so this is more
                    informative than accuracy or ROC AUC. A random ranking
                    scores the closure rate.
    roc_auc         chance a random closed restaurant is ranked above a random
                    open one (0.5 = random).
    brier           mean squared error of the predicted probabilities (lower is
                    better). Only for models that output probabilities.
    precision_topK  share of the riskiest K% of restaurants that closed: what a
                    credit officer reviewing that list would find.
    recall_topK     share of all closures caught in the riskiest K%.
    lift_topK       precision_topK divided by the closure rate (1 = no better
                    than reviewing restaurants at random).
"""

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# Shares of the test set flagged as riskiest, for the precision/recall/lift columns.
TOP_FRACTIONS = (0.05, 0.10, 0.20)


def _align(labels, scores):
    """Return (y_true, y_score) as arrays in the row order of `labels`."""
    if isinstance(scores, pd.Series):
        if scores.index.has_duplicates:
            raise ValueError("scores has duplicate business_ids")
        missing = ~labels["business_id"].isin(scores.index)
        if missing.all():
            raise ValueError("scores is a Series, but its index holds no test business_ids. "
                             "Index it by business_id, or pass .to_numpy() if it is already "
                             "in the row order of labels")
        if missing.any():
            raise ValueError(f"scores is missing {missing.sum():,} of the {len(labels):,} "
                             "test restaurants; score every row of the test set")
        y_score = scores.reindex(labels["business_id"]).to_numpy(dtype=float)
    else:
        y_score = np.asarray(scores, dtype=float)
        if y_score.shape != (len(labels),):
            raise ValueError(f"scores has shape {y_score.shape}, expected ({len(labels)},): "
                             "one score per test row, or pass a Series indexed by business_id")
    if np.isnan(y_score).any():
        raise ValueError(f"scores has {np.isnan(y_score).sum():,} missing values")
    return labels["closed"].to_numpy(dtype=int), y_score


def _top_mask(y_score, frac):
    """True for the riskiest ceil(frac * n) rows. Ties are broken by row order."""
    k = math.ceil(frac * len(y_score))
    order = np.argsort(-y_score, kind="stable")
    mask = np.zeros(len(y_score), dtype=bool)
    mask[order[:k]] = True
    return mask


def evaluate(labels, scores, name=None, probabilities=True, top_fractions=TOP_FRACTIONS):
    """Score one model on a test set; returns a dict of metrics (see module docstring).

    labels: test table with business_id and closed columns (from train_test_split).
    scores: closure risk per restaurant, higher = riskier (Series by business_id, or array).
    probabilities: False for rules or scores that are not probabilities (e.g. minus
        the star rating); the Brier score is then left out.
    """
    y_true, y_score = _align(labels, scores)
    if y_true.min() == y_true.max():
        raise ValueError("the test set has only one class; metrics are undefined")
    if probabilities and (y_score.min() < 0 or y_score.max() > 1):
        raise ValueError("probabilities=True but scores fall outside [0, 1]; "
                         "pass probabilities=False for a rule or raw score")

    rate = y_true.mean()
    result = {
        "model": name,
        "n": len(y_true),
        "closed": int(y_true.sum()),
        "closure_rate": rate,
        "avg_precision": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score) if probabilities else np.nan,
    }
    for frac in top_fractions:
        top = _top_mask(y_score, frac)
        pct = f"{frac * 100:g}"
        precision = y_true[top].mean()
        result[f"precision_top{pct}"] = precision
        result[f"recall_top{pct}"] = y_true[top].sum() / y_true.sum()
        result[f"lift_top{pct}"] = precision / rate
    return result


def compare(results, digits=3):
    """Table of evaluate() results, one row per model, sorted by average precision."""
    table = pd.DataFrame(results).set_index("model")
    return table.sort_values("avg_precision", ascending=False).round(digits)
