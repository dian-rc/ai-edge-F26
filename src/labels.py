"""Active-restaurant filter, closure label and train/test split.

Typical use:
    from src.labels import train_test_split

    train, test = train_test_split()
    # each has one row per restaurant: business_id, cutoff, closed (0/1)

Build features for a row only from data on or before its cutoff.

Target: closed = 1 means Yelp flags the restaurant closed (is_open == 0) and it
has no reviews after the end of the horizon. Because Yelp only gives a closed
flag, not a closing date, this includes restaurants that may have closed
shortly before the cutoff.

Label rules, for a cutoff date and the settings in src/config.py:
- Active: at least one review in the ACTIVE_WINDOW_MONTHS up to the cutoff.
- closed = 1: flagged closed and its last review falls in the HORIZON_MONTHS
  after the cutoff.
- closed = 0: flagged open, or flagged closed but it still had reviews after the
  horizon (so it survived the horizon).
- Pre-cutoff closures: flagged closed with its last review on or before the
  cutoff. It may already have closed by the cutoff. PRE_CUTOFF_CLOSURES decides
  what happens to them:
    "closed"  (default) label them 1, consistent with the target above;
    "exclude" drop them, as neither a clean positive nor a clean negative.
              This is the headline label in the README; results are reported
              for both.
"""

import pandas as pd

from src import config
from src.data import load_restaurant_reviews, load_restaurants

LABEL_COLUMNS = ["business_id", "cutoff", "closed"]
PRE_CUTOFF_OPTIONS = ("closed", "exclude")


def _end_of_day(day):
    """The instant just after `day` ends, so a review on the cutoff day counts as before it."""
    return day.normalize() + pd.Timedelta(days=1)


def build_labels(cutoff=config.CUTOFF,
                 active_window_months=config.ACTIVE_WINDOW_MONTHS,
                 horizon_months=config.HORIZON_MONTHS,
                 pre_cutoff_closures=config.PRE_CUTOFF_CLOSURES,
                 restaurants=None, reviews=None, return_details=False):
    """One row per active restaurant at `cutoff`, with its 0/1 closure label.

    pre_cutoff_closures is "closed" (label flagged-closed restaurants with no
    review after the cutoff as 1) or "exclude" (drop them).

    restaurants and reviews default to load_restaurants() and the business_id
    and date columns of load_restaurant_reviews(); pass them in to reuse data
    already loaded. return_details=True also returns the pre-cutoff closures
    and the last-review dates, for checking the label.
    """
    if pre_cutoff_closures not in PRE_CUTOFF_OPTIONS:
        raise ValueError(f"pre_cutoff_closures must be one of {PRE_CUTOFF_OPTIONS}, "
                         f"got {pre_cutoff_closures!r}")
    if restaurants is None:
        restaurants = load_restaurants()
    if reviews is None:
        reviews = load_restaurant_reviews(columns=["business_id", "date"])

    cutoff = pd.Timestamp(cutoff)
    cutoff_end = _end_of_day(cutoff)
    window_start = cutoff_end - pd.DateOffset(months=active_window_months)
    horizon_end = cutoff_end + pd.DateOffset(months=horizon_months)

    in_window = reviews["date"].ge(window_start) & reviews["date"].lt(cutoff_end)
    active_ids = reviews.loc[in_window, "business_id"].unique()
    last_review = reviews.groupby("business_id")["date"].max().rename("last_review")

    df = (restaurants.loc[restaurants["business_id"].isin(active_ids), ["business_id", "is_open"]]
          .join(last_review, on="business_id"))

    flagged_closed = df["is_open"].eq(0)
    closed_in_horizon = flagged_closed & df["last_review"].ge(cutoff_end) & df["last_review"].lt(horizon_end)
    pre_cutoff = flagged_closed & df["last_review"].lt(cutoff_end)

    df["cutoff"] = cutoff
    if pre_cutoff_closures == "closed":
        df["closed"] = (closed_in_horizon | pre_cutoff).astype(int)
        keep = pd.Series(True, index=df.index)
    else:
        df["closed"] = closed_in_horizon.astype(int)
        keep = ~pre_cutoff
    labels = df.loc[keep, LABEL_COLUMNS].reset_index(drop=True)

    if return_details:
        return labels, df.loc[pre_cutoff].reset_index(drop=True), df.reset_index(drop=True)
    return labels


def train_test_split(active_window_months=config.ACTIVE_WINDOW_MONTHS,
                     horizon_months=config.HORIZON_MONTHS,
                     pre_cutoff_closures=config.PRE_CUTOFF_CLOSURES,
                     restaurants=None, reviews=None):
    """Time-based split: train at TRAIN_CUTOFF, test at CUTOFF.

    A restaurant can appear in both, as two snapshots a year apart, but the
    training rows only use data up to TRAIN_CUTOFF plus labels from the year
    after it, which ends at CUTOFF. Nothing from the test year leaks into training.

    The training cutoff is CUTOFF minus horizon_months, so a non-default
    horizon still keeps the training labels before CUTOFF.
    """
    if restaurants is None:
        restaurants = load_restaurants()
    if reviews is None:
        reviews = load_restaurant_reviews(columns=["business_id", "date"])
    train_cutoff = config.CUTOFF - pd.DateOffset(months=horizon_months)
    kwargs = dict(active_window_months=active_window_months, horizon_months=horizon_months,
                  pre_cutoff_closures=pre_cutoff_closures,
                  restaurants=restaurants, reviews=reviews)
    train = build_labels(train_cutoff, **kwargs)
    test = build_labels(config.CUTOFF, **kwargs)
    return train, test


def summarize(labels):
    """Row count and closure rate, for a quick check of a label table."""
    return pd.Series({"restaurants": len(labels), "closed": labels["closed"].sum(),
                      "closure_rate": round(labels["closed"].mean(), 4)})


if __name__ == "__main__":
    restaurants = load_restaurants()
    reviews = load_restaurant_reviews(columns=["business_id", "date"])
    table = {}
    for option in PRE_CUTOFF_OPTIONS:
        train, test = train_test_split(pre_cutoff_closures=option,
                                       restaurants=restaurants, reviews=reviews)
        for name, labels in [("train", train), ("test", test)]:
            s = summarize(labels)
            table[(option, name)] = {"restaurants": f"{int(s['restaurants']):,}",
                                     "closed": f"{int(s['closed']):,}",
                                     "closure_rate": f"{s['closure_rate']:.1%}"}
    print(f"PRE_CUTOFF_CLOSURES (default: {config.PRE_CUTOFF_CLOSURES!r})")
    print(pd.DataFrame(table))
