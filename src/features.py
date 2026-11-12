"""Leak-free review features, computed only from reviews on or before a cutoff.

Do not use `stars` or `review_count` from the business file: they are 2022
values, so they include reviews written after the cutoff and leak the future.

Typical use:
    from src.labels import train_test_split
    from src.features import add_rating_features

    train, test = train_test_split()
    train = add_rating_features(train)
    test = add_rating_features(test)
    # adds avg_stars and review_count, each as of the row's own cutoff
"""

import pandas as pd

from src.data import load_restaurant_reviews
from src.labels import _end_of_day

RATING_COLUMNS = ["avg_stars", "review_count"]


def rating_features(cutoff, reviews=None):
    """Average star rating and review count per business, from reviews dated on or before `cutoff`.

    reviews defaults to the business_id, date and stars columns of
    load_restaurant_reviews(); pass it in to reuse data already loaded.
    """
    if reviews is None:
        reviews = load_restaurant_reviews(columns=["business_id", "date", "stars"])
    cutoff_end = _end_of_day(pd.Timestamp(cutoff))
    before = reviews[reviews["date"].lt(cutoff_end)]
    return (before.groupby("business_id")["stars"]
            .agg(avg_stars="mean", review_count="size")
            .reset_index())


def add_rating_features(labels, reviews=None):
    """Join avg_stars and review_count onto a label table, each row at its own cutoff."""
    if reviews is None:
        reviews = load_restaurant_reviews(columns=["business_id", "date", "stars"])
    parts = []
    for cutoff, group in labels.groupby("cutoff", sort=False):
        parts.append(group.merge(rating_features(cutoff, reviews), on="business_id", how="left"))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    from src.labels import train_test_split

    reviews = load_restaurant_reviews(columns=["business_id", "date", "stars"])
    train, test = train_test_split(reviews=reviews[["business_id", "date"]])
    for name, df in [("train", add_rating_features(train, reviews)),
                     ("test", add_rating_features(test, reviews))]:
        print(f"{name}: cutoff {df['cutoff'].iloc[0].date()}, "
              f"missing avg_stars: {df['avg_stars'].isna().sum()}")
        print(df.groupby("closed")[RATING_COLUMNS].mean().round(3))
