"""Star-rating baseline (task 7): rank restaurants by their Yelp rating, lowest first.

This is the rule a credit officer could apply without any model: treat a
lower-rated restaurant as riskier. The models must beat it to be worth using.

The rating is the average of the restaurant's reviews on or before the cutoff
(src.features.rating_features), never the business file's `stars`, which is a
2022 value and includes later reviews.

Two versions:
    star_rating_scores(test)                -> exact average, e.g. 3.87
    star_rating_scores(test, rounded=True)  -> rounded to the nearest half star,
                                               as Yelp displays it (3.87 -> 4.0)

Run `python -m src.baseline` to print both on the test set, for both labels.
"""

import numpy as np
import pandas as pd

from src.features import add_rating_features


def star_rating_scores(labels, reviews=None, rounded=False):
    """Risk score per restaurant: minus its average star rating as of its cutoff.

    Returns a Series indexed by business_id (higher = riskier), ready for
    src.evaluate.evaluate(..., probabilities=False). reviews needs the
    business_id, date and stars columns; pass it in to reuse data already loaded.
    """
    rated = add_rating_features(labels, reviews)
    stars = rated["avg_stars"]
    if stars.isna().any():
        # Every active restaurant has a review before its cutoff, so this means
        # the labels and reviews do not match.
        raise ValueError(f"{stars.isna().sum():,} restaurants have no reviews before the cutoff")
    if rounded:
        stars = np.round(stars * 2) / 2
    return pd.Series(-stars.to_numpy(), index=rated["business_id"], name="star_rating_risk")


if __name__ == "__main__":
    from src.data import load_restaurant_reviews
    from src.evaluate import compare, evaluate
    from src.labels import train_test_split

    reviews = load_restaurant_reviews(columns=["business_id", "date", "stars"])
    dates = reviews[["business_id", "date"]]
    pd.set_option("display.width", 200)
    for option in ("closed", "exclude"):
        _, test = train_test_split(pre_cutoff_closures=option, reviews=dates)
        results = [
            evaluate(test, star_rating_scores(test, reviews), "Average stars",
                     probabilities=False),
            evaluate(test, star_rating_scores(test, reviews, rounded=True),
                     "Stars rounded to half (as Yelp displays)", probabilities=False),
        ]
        print(f"\nLabel: {option!r}")
        print(compare(results).T)
