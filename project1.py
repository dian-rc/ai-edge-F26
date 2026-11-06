from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
FILES = {
    "business": DATA_DIR / "yelp_academic_dataset_business.json",
    "review": DATA_DIR / "yelp_academic_dataset_review.json",
}


def load_yelp(name, nrows=None):
    """Read the business or review file from data/ into a DataFrame.

    business is small (150,346 rows). review is ~7.0M rows / 5.3 GB, so pass
    nrows to load a sample, or use iter_reviews() to process all of it.
    """
    return pd.read_json(FILES[name], lines=True, nrows=nrows)


def iter_reviews(chunksize=200_000):
    """Yield the review file in DataFrame chunks, so it never sits in memory whole."""
    with pd.read_json(FILES["review"], lines=True, chunksize=chunksize) as reader:
        yield from reader


if __name__ == "__main__":
    business = load_yelp("business")
    review = load_yelp("review", nrows=100_000)

    for name, df in [("business", business), ("review", review)]:
        print(f"\n=== {name}: {df.shape} ===")
        print(df.dtypes)
        print(df.head(3))

    first_chunk = next(iter_reviews())
    print(f"\nFirst review chunk: {first_chunk.shape}")
