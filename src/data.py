"""Shared loaders for the Yelp files in data/.

Typical use:
    from src.data import load_restaurants, load_restaurant_reviews

    restaurants = load_restaurants()
    reviews = load_restaurant_reviews(columns=["business_id", "date", "stars"])

load_restaurant_reviews() reads data/restaurant_reviews.parquet. Build that file
once on each laptop with:  python -m src.data
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config

# Columns kept in the restaurant review cache, with fixed types so every chunk
# writes to the same Parquet schema.
REVIEW_SCHEMA = pa.schema([
    ("review_id", pa.string()),
    ("user_id", pa.string()),
    ("business_id", pa.string()),
    ("stars", pa.float64()),
    ("useful", pa.int64()),
    ("funny", pa.int64()),
    ("cool", pa.int64()),
    ("text", pa.string()),
    ("date", pa.timestamp("ns")),
])


def load_business():
    """Read the business file (150,346 rows, small enough to load whole)."""
    return pd.read_json(config.BUSINESS_PATH, lines=True)


def is_restaurant(categories):
    """True where RESTAURANT_CATEGORY is one of the comma-separated categories.

    An exact match, so "Restaurant Supplies" does not count.
    """
    cats = categories.fillna("").str.split(r",\s*")
    return cats.apply(lambda c: config.RESTAURANT_CATEGORY in c)


def load_restaurants():
    """The business file, restaurants only."""
    business = load_business()
    return business[is_restaurant(business["categories"])].reset_index(drop=True)


def iter_reviews(chunksize=200_000, columns=None):
    """Yield the full review file (~7.0M rows, 5.3 GB) in DataFrame chunks.

    The file never sits in memory whole. Pass columns, e.g.
    ["business_id", "date"], to drop the review text from each chunk and
    save memory. The date column is parsed to datetime.
    """
    with pd.read_json(config.REVIEW_PATH, lines=True, chunksize=chunksize) as reader:
        for chunk in reader:
            yield chunk[columns] if columns is not None else chunk


def build_restaurant_reviews(chunksize=200_000):
    """One pass over the review file, keeping restaurant reviews only.

    Writes config.RESTAURANT_REVIEWS_PATH. Takes several minutes; run it once
    per laptop, and again only if the restaurant definition changes.
    """
    restaurant_ids = set(load_restaurants()["business_id"])
    tmp_path = config.RESTAURANT_REVIEWS_PATH.with_suffix(".parquet.tmp")
    kept = total = 0
    with pq.ParquetWriter(tmp_path, REVIEW_SCHEMA) as writer:
        for chunk in iter_reviews(chunksize=chunksize):
            total += len(chunk)
            chunk = chunk[chunk["business_id"].isin(restaurant_ids)]
            kept += len(chunk)
            table = pa.Table.from_pandas(chunk[REVIEW_SCHEMA.names], schema=REVIEW_SCHEMA,
                                         preserve_index=False)
            writer.write_table(table)
            print(f"\r{total:,} reviews read, {kept:,} restaurant reviews kept", end="")
    # Rename only once complete, so a crash never leaves a half-written cache.
    tmp_path.replace(config.RESTAURANT_REVIEWS_PATH)
    print(f"\nSaved {config.RESTAURANT_REVIEWS_PATH}")


def load_restaurant_reviews(columns=None):
    """Restaurant reviews from the Parquet cache; pass columns to load only those."""
    if not config.RESTAURANT_REVIEWS_PATH.exists():
        raise FileNotFoundError(
            f"{config.RESTAURANT_REVIEWS_PATH} not found. Build it once with: python -m src.data"
        )
    return pd.read_parquet(config.RESTAURANT_REVIEWS_PATH, columns=columns)


if __name__ == "__main__":
    business = load_business()
    print(f"business: {business.shape}  (expected (150346, 14))")
    build_restaurant_reviews()
