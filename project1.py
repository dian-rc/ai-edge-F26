import tarfile
from pathlib import Path

import pandas as pd

TAR_PATH = Path(__file__).parent / "data" / "yelp_dataset.tar"


def load_yelp(name, nrows=None):
    """Read one Yelp file (business, checkin, review, tip, user) straight from the tar.

    review (~7.0M rows, 5.3 GB) and user (~2.0M rows, 3.4 GB) are large; pass nrows
    to load a sample instead of the full file.
    """
    with tarfile.open(TAR_PATH) as tar:
        f = tar.extractfile(f"yelp_academic_dataset_{name}.json")
        return pd.read_json(f, lines=True, nrows=nrows)


if __name__ == "__main__":
    business = load_yelp("business")
    checkin = load_yelp("checkin")
    tip = load_yelp("tip")
    review = load_yelp("review", nrows=100_000)
    user = load_yelp("user", nrows=100_000)

    for name, df in [("business", business), ("checkin", checkin), ("tip", tip),
                     ("review", review), ("user", user)]:
        print(f"\n=== {name}: {df.shape} ===")
        print(df.dtypes)
        print(df.head(3))
