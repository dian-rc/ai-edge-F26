"""Project-wide settings, defined once and imported everywhere.

Change a setting here, never in a notebook, so all three models are always
built and scored on exactly the same restaurants, labels and split.
"""

from pathlib import Path

import pandas as pd

# ---- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

BUSINESS_PATH = DATA_DIR / "yelp_academic_dataset_business.json"
REVIEW_PATH = DATA_DIR / "yelp_academic_dataset_review.json"

# Restaurant-only copy of the review file, built once by
# src.data.build_restaurant_reviews(). Lives in data/, so it is never committed.
RESTAURANT_REVIEWS_PATH = DATA_DIR / "restaurant_reviews.parquet"

# ---- Restaurant definition ------------------------------------------------
# A business is a restaurant if this is one of its comma-separated categories.
RESTAURANT_CATEGORY = "Restaurants"

# ---- Label settings --------------------------------------------------------
# Test cutoff: features use data up to and including this day; the label asks
# whether the restaurant closes in the HORIZON_MONTHS after it (calendar 2019).
CUTOFF = pd.Timestamp("2018-12-31")

# A restaurant counts as active at a cutoff if it has at least one review in
# this many months before the cutoff. Sensitivity check: 3, 6 and 12.
ACTIVE_WINDOW_MONTHS = 6

# Prediction horizon in months.
HORIZON_MONTHS = 12

# Training cutoff: one horizon earlier, so the model is trained on 2018 closures
# and tested on 2019 closures (a time-based split; no 2019 data in training).
# Calculated from HORIZON_MONTHS so changing the horizon can never leak test-year
# labels into training. With a 12-month horizon this is 2017-12-31.
TRAIN_CUTOFF = CUTOFF - pd.DateOffset(months=HORIZON_MONTHS)

# Flagged-closed restaurants whose last review is on or before the cutoff:
# "closed" labels them 1 (default); "exclude" drops them (sensitivity check).
PRE_CUTOFF_CLOSURES = "closed"

# ---- Reproducibility -----------------------------------------------------
RANDOM_SEED = 42
