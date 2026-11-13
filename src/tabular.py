"""Tabular features and models (role B).

Every feature is built only from data on or before each row's cutoff, except
the business-file fields listed below, which are 2022 snapshots.

Typical use:
    from src.tabular import load_inputs, build_features, fit_model, FEATURE_GROUPS

    restaurants, reviews = load_inputs()
    train, test = train_test_split(restaurants=restaurants, reviews=reviews[["business_id", "date"]])
    X_train = build_features(train, restaurants, reviews)
    X_test = build_features(test, restaurants, reviews)
    model = fit_model(X_train, train["closed"], groups=DEFAULT_GROUPS)
    p = predict(model, X_test)

Feature groups (pass a subset to fit_model to compare):
    volume      review counts overall and per window, growth, restaurant age
    rating      average stars overall and recent, trend, spread, share of 1-2 stars
    recency     days since last review, reviews in the last 3 / 6 months.
                Mostly flags restaurants that went quiet before the cutoff, so
                results are reported with and without it.
    location    state, latitude, longitude, active restaurants in the same postal code
    chain       how many restaurants share this name (first reviewed before the cutoff)
    categories  top cuisine / type categories (rarely change, so low leak risk)
    attributes  price range, takeout, delivery, ... 2022 snapshot: for closed
                restaurants it is frozen at closing, for open ones it includes
                later changes (e.g. delivery after 2020). Sensitivity check only.

`hours` is never used: it is missing for 22.6% of closed restaurants but only
9.6% of open ones, so it leaks the label.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.data import load_restaurant_reviews, load_restaurants
from src.labels import _end_of_day

REVIEW_COLUMNS = ["business_id", "date", "stars", "useful"]
N_CATEGORIES = 40
# Categories too broad to tell restaurants apart.
SKIP_CATEGORIES = {"Restaurants", "Food"}

ATTRIBUTE_FLAGS = ["RestaurantsTakeOut", "RestaurantsDelivery", "RestaurantsReservations",
                   "OutdoorSeating", "HasTV", "GoodForKids", "RestaurantsGoodForGroups",
                   "RestaurantsTableService", "Caters", "BusinessAcceptsCreditCards"]

FEATURE_GROUPS = {
    "volume": ["review_count", "n_12m", "n_prev_12m", "growth_12m", "age_days", "useful_per_review"],
    "rating": ["avg_stars", "avg_stars_12m", "stars_trend", "std_stars", "share_low_12m"],
    "recency": ["days_since_last", "n_3m", "n_6m"],
    "location": ["state", "latitude", "longitude", "postal_density"],
    "chain": ["chain_size"],
    "categories": [],    # filled in by build_features (cat_* columns)
    "attributes": ["price_range", "alcohol", "wifi_free", "n_attributes"]
                  + [f"attr_{a}" for a in ATTRIBUTE_FLAGS],
}
CATEGORICAL = ["state"]

# Headline model: everything that is leak-free. Recency is reported both ways.
DEFAULT_GROUPS = ["volume", "rating", "recency", "location", "chain", "categories"]
NO_RECENCY_GROUPS = [g for g in DEFAULT_GROUPS if g != "recency"]


# ---- Data --------------------------------------------------------------------

def load_inputs():
    """Restaurants (business file) and the review columns the features need."""
    return load_restaurants(), load_restaurant_reviews(columns=REVIEW_COLUMNS)


def _top_categories(restaurants, n=N_CATEGORIES):
    cats = restaurants["categories"].fillna("").str.split(r",\s*").explode()
    cats = cats[~cats.isin(SKIP_CATEGORIES) & cats.ne("")]
    return list(cats.value_counts().head(n).index)


def _norm_name(names):
    return names.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


# ---- Features ----------------------------------------------------------------

def review_features(cutoff, reviews):
    """Volume, rating and recency features per business, from reviews on or before `cutoff`."""
    c_end = _end_of_day(pd.Timestamp(cutoff))
    before = reviews[reviews["date"].lt(c_end)]

    def since(months):
        return before["date"].ge(c_end - pd.DateOffset(months=months))

    g = before.groupby("business_id")
    f = pd.DataFrame({
        "review_count": g.size(),
        "avg_stars": g["stars"].mean(),
        "std_stars": g["stars"].std(),
        "useful_per_review": g["useful"].mean(),
        "first_review": g["date"].min(),
        "last_review": g["date"].max(),
    })
    last12 = before[since(12)]
    g12 = last12.groupby("business_id")
    f["n_12m"] = g12.size()
    f["avg_stars_12m"] = g12["stars"].mean()
    f["share_low_12m"] = last12["stars"].le(2).groupby(last12["business_id"]).mean()
    f["n_3m"] = before[since(3)].groupby("business_id").size()
    f["n_6m"] = before[since(6)].groupby("business_id").size()
    prev = before[since(24) & ~since(12)]
    f["n_prev_12m"] = prev.groupby("business_id").size()

    for col in ["n_12m", "n_3m", "n_6m", "n_prev_12m"]:
        f[col] = f[col].fillna(0)
    f["growth_12m"] = np.log1p(f["n_12m"]) - np.log1p(f["n_prev_12m"])
    f["stars_trend"] = f["avg_stars_12m"] - f["avg_stars"]
    f["days_since_last"] = (c_end - f["last_review"]).dt.days
    f["age_days"] = (c_end - f["first_review"]).dt.days
    return f.drop(columns=["first_review", "last_review"])


def business_features(restaurants, categories, cutoff, reviews):
    """Location, chain, category and attribute features (one row per restaurant)."""
    r = restaurants.set_index("business_id")
    f = pd.DataFrame(index=r.index)
    f["state"] = r["state"]
    f["latitude"] = r["latitude"]
    f["longitude"] = r["longitude"]

    # Chain size: restaurants with the same name already reviewed by the cutoff.
    c_end = _end_of_day(pd.Timestamp(cutoff))
    existed = reviews.loc[reviews["date"].lt(c_end), "business_id"].unique()
    names = _norm_name(r["name"])
    f["chain_size"] = names.map(names[names.index.isin(existed)].value_counts()).fillna(0)

    cat_lists = r["categories"].fillna("").str.split(r",\s*")
    for c in categories:
        f["cat_" + _slug(c)] = cat_lists.apply(lambda cs, c=c: c in cs).astype(int)

    attrs = r["attributes"].apply(lambda a: a if isinstance(a, dict) else {})
    a = pd.json_normalize(attrs.tolist()).set_index(r.index)
    f["n_attributes"] = attrs.apply(len)
    f["price_range"] = pd.to_numeric(a.get("RestaurantsPriceRange2"), errors="coerce")
    alcohol = a.get("Alcohol", pd.Series(index=r.index, dtype=object)).astype(str)
    f["alcohol"] = np.where(alcohol.str.contains("none"), 0,
                            np.where(alcohol.str.contains("beer|full"), 1, np.nan))
    wifi = a.get("WiFi", pd.Series(index=r.index, dtype=object)).astype(str)
    f["wifi_free"] = np.where(wifi.str.contains("free"), 1,
                              np.where(wifi.str.contains("no|paid"), 0, np.nan))
    for col in ATTRIBUTE_FLAGS:
        s = a.get(col, pd.Series(index=r.index, dtype=object)).astype(str)
        f["attr_" + col] = np.where(s.eq("True"), 1, np.where(s.eq("False"), 0, np.nan))
    return f


def _slug(category):
    return "".join(ch if ch.isalnum() else "_" for ch in category.lower()).strip("_")


def build_features(labels, restaurants, reviews, categories=None):
    """Feature table aligned row for row with `labels` (one cutoff per call or several)."""
    if categories is None:
        categories = _top_categories(restaurants)
    parts = []
    for cutoff, group in labels.groupby("cutoff", sort=False):
        rf = review_features(cutoff, reviews)
        bf = business_features(restaurants[restaurants["business_id"].isin(group["business_id"])],
                               categories, cutoff, reviews)
        part = group[["business_id"]].join(rf, on="business_id").join(bf, on="business_id")
        # Active restaurants in the same postal code at this cutoff (competition).
        postal = group["business_id"].map(restaurants.set_index("business_id")["postal_code"])
        part["postal_density"] = postal.map(postal.value_counts()).to_numpy()
        part.index = group.index
        parts.append(part)
    X = pd.concat(parts).loc[labels.index].drop(columns="business_id")
    X["state"] = X["state"].astype("category")
    return X


def columns_for(X, groups):
    cols = []
    for g in groups:
        cols += [c for c in X.columns if c.startswith("cat_")] if g == "categories" else FEATURE_GROUPS[g]
    return cols


# ---- Models --------------------------------------------------------------------

HGB_PARAMS = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=15, min_samples_leaf=100,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                  n_iter_no_change=30, random_state=config.RANDOM_SEED)


def fit_model(X, y, groups=DEFAULT_GROUPS, kind="hgb", **params):
    """Fit a gradient-boosted tree model ("hgb") or logistic regression ("logit")."""
    cols = columns_for(X, groups)
    Xg = X[cols]
    if kind == "hgb":
        model = HistGradientBoostingClassifier(categorical_features="from_dtype",
                                               **{**HGB_PARAMS, **params})
    elif kind == "logit":
        cat = [c for c in cols if c in CATEGORICAL]
        num = [c for c in cols if c not in CATEGORICAL]
        prep = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  StandardScaler()), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat),
        ])
        model = make_pipeline(prep, LogisticRegression(max_iter=2000, C=params.get("C", 1.0)))
    else:
        raise ValueError(f"kind must be 'hgb' or 'logit', got {kind!r}")
    model.fit(Xg, y)
    model.feature_columns_ = cols
    return model


def predict(model, X):
    """Predicted probability of closure."""
    return model.predict_proba(X[model.feature_columns_])[:, 1]
