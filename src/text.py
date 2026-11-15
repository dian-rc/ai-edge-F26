"""Review-text features and model (role C, task 9).

Each restaurant is represented by what customers wrote about it before the
cutoff: its most recent reviews are turned into sentence embeddings (one
384-number vector per review, from a pretrained sentence-transformer) and
averaged. A logistic regression on that average predicts closure.

Only reviews dated on or before each row's cutoff are used, the same rule as
src.features and src.tabular, so nothing from the prediction year leaks in.

Typical use:
    from src.text import load_text_reviews, build_text_features, fit_text_model, predict

    reviews = load_text_reviews()
    train, test = train_test_split(reviews=reviews[["business_id", "date"]])
    X_train = build_text_features(train, reviews)   # embeds and caches on first run
    X_test = build_text_features(test, reviews)
    model = fit_text_model(X_train, train["closed"])
    p = predict(model, X_test)

X has one row per label row (same index as `labels`), with columns emb_0 ...
emb_383, so it can be joined to src.tabular.build_features() output for the
combined model (task 11).

Embeddings are cached per review in models/ (never committed), so each review
is embedded once on each laptop: about 10 to 15 minutes on an Apple-silicon
GPU for the first run, seconds after that.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data import load_restaurant_reviews
from src.labels import _end_of_day

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Most recent reviews per restaurant that are embedded, and how many are
# averaged by default. Embedding MAX_REVIEWS lets notebooks try fewer for free.
MAX_REVIEWS = 50
N_REVIEWS = 20
REVIEW_COLUMNS = ["review_id", "business_id", "date", "text"]
CACHE_DIR = config.MODELS_DIR / "review_embeddings" / TEXT_MODEL.split("/")[-1]


# ---- Data ---------------------------------------------------------------------

def load_text_reviews():
    """Restaurant reviews with the columns text features need (includes the text)."""
    return load_restaurant_reviews(columns=REVIEW_COLUMNS)


def recent_reviews(labels, reviews, n_reviews=MAX_REVIEWS):
    """The last n_reviews reviews of each label row's restaurant, dated on or before its cutoff.

    Returns review_id, business_id, cutoff and date, newest last within each restaurant.
    """
    parts = []
    for cutoff, group in labels.groupby("cutoff", sort=False):
        before = reviews[reviews["business_id"].isin(group["business_id"])
                         & reviews["date"].lt(_end_of_day(pd.Timestamp(cutoff)))]
        last = (before.sort_values(["business_id", "date", "review_id"])
                .groupby("business_id").tail(n_reviews))
        parts.append(last[["review_id", "business_id", "date"]].assign(cutoff=cutoff))
    return pd.concat(parts, ignore_index=True)


# ---- Embeddings -----------------------------------------------------------------

def _load_cache():
    ids_path, emb_path = CACHE_DIR / "review_ids.npy", CACHE_DIR / "embeddings.npy"
    if not ids_path.exists():
        return np.array([], dtype="U22"), np.zeros((0, 384), dtype=np.float16)
    return np.load(ids_path), np.load(emb_path)


def _save_cache(ids, emb):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Write to temporary files and rename, so a crash never leaves a broken cache.
    for name, arr in [("review_ids", ids), ("embeddings", emb)]:
        tmp = CACHE_DIR / f"{name}.tmp.npy"
        np.save(tmp, arr)
        tmp.replace(CACHE_DIR / f"{name}.npy")


def embed_reviews(review_ids, reviews, batch_size=128, save_every=100_000):
    """Embeddings (float16, unit length) for review_ids, in that order.

    Reviews already in the cache are reused; the rest are embedded with
    TEXT_MODEL and added to the cache.
    """
    from sentence_transformers import SentenceTransformer  # slow import; only when needed
    import torch

    review_ids = pd.Index(review_ids)
    ids, emb = _load_cache()
    todo = review_ids.unique().difference(pd.Index(ids))
    if len(todo):
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Embedding {len(todo):,} new reviews with {TEXT_MODEL} on {device} "
              f"({len(ids):,} already cached)")
        model = SentenceTransformer(TEXT_MODEL, device=device)
        texts = reviews.set_index("review_id")["text"].reindex(todo)
        for start in range(0, len(todo), save_every):
            chunk = texts.iloc[start:start + save_every]
            new = model.encode(chunk.tolist(), batch_size=batch_size, normalize_embeddings=True,
                               show_progress_bar=True).astype(np.float16)
            ids = np.concatenate([ids, chunk.index.to_numpy(dtype="U22")])
            emb = np.concatenate([emb, new])
            _save_cache(ids, emb)
    pos = pd.Index(ids).get_indexer(review_ids)
    if (pos < 0).any():
        raise RuntimeError(f"{(pos < 0).sum():,} reviews missing from the embedding cache")
    return emb[pos]


# ---- Features -------------------------------------------------------------------

def build_text_features(labels, reviews, n_reviews=N_REVIEWS):
    """Mean embedding of each row's last n_reviews pre-cutoff reviews, aligned with `labels`.

    Embeds (and caches) the last MAX_REVIEWS reviews per row, so later calls
    with any n_reviews <= MAX_REVIEWS need no new embedding.
    """
    if n_reviews > MAX_REVIEWS:
        raise ValueError(f"n_reviews must be at most MAX_REVIEWS ({MAX_REVIEWS})")
    recent = recent_reviews(labels, reviews, MAX_REVIEWS)
    emb = embed_reviews(recent["review_id"], reviews).astype(np.float32)
    recent = recent.groupby(["business_id", "cutoff"]).tail(n_reviews)
    emb = emb[recent.index]
    cols = [f"emb_{i}" for i in range(emb.shape[1])]
    means = (pd.DataFrame(emb, columns=cols)
             .assign(business_id=recent["business_id"].to_numpy(),
                     cutoff=recent["cutoff"].to_numpy())
             .groupby(["business_id", "cutoff"])[cols].mean())
    keys = pd.MultiIndex.from_frame(labels[["business_id", "cutoff"]])
    X = means.reindex(keys)
    if X.isna().any().any():
        raise ValueError("some label rows have no reviews before their cutoff")
    X.index = labels.index
    return X


# ---- Model ----------------------------------------------------------------------

def fit_text_model(X, y, Cs=(0.001, 0.01, 0.1, 1.0)):
    """Logistic regression on the embedding features.

    The regularisation strength is chosen by 5-fold cross-validation on the
    training rows only, scored by average precision.
    """
    model = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=list(Cs), cv=5, scoring="average_precision", max_iter=3000,
                             random_state=config.RANDOM_SEED),
    )
    model.fit(X, y)
    model.feature_columns_ = list(X.columns)
    return model


def predict(model, X):
    """Predicted probability of closure."""
    return model.predict_proba(X[model.feature_columns_])[:, 1]
