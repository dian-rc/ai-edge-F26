# Restaurant closure early warning for small-business lenders

Predicts which restaurants close within 12 months, so a lender can decide which restaurant
borrowers to review or monitor. Built on the Yelp Open Dataset for MIT 15.071 (The AI Edge),
Project 1, by Shanthini, Dian and Mariana.

## Problem and prediction task

Restaurants fail often, and lenders see their borrowers' finances only through infrequent,
backward-looking statements. We score each active restaurant's probability of closing in the
next 12 months from public signals: business attributes, review statistics, and review text.

- **Cutoff:** 2018-12-31. Features use only data on or before the cutoff; the label is closure
  during 2019 (before the pandemic). Settings live in `src/config.py`.
- **Active restaurant:** at least one review in the 6 months before the cutoff (3 and 12 tested).
- **Label:** Yelp gives an open/closed flag but no closing date, so a restaurant counts as closed
  if it is flagged closed and its last review falls in 2019. Flagged-closed restaurants whose last
  review is *before* the cutoff may already have closed; the `closed` label counts them as
  closures, the stricter `exclude` label drops them.
- **Headline label: `exclude`.** `closed` includes restaurants that had likely already closed
  before the cutoff, which inflates its scores; `closed` results are shown alongside for reference.
- **Split by time:** train on the 2017-12-31 snapshot (2018 closures, 28,807 restaurants), test on
  the 2018-12-31 snapshot (2019 closures, 30,080 restaurants, 9.4% closed; `exclude`: 29,041,
  6.2% closed).

## Models

| Model | Code | What it uses |
|---|---|---|
| Star rating rule | `src/baseline.py` | Lowest average star rating as of the cutoff = riskiest |
| Recency rule | `src/tabular.py` | Longest time since the last review = riskiest |
| Text | `src/text.py` | Last 20 pre-cutoff reviews embedded with `all-MiniLM-L6-v2`, averaged; logistic regression |
| Tabular | `src/tabular.py` | Review volume, rating and recency statistics, location, chain size, cuisine; gradient boosting |
| **Tabular + text** | `src/combined.py` | Tabular model plus the text model's out-of-fold risk score as one feature |

All models are scored by one function, `src/evaluate.py`. Business attributes and opening hours
are excluded from the headline models: they are 2022 snapshots that leak which restaurants survived.

## Results (2019 test set)

| Model | Avg precision, `exclude` | Avg precision, `closed` | Lift in riskiest 10%, `exclude` | Losses avoided per 1,000 loans, `exclude` |
|---|---|---|---|---|
| Random review | 0.064 | 0.096 | 1.0x | $42k |
| Star rating | 0.059 | 0.092 | 0.9x | $37k |
| Recency | 0.065 | 0.143 | 1.0x | $40k |
| Text | 0.126 | 0.211 | 2.6x | $110k |
| Tabular | 0.141 | 0.331 | 2.7x | $115k |
| **Tabular + text** | **0.164** | **0.366** | **3.1x** | **$128k** |

- **Star ratings do not predict closure**; the combined model does, and review text adds to the
  tabular model on both labels (paired bootstrap 95% interval for the gain in average precision:
  +0.015 to +0.031 on `exclude`, +0.027 to +0.043 on `closed`).
- **Calibrated:** in the riskiest tenth, predicted vs actual closure rates are 19.7% vs 18.9%
  (`exclude`) and 36.4% vs 36.1% (`closed`).
- **Dollar impact** (`src/impact.py`): reviewing the riskiest 10% of a 1,000-loan restaurant
  portfolio each year, using SBA 7(a) figures (1.26% annual charge-off rate, $175k median loan,
  19% lender loss after the guarantee; `src/sba.py`). This is an upper bound: it assumes defaults
  are spread like closures and that reviewing a loan prevents its loss. With `closed` the
  combined model avoids $161k (38% of expected losses).
- Charts and tables for the slides: `results/` (start with `model_comparison.png`).

## Reproduce the results

Tested with Python 3.13.15 on two laptops, reproducing the files in `results/`. **The text steps
need about 18 GB of RAM** (they load all restaurant review text) and use the GPU if available.
Times below are from an Apple-silicon Mac; slower laptops can take a few times longer.

1. **Environment:** create a virtual environment and install `requirements.txt`
   (`python -m venv .venv`, activate it, `pip install -r requirements.txt`).
2. **Data:** download the [Yelp Open Dataset](https://www.yelp.com/dataset), accept its terms, and
   put `yelp_academic_dataset_business.json` and `yelp_academic_dataset_review.json` in `data/`.
   The data is not in this repository: Yelp's terms do not allow redistribution.
3. Run from the project folder, in order:

| Command | Time | Output |
|---|---|---|
| `python -m src.data` | ~4 min | `data/restaurant_reviews.parquet`, restaurant reviews only; prints business shape `(150346, 14)` |
| `python -m src.labels` | ~1 min | Restaurants and closure rates per split (should match the numbers above) |
| `python -m src.baseline` | ~1 min | Star-rating rule results |
| `python -m src.impact` | ~2 min | Tabular model: calibration, dollar impact, active-window sensitivity, in `results/` |
| `python -m src.combined` | First run: ~15 min (Apple-silicon GPU) to 1-2 hours (laptop CPU); then 1-5 min | All models compared, bootstrap, impact and calibration of the combined model, in `results/` |

The first `src.combined` run embeds about 1.4M reviews and caches them in `models/`; that is
almost all of its run time. Results are deterministic: rerunning reproduces the files in
`results/` byte for byte (on another machine, up to floating-point noise in the last decimals).

Optional: `python -m src.sba` re-derives the loan assumptions hard-coded in `src/impact.py`. It
needs "FOIA - 7(a) (FY2010-FY2019)" from [data.sba.gov](https://data.sba.gov/en/dataset/7-a-504-foia),
downloaded in a browser into `data/`.

## Repository layout

```
src/        shared code: config.py (settings), data.py, labels.py, features.py, evaluate.py,
            baseline.py, tabular.py, text.py, combined.py, sba.py, impact.py
notebooks/  exploration, one per person: data exploration, tabular model, text model
results/    tables and charts produced by the commands above
data/       Yelp and SBA files (local only, git-ignored)
models/     review embedding cache (local only, git-ignored)
```

## Limitations

- **Closure dates are inferred** from the last review; "closed" can also mean relocation or new
  owners. Restaurants with no late-2019 reviews that closed in 2020 may be counted as 2019 closures.
- **One test year.** The bootstrap covers which restaurants were sampled, not whether results
  hold in other years.
- **Probabilities run slightly low** (5.4% predicted vs 6.2% actual on `exclude`) because 2018,
  the training year, had fewer closures than 2019.
- **Many false alarms:** about 80% of restaurants flagged in the riskiest 10% do not close, so the
  model suits monitoring and review, not automatic decline.
- **Text partly encodes restaurant type** (independent cuisines riskier, chains safer), which
  overlaps with the tabular chain and cuisine features.
- **Yelp coverage:** only restaurants listed and reviewed on Yelp; the closure flag is as of 2022.

## Use of AI

As the course encourages, we used generative AI throughout: Claude Code (Anthropic) helped write
and modify code, debug, explore the data, make charts, and draft documentation, including this
README. Its contributions are attributed in the history: commits it helped write carry a
"Co-Authored-By: Claude" line, and pull requests it helped draft say "Generated with Claude Code".

The team made and owns the decisions: the stakeholder and problem framing, the cutoff and label
design, the leakage rules, the model comparison, and the interpretation. We verified the results
rather than taking them on trust: the evaluation function reproduces the tabular model's
published scores, results were rerun from scratch on a second laptop with matching outputs, and
each pull request's description records how it was checked. Responsibility for the correctness
of everything submitted remains with us.
