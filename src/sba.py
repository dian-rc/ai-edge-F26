"""SBA 7(a) loan figures for the impact estimate (task 10).

Turns the SBA's loan-level 7(a) data into the few numbers the impact estimate
needs: how big a restaurant loan is, how often restaurant loans are charged
off, and how much is lost when one is.

Data: "FOIA - 7(a) (FY2010-FY2019)" from https://data.sba.gov/en/dataset/7-a-504-foia
(download it in a browser into data/; the site blocks scripted downloads). It
holds borrower names and addresses, so only aggregates leave this module.

Typical use:
    from src.sba import load_sba_loans, loan_figures

    loans = load_sba_loans()
    figures = loan_figures(loans)          # restaurant loans, mature cohorts
    figures["median_loan"], figures["lender_loss_rate"]

Definitions:
- Restaurant loan: NAICS 7225xx, or the pre-2012 codes 722110, 722211, 722212,
  722213 (NAICS was revised in 2012).
- Disbursed loans only: cancelled loans (CANCLD) were never paid out.
- Mature cohorts: loans approved in FY2010-2014 have had 12+ years to be repaid
  or charged off. Later cohorts still have many open loans (status EXEMPT),
  so their charge-off rates are not final.
- charge_off_rate: charged-off loans / disbursed loans. Open loans count as not
  charged off, so this is a lower bound for cohorts that are still running.
- loss_rate: gross charge-off amount / approved amount, for charged-off loans.
- lender_loss_rate: the same, times the unguaranteed share. The SBA repays its
  guaranteed share, so the lender loses only the rest.
"""

import pandas as pd

from src import config

SBA_7A_PATH = config.DATA_DIR / "FOIA_7a_FY2010_FY2019_asof_260930.csv"

SBA_COLUMNS = ["AsOfDate", "GrossApproval", "SBAGuaranteedApproval", "ApprovalDate", "ApprovalFY",
               "FirstDisbursementDate", "NaicsCode", "FranchiseCode", "TermInMonths", "LoanStatus",
               "PaidInFullDate", "ChargeOffDate", "GrossChargeOffAmount"]
DATE_COLUMNS = ["AsOfDate", "ApprovalDate", "FirstDisbursementDate", "PaidInFullDate", "ChargeOffDate"]

# Calendar years for the annual charge-off rate: enough loans outstanding, and
# before the pandemic (as with the Yelp cutoff).
ANNUAL_RATE_YEARS = (2012, 2019)

RESTAURANT_NAICS_PREFIX = "7225"                                     # NAICS 2012 onwards
RESTAURANT_NAICS_PRE2012 = {"722110", "722211", "722212", "722213"}  # NAICS 2007
MATURE_FY = (2010, 2014)


def load_sba_loans(path=SBA_7A_PATH):
    """Disbursed 7(a) loans with an is_restaurant flag (cancelled loans dropped)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download 'FOIA - 7(a) (FY2010-FY2019)' from "
            "https://data.sba.gov/en/dataset/7-a-504-foia into data/")
    loans = pd.read_csv(path, usecols=SBA_COLUMNS, dtype={"NaicsCode": str, "FranchiseCode": str},
                        parse_dates=DATE_COLUMNS)
    loans = loans[loans["LoanStatus"].ne("CANCLD")].copy()
    naics = loans["NaicsCode"].fillna("")
    loans["is_restaurant"] = naics.str.startswith(RESTAURANT_NAICS_PREFIX) | naics.isin(RESTAURANT_NAICS_PRE2012)
    loans["is_franchise"] = loans["FranchiseCode"].notna()
    loans["charged_off"] = loans["LoanStatus"].eq("CHGOFF")
    loans["guaranteed_share"] = loans["SBAGuaranteedApproval"] / loans["GrossApproval"]
    return loans.reset_index(drop=True)


def loan_figures(loans, restaurants=True, fy=MATURE_FY):
    """Loan size, charge-off rate and loss per charged-off loan, as a dict.

    restaurants: True for restaurant loans, False for all other industries,
    None for every loan. fy: (first, last) approval fiscal years, or None for all.
    """
    df = loans
    if restaurants is not None:
        df = df[df["is_restaurant"] == restaurants]
    if fy is not None:
        df = df[df["ApprovalFY"].between(*fy)]
    co = df[df["charged_off"]]
    loss_rate = co["GrossChargeOffAmount"] / co["GrossApproval"]
    lender_loss_rate = loss_rate * (1 - co["guaranteed_share"])
    years_to_co = (co["ChargeOffDate"] - co["ApprovalDate"]).dt.days / 365.25
    return {
        "loans": len(df),
        "median_loan": df["GrossApproval"].median(),
        "mean_loan": df["GrossApproval"].mean(),
        "guaranteed_share": df["guaranteed_share"].mean(),
        "franchise_share": df["is_franchise"].mean(),
        "still_open_share": df["LoanStatus"].eq("EXEMPT").mean(),
        "charge_off_rate": df["charged_off"].mean(),
        "median_years_to_charge_off": years_to_co.median(),
        "loss_rate": loss_rate.mean(),
        "lender_loss_rate": lender_loss_rate.mean(),
        "median_charge_off": co["GrossChargeOffAmount"].median(),
    }


def figures_table(loans):
    """Restaurants vs other industries, for mature cohorts and for all years."""
    rows = {
        "Restaurants, FY2010-14": loan_figures(loans, True, MATURE_FY),
        "Other industries, FY2010-14": loan_figures(loans, False, MATURE_FY),
        "Restaurants, FY2010-19": loan_figures(loans, True, None),
        "Other industries, FY2010-19": loan_figures(loans, False, None),
    }
    return pd.DataFrame(rows)


def annual_charge_off_rate(loans, restaurants=True, years=ANNUAL_RATE_YEARS):
    """Charge-offs per loan-year outstanding, counted within calendar `years`.

    A loan is outstanding from its first disbursement (approval date if missing)
    until it is repaid, charged off, or the file's as-of date. Only the part of
    that span inside `years` counts, and only charge-offs dated inside it. This
    is comparable to the Yelp closure rate, which is also per year.
    """
    df = loans if restaurants is None else loans[loans["is_restaurant"] == restaurants]
    start = pd.Timestamp(f"{years[0]}-01-01")
    end = pd.Timestamp(f"{years[1] + 1}-01-01")
    opened = df["FirstDisbursementDate"].fillna(df["ApprovalDate"])
    closed = df["ChargeOffDate"].fillna(df["PaidInFullDate"]).fillna(df["AsOfDate"])
    span = (closed.clip(upper=end) - opened.clip(lower=start)).dt.days.clip(lower=0) / 365.25
    charge_offs = df["ChargeOffDate"].ge(start) & df["ChargeOffDate"].lt(end)
    return charge_offs.sum() / span.sum()


def charge_off_by_cohort(loans, restaurants=True):
    """Charge-off rate and still-open share per approval year."""
    df = loans[loans["is_restaurant"] == restaurants]
    return df.groupby("ApprovalFY").agg(
        loans=("charged_off", "size"),
        charge_off_rate=("charged_off", "mean"),
        still_open_share=("LoanStatus", lambda s: s.eq("EXEMPT").mean()),
    )


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    loans = load_sba_loans()
    table = figures_table(loans)
    money = ["median_loan", "mean_loan", "median_charge_off"]
    counts = ["loans"]
    shown = table.astype(object)
    for row in table.index:
        if row in money:
            shown.loc[row] = table.loc[row].map(lambda v: f"${v:,.0f}")
        elif row in counts:
            shown.loc[row] = table.loc[row].map(lambda v: f"{v:,.0f}")
        elif row == "median_years_to_charge_off":
            shown.loc[row] = table.loc[row].map(lambda v: f"{v:.1f}")
        else:
            shown.loc[row] = table.loc[row].map(lambda v: f"{v:.1%}")
    print(shown.to_string())
    print("\nRestaurant loans by approval year:")
    print(charge_off_by_cohort(loans).round(3).to_string())
