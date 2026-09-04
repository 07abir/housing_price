"""
data_prep.py
==============================================================
The "cleaning brain" of the project.

Our raw file (House_Price.csv) was scraped from a property website,
so it is MESSY:
    - Prices look like text:  "₹8.5 Cr", "₹45.0 L"
    - Sizes look like text:   "4200 sq.ft"
    - BHK looks like text:    "6 BHK"
    - Lots of junk columns we do not need.

A machine-learning model can only learn from NUMBERS, so this file's
job is to turn that mess into a clean table of numbers.

Both compare_models.py and train_model.py import the function
`load_and_clean_data()` from here, so the cleaning logic lives in
ONE place only (no copy-paste).

You can also run this file on its own to SEE the cleaning happen:
    python data_prep.py
==============================================================
"""

import os
import re
import pandas as pd

# Always work relative to THIS file's folder, so the script can find
# House_Price.csv no matter which directory you run it from.
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "House_Price.csv")


# --------------------------------------------------------------
# Small helper #1: turn a price string like "₹8.5 Cr" into Lakhs.
# --------------------------------------------------------------
# In India, money is written in Lakhs (1 Lakh = 100,000) and
# Crores (1 Crore = 100 Lakhs = 10,000,000). We pick "Lakhs" as our
# single unit so every price is comparable.
#
#   "₹8.5 Cr"  -> 8.5 * 100 = 850.0   (Lakhs)
#   "₹45.0 L"  -> 45.0 * 1   = 45.0   (Lakhs)
#   "₹50 K"    -> 50  * 0.01 = 0.5    (Lakhs)
def price_to_lakh(text):
    if not isinstance(text, str):
        return None

    # Find the number and the unit word, e.g. "8.5" and "Cr".
    match = re.search(r"([\d.]+)\s*([A-Za-z]+)", text)
    if not match:
        return None

    number = float(match.group(1))
    unit = match.group(2).lower()   # "cr", "l", "lac", "k", ...

    # How many Lakhs is 1 of each unit?
    if unit.startswith("arab"):      # 1 Arab = 100 Crore (very rare)
        multiplier = 10000
    elif unit.startswith("cr"):      # Crore
        multiplier = 100
    elif unit.startswith("l"):       # L / Lac / Lacs / Lakh
        multiplier = 1
    elif unit.startswith("k") or unit.startswith("thousand"):
        multiplier = 0.01
    else:
        return None                  # unknown unit -> treat as bad data

    return number * multiplier


# --------------------------------------------------------------
# Small helper #2: pull the first number out of any text.
# --------------------------------------------------------------
#   "6 BHK"       -> 6.0
#   "4200 sq.ft"  -> 4200.0
#   "1,050 sq.ft" -> 1050.0   (commas removed first)
def first_number(text):
    if not isinstance(text, str):
        return None
    text = text.replace(",", "")           # "1,050" -> "1050"
    match = re.search(r"([\d.]+)", text)
    return float(match.group(1)) if match else None


# --------------------------------------------------------------
# The main function everything else uses.
# --------------------------------------------------------------
def load_and_clean_data(verbose=False):
    """Read House_Price.csv and return a clean table with columns:
    location, area_type, total_sqft, bhk, price   (price is in Lakhs)."""

    # 1. Read the raw CSV. It has messy/duplicate headers, but the five
    #    columns we care about have clean names, so we grab them by name.
    raw = pd.read_csv(CSV_PATH)
    rows_before = len(raw)

    df = pd.DataFrame()
    df["price"]      = raw["Flat_Price"].apply(price_to_lakh)   # target (Lakhs)
    df["bhk"]        = raw["BHK"].apply(first_number)
    df["total_sqft"] = raw["Total_Sq.ft"].apply(first_number)
    df["location"]   = raw["Location"].astype(str).str.strip()
    df["area_type"]  = raw["Area_Type"].astype(str).str.strip()

    # 2. Throw away rows where a number failed to parse (missing values).
    df = df.dropna(subset=["price", "bhk", "total_sqft"])

    # 3. Remove obviously-broken rows (data-entry errors / typos).
    #    These simple rules keep the model from learning nonsense.
    df = df[df["price"] > 0]
    df = df[(df["total_sqft"] >= 100) & (df["total_sqft"] <= 30000)]
    df = df[(df["bhk"] >= 1) & (df["bhk"] <= 20)]

    #    A real home has room to breathe. If a listing claims fewer than
    #    250 sqft PER bedroom, it is almost certainly a mistake -> drop it.
    df = df[df["total_sqft"] / df["bhk"] >= 250]

    #    Focus on the "normal" market and throw out obvious errors:
    #      - price-per-sqft below ₹1,000 or above ₹25,000 is almost always
    #        a typo (e.g. ₹356/sqft) or a rare luxury outlier.
    #      - we keep homes from ₹10 Lakh to ₹3 Crore. A handful of ₹10+
    #        Crore mansions can't be predicted from just size & location,
    #        and they badly distort the error, so we leave them out.
    #    (This one step roughly HALVES the error and lifts R² a lot.)
    price_per_sqft = df["price"] * 100000 / df["total_sqft"]   # in rupees
    df = df[price_per_sqft.between(1000, 25000)]
    df = df[df["price"].between(10, 300)]                      # ₹10 L – ₹3 Cr

    # 4. Drop exact duplicate listings so the SAME home can't end up in
    #    both the training and the test set (that would fake a good score).
    df = df.drop_duplicates()

    # 5. bhk is a whole number of bedrooms.
    df["bhk"] = df["bhk"].astype(int)

    df = df.reset_index(drop=True)

    if verbose:
        print(f"Rows in raw file : {rows_before}")
        print(f"Rows after cleaning: {len(df)}")
        print("\nFirst 5 clean rows:")
        print(df.head())
        print("\nPrice column (Lakhs) summary:")
        print(df["price"].describe())

    return df


# This block only runs when you do:  python data_prep.py
# It lets you check the cleaning is working before training anything.
if __name__ == "__main__":
    load_and_clean_data(verbose=True)
