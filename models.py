"""
models.py
==============================================================
This file defines our TWO competing models in ONE place, so that
compare_models.py and train_model.py always use identical setups.
(If each file built its own models, they could drift apart and the
"winner" the trainer saves might not match the comparison. Bad!)

Both models share the SAME preprocessing:
  - location & area_type are WORDS. A model needs numbers, so we use
    "One-Hot Encoding": each location becomes its own yes/no (1/0)
    column. (e.g. is_Ballygunge? is_Kasba? ...)
  - total_sqft & bhk are already numbers, so we pass them through.

The two models:
  1. Linear Regression  -> draws the best straight-line relationship.
                           Simple and fast, but can't bend to complex data.
  2. Random Forest      -> asks lots of "decision trees" and averages them.
                           Slower, but usually much more accurate here.
==============================================================
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Which columns are words (categorical) and which are already numbers?
CATEGORICAL = ["location", "area_type"]
NUMERIC = ["total_sqft", "bhk"]

# The features we feed in, and the value we predict.
FEATURES = CATEGORICAL + NUMERIC
TARGET = "price"


def build_preprocessor():
    """Turn our columns into pure numbers the models can use."""
    return ColumnTransformer(
        transformers=[
            # One-hot encode the word columns. handle_unknown="ignore"
            # means: if the website sends a location the model never saw
            # in training, don't crash — just treat it as "all zeros".
            ("words", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            # "passthrough" = keep these number columns exactly as-is.
            ("numbers", "passthrough", NUMERIC),
        ]
    )


def build_models():
    """Return the two models we want to compare, as a name -> model dict.
    Each model is a Pipeline = [ preprocessing  ->  the regressor ]."""
    return {
        "Linear Regression": Pipeline([
            ("prep", build_preprocessor()),
            ("model", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("prep", build_preprocessor()),
            # n_estimators=100 -> use 100 trees.
            # random_state=42  -> same result every run (reproducible).
            # n_jobs=-1        -> use all CPU cores to train faster.
            ("model", RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1)),
        ]),
    }
