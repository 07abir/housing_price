"""
train_model.py
==============================================================
STEP 1 of the website: prepare everything the site needs.

It does 4 things:
  1. Cleans the data (via data_prep.py).
  2. Trains BOTH models and measures them (MAE + R²) on a 20% test set.
  3. Keeps the BETTER model and saves it to house_price_model.pkl.
  4. Saves the dropdown lists + the scores the website will display.

Run this ONCE before starting the website:
    python train_model.py

(compare_models.py is for exploring the head-to-head. This file is
what actually feeds the website — but it also compares, because it
has to know which model to keep.)
==============================================================
"""

import os
import sys
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from data_prep import load_and_clean_data
from models import build_models, FEATURES, TARGET

# Let the Windows terminal print the ² in "R²" without crashing.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


def save_json(name, data):
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(data, f, indent=2)


def main():
    # 1. Clean data and separate features (X) from the price (y).
    df = load_and_clean_data()
    X = df[FEATURES]
    y = df[TARGET]

    # 2. Split into 80% training / 20% testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)

    print(f"Cleaned listings: {len(df)}  "
          f"(train {len(X_train)} / test {len(X_test)})")

    # 3. Train + score both models, and keep the trained pipelines.
    trained = {}
    scores = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores[name] = {
            "mae": round(mean_absolute_error(y_test, predictions), 3),
            "r2": round(r2_score(y_test, predictions), 4),
        }
        trained[name] = model
        print(f"  {name:<20} MAE={scores[name]['mae']:>8} Lakhs   "
              f"R²={scores[name]['r2']}")

    # 4. Pick the winner (best R²) and save THAT model for the website.
    winner = max(scores, key=lambda name: scores[name]["r2"])
    joblib.dump(trained[winner], os.path.join(HERE, "house_price_model.pkl"))
    print(f"Saved winner '{winner}' -> house_price_model.pkl")

    # 5. Save the scores the website will show (both models + the split).
    save_json("metrics.json", {
        "winner": winner,
        "n_total": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "scores": scores,
    })

    # 6. Save the dropdown choices for the form (sorted, no repeats).
    save_json("locations.json", sorted(df["location"].unique().tolist()))
    save_json("area_types.json", sorted(df["area_type"].unique().tolist()))
    print("Saved metrics.json, locations.json, area_types.json")
    print("Done! Now run:  python app.py")


if __name__ == "__main__":
    main()
