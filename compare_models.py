"""
compare_models.py
==============================================================
⭐ THE COMPARISON PROGRAM ⭐

This is the program you asked for: it trains BOTH models on the same
data, scores them on the same unseen test data, and tells you which
one is better.

Run it with:
    python compare_models.py

It prints a side-by-side scoreboard and saves the numbers to
model_comparison.json (so the website can show them too).

Two scores are used:
  - MAE (Mean Absolute Error): on average, how many Lakhs is the
    prediction off by? LOWER is better.
  - R²  (R-squared): how much of the price variation the model
    explains, from 0 to 1. HIGHER is better (1.0 = perfect).
==============================================================
"""

import os
import sys
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from data_prep import load_and_clean_data
from models import build_models, FEATURES, TARGET

# Let the Windows terminal print ₹, ² and 🏆 without crashing.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))


def rupees(lakhs):
    """Turn a number of Lakhs into a readable ₹ string for humans."""
    if lakhs >= 100:
        return f"₹{lakhs / 100:.2f} Cr"
    return f"₹{lakhs:.2f} Lakh"


def main():
    # 1. Get the clean data.
    df = load_and_clean_data()
    X = df[FEATURES]
    y = df[TARGET]

    # 2. Split 80% for training, 20% for testing.
    #    random_state=42 makes the split the same every time we run.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)

    print("=" * 56)
    print("  HOUSE PRICE MODEL COMPARISON  (prices in Lakhs)")
    print("=" * 56)
    print(f"Total clean listings : {len(df)}")
    print(f"Training rows (80%)  : {len(X_train)}")
    print(f"Testing rows  (20%)  : {len(X_test)}")
    print("-" * 56)
    print(f"{'Model':<20}{'MAE (Lakhs)':>16}{'R²':>10}  ")
    print(f"{'-'*20}{'-'*16:>16}{'-'*10:>10}  ")

    # 3. Train + score each model, remembering the results.
    results = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)               # learn from training rows
        predictions = model.predict(X_test)       # guess the test rows
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results[name] = {"mae": round(mae, 3), "r2": round(r2, 4)}
        print(f"{name:<20}{mae:>16.2f}{r2:>10.3f}  ")

    print("-" * 56)

    # 4. Decide the winner. Higher R² = explains prices better.
    winner = max(results, key=lambda name: results[name]["r2"])
    win = results[winner]
    print(f"🏆 WINNER: {winner}")
    print(f"   It explains {win['r2'] * 100:.1f}% of price variation,")
    print(f"   and is off by about {rupees(win['mae'])} on average.")
    print("=" * 56)

    # 5. Save the comparison so the website (or you) can read it later.
    out = {
        "winner": winner,
        "n_total": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "results": results,
    }
    with open(os.path.join(HERE, "model_comparison.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved model_comparison.json")


if __name__ == "__main__":
    main()
