"""
app.py
==============================================================
STEP 2 of the website: the Flask backend (the server).

Run this AFTER train_model.py has created the files:
    python app.py

Then open the link it prints (usually http://127.0.0.1:5000).

How it works:
  - GET  "/"  -> show the empty form.
  - POST "/"  -> read the form, predict the price, show the result.
==============================================================
"""

import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request

# Run from THIS file's folder so it can always find the model/data files.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Load everything ONCE when the server starts (keeps the site fast).
model = joblib.load("house_price_model.pkl")

with open("locations.json") as f:
    locations = json.load(f)
with open("area_types.json") as f:
    area_types = json.load(f)
with open("metrics.json") as f:
    metrics = json.load(f)


def format_price(lakhs):
    """Turn a number of Lakhs into a friendly ₹ string.
    e.g. 850 -> '₹8.50 Cr',  45 -> '₹45.00 Lakh'."""
    if lakhs >= 100:
        return f"₹{lakhs / 100:.2f} Cr"
    return f"₹{lakhs:.2f} Lakh"


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None   # None = form not submitted yet

    if request.method == "POST":
        # 1. Read what the user typed/selected.
        location   = request.form["location"]
        area_type  = request.form["area_type"]
        total_sqft = float(request.form["total_sqft"])
        bhk        = int(request.form["bhk"])

        # 2. Build a one-row table with the SAME column names used in
        #    training. The model's pipeline handles all the encoding.
        input_row = pd.DataFrame(
            [[location, area_type, total_sqft, bhk]],
            columns=["location", "area_type", "total_sqft", "bhk"],
        )

        # 3. Predict (result is in Lakhs). Never show a negative price.
        price_lakhs = max(model.predict(input_row)[0], 0)
        prediction = format_price(price_lakhs)

    # Pull the winning model's scores out for easy display.
    winner = metrics["winner"]
    win_scores = metrics["scores"][winner]

    return render_template(
        "index.html",
        locations=locations,
        area_types=area_types,
        prediction=prediction,
        # Model-quality info shown on the page:
        winner=winner,
        mae_lakhs=win_scores["mae"],
        mae_pretty=format_price(win_scores["mae"]),
        r2=win_scores["r2"],
        r2_percent=round(win_scores["r2"] * 100, 1),
        n_train=metrics["n_train"],
        n_test=metrics["n_test"],
        n_total=metrics["n_total"],
        all_scores=metrics["scores"],   # both models, for the compare table
    )


if __name__ == "__main__":
    # debug=True shows helpful error pages while you learn.
    app.run(debug=True, use_reloader=False)
