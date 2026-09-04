# 🏠 Kolkata House Price Estimator

A beginner-friendly machine-learning website. You type a home's details
(location, area type, size, bedrooms) and it predicts the price. It also
trains **two** models, compares them, and uses the better one.

## The tech
- **Python + pandas** — read and clean the data
- **scikit-learn** — the two ML models (Linear Regression & Random Forest)
- **Flask** — the web server (backend)
- **HTML + CSS** — the web page (frontend)

## The files (in the order they matter)
| File | Job |
|------|-----|
| `House_Price.csv` | The raw data (3,900+ Kolkata listings). |
| `data_prep.py` | Cleans the messy data into numbers. |
| `models.py` | Defines the two models (shared setup). |
| `compare_models.py` | ⭐ Trains both, prints a scoreboard, picks a winner. |
| `train_model.py` | Trains both, saves the winner + scores for the website. |
| `app.py` | The Flask website. |
| `templates/index.html` + `static/style.css` | The web page. |

## How to run it (3 steps)

**1. Install the packages** (only needed once):
```bash
pip install -r requirements.txt
```

**2. Train the model** (creates `house_price_model.pkl` + score files):
```bash
python train_model.py
```

**3. Start the website:**
```bash
python app.py
```
Then open the link it prints — usually <http://127.0.0.1:5000>.

## Want to just see the model comparison?
```bash
python compare_models.py
```
This prints MAE and R² for both models side-by-side and names the winner.

## Two scores explained
- **MAE (Mean Absolute Error)** — on average, how far off the guess is,
  in rupees. **Lower is better.**
- **R² (R-squared)** — how much of the price variation the model explains,
  from 0 to 1 (shown as a %). **Higher is better.**

Both are measured on a 20% "test" slice of homes the model never saw during
training (the classic **80/20 split**).
