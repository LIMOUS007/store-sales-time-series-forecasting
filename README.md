# Store Sales Time-Series Forecasting

Predicts daily sales for each store and product family using a full time-series ML pipeline.

**Dataset:** Kaggle – Store Sales Time Series Forecasting
**Goal:** Predict future daily sales per store and product family
**Type:** Time-series regression — evaluated with RMSLE on Kaggle (local validation used RMSE)

---

## Approach

### 1. Data Integration & Preprocessing
Multiple datasets were merged to enrich signal:

- `train.csv` → main sales data
- `stores.csv` → store metadata
- `holidays_events.csv` → holiday signal
- `oil.csv` → external economic factor
- `transactions.csv` → customer activity

Key steps: converted `date` to datetime, sorted by `store_nbr`/`family`/`date`, created a holiday indicator (`is_holiday`), forward-filled oil prices, and filled missing transactions with 0.

### 2. Feature Engineering
- **Time:** `day`, `month`, `dayofweek`, `weekofyear`, `isweekend`
- **Lag:** `sales_lag_1/7/14/30`
- **Rolling:** `rolling_mean_7/30`, `rolling_std_7`
- **Promotion:** `promo_lag_1`, `promo_sum_7`
- **Oil:** `oil_lag_7`, `oil_change`
- **Interaction:** `holiday_promo`, `promo_x_sales`, `weekend_x_promo`
- **Behavioral:** `sales_ratio`, `sales_momentum`
- **Encoding:** one-hot encoding of `family`

### 3. Models used
- Random Forest Regressor
- Gradient Boosting Regressor

---

## Results

| Model | RMSE (Validation) |
|-------|-------------------|
| Random Forest | ~239.6 |
| Gradient Boosting | ~209.6 |
| Ensemble (RF + GB) | ~211.8 |

Gradient Boosting performed best on local validation.

### Kaggle submissions

| Submission File | Description | Score |
|-----------------|-------------|-------|
| `submission_gbr.csv` | Gradient Boosting only | 3.55390 |
| `submission_ensemble.csv` | Ensemble (GB + RF) | 3.54247 |

The ensemble slightly improved the leaderboard score despite a worse local RMSE.

---

## How to run

```bash
pip install -r requirements.txt

# 1. Train models -> gbr_model.pkl, rf_model.pkl
python store_sales.py

# 2. Generate submissions -> submission_gbr.csv, submission_ensemble.csv
python submission.py
```