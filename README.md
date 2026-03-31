Store Sales Forecasting – Time Series Regression Project

Dataset: Kaggle – Store Sales Time Series Forecasting
Goal: Predict daily sales for each store and product family

Problem Overview

This dataset contains historical sales data across multiple stores and product categories.
The task is to predict future daily sales, making this a time-series regression problem.

Evaluation metric: RMSLE (Root Mean Squared Log Error) on Kaggle
(Local validation used RMSE)

Approach Summary

A full time-series ML pipeline was built instead of using raw models.

1️ Data Integration & Preprocessing

Multiple datasets were merged to enrich signal:

train.csv → main sales data
stores.csv → store metadata
holidays_events.csv → holiday signal
oil.csv → external economic factor
transactions.csv → customer activity

Key steps:

Converted date to datetime
Sorted data by store_nbr, family, date
Created holiday indicator (is_holiday)
Forward-filled oil prices
Filled missing transactions with 0
2️ Feature Engineering (Core Strength)

Feature engineering was the main driver of performance.

Time Features
day, month, dayofweek, weekofyear
isweekend
Lag Features (Historical Memory)
sales_lag_1, sales_lag_7, sales_lag_14, sales_lag_30
Rolling Statistics (Trend)
rolling_mean_7, rolling_mean_30
rolling_std_7
Promotion Features
promo_lag_1
promo_sum_7
Oil Features
oil_lag_7
oil_change
Interaction Features
holiday_promo
promo_x_sales
weekend_x_promo
Behavioral Features
sales_ratio
sales_momentum
Encoding
One-hot encoding of family
3️ Models Used

Two regression models were trained:

Random Forest Regressor
Gradient Boosting Regressor
4️ Results
Model	RMSE (Validation)
Random Forest	~239.6
Gradient Boosting	~209.6
Ensemble (RF + GB)	~211.8

Gradient Boosting performed best.

Kaggle Submissions
Submission File	Description	Score
submission_gbr.csv	Gradient Boosting only	3.55390
submission_ensemble.csv	Ensemble (GB + RF)	3.54247

Ensemble slightly improved leaderboard score despite worse local RMSE.

How to Run
1. Install dependencies
pip install -r requirements.txt
2. Train models
python train.py

Outputs:

models/gbr_model.pkl
models/rf_model.pkl
3. Generate submissions
python submission.py

Outputs:

submission_gbr.csv
submission_ensemble.csv
