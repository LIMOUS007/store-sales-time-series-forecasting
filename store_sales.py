import pandas as pd
import numpy as np
import sklearn
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
pd.set_option('display.max_columns', None)
df = pd.read_csv(r"C:\coding\MACHINE\store-sales-time-series-forecasting\train.csv")
store = pd.read_csv(r"C:\coding\MACHINE\store-sales-time-series-forecasting\stores.csv")
holi = pd.read_csv(r"C:\coding\MACHINE\store-sales-time-series-forecasting\holidays_events.csv")
oil = pd.read_csv(r"C:\coding\MACHINE\store-sales-time-series-forecasting\oil.csv")
trans = pd.read_csv(r"C:\coding\MACHINE\store-sales-time-series-forecasting\transactions.csv")
df = pd.merge(df, store, on = "store_nbr", how = "left")
holi_simple = holi[["date"]].copy()
holi_simple["is_holiday"] = 1
holi_simple = holi_simple.drop_duplicates()
df = pd.merge(df, holi_simple, on = "date", how = "left")
df["is_holiday"] = df["is_holiday"].fillna(0)
df = pd.merge(df, oil, on = "date", how = "left")
df = pd.merge(df, trans, on = ["date","store_nbr"], how = "left")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=["store_nbr", "family", "date"])
df["dayofweek"] = df["date"].dt.day_of_week
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
df["isweekend"] = (df["dayofweek"] >= 5).astype(int)
df["sales_lag_1"] = df.groupby(["store_nbr", "family"])["sales"].shift(1)
df["sales_lag_7"] = df.groupby(["store_nbr", "family"])["sales"].shift(7)
df["sales_lag_14"] = df.groupby(["store_nbr", "family"])["sales"].shift(14)
df["sales_lag_30"] = df.groupby(["store_nbr", "family"])["sales"].shift(30)
df["rolling_mean_7"] = df.groupby(["store_nbr","family"])["sales"].shift(1).rolling(7).mean()
df["rolling_mean_30"] = df.groupby(["store_nbr","family"])["sales"].shift(1).rolling(30).mean()
df["rolling_std_7"] = df.groupby(["store_nbr","family"])["sales"].shift(1).rolling(7).std()
df["promo_lag_1"] = df.groupby(["store_nbr", "family"])["onpromotion"].shift(1)
df["promo_sum_7"] = df.groupby(["store_nbr", "family"])["onpromotion"].shift(1).rolling(7).sum()
df["oil_lag_7"] = df["dcoilwtico"].shift(7)
df["oil_change"] = df["dcoilwtico"] - df["oil_lag_7"]
df["holiday_promo"] = df["is_holiday"] * df["onpromotion"]
df["sales_momentum"] = df["rolling_mean_7"] / (df["sales_lag_30"] + 1)
df["sales_ratio"] = df["sales_lag_1"] / (df["sales_lag_7"] + 1)
df["holiday_lag_1"] = df["is_holiday"].shift(1)
df["holiday_lead_1"] = df["is_holiday"].shift(-1)
df["promo_x_sales"] = df["promo_lag_1"] * df["sales_lag_1"]
df["weekend_x_promo"] = df["isweekend"] * df["onpromotion"]
df = df.dropna(subset = ["sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_30", "oil_lag_7", "oil_change"])
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[["family"]])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["family"]), index = df.index)
df = pd.concat([df, encoded_df], axis = 1).drop("family", axis = 1)
df["dcoilwtico"] = df['dcoilwtico'].ffill()
df["transactions"] = df["transactions"].fillna(0)
df = df.fillna(0)
train_df = df[df['date'] < '2017-01-01']
val_df   = df[df['date'] >= '2017-01-01']
cols = ["id", "date", "city", "state", "type"]
train_df = train_df.drop(cols, axis=1)
val_df = val_df.drop(cols, axis=1)
X_train = train_df.drop('sales', axis=1)
y_train = train_df['sales']
X_val = val_df.drop('sales', axis=1)
y_val = val_df['sales']
gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    random_state=42
)
gbr.fit(X_train, y_train)
pred_gbr = gbr.predict(X_val)
rmse_gbr = np.sqrt(mean_squared_error(y_val, pred_gbr))
print("Gradient Boosting RMSE:", rmse_gbr)
joblib.dump(gbr, "gbr_model.pkl")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_val)
rmse_rf = np.sqrt(mean_squared_error(y_val, pred_rf))
print("Random Forest RMSE:", rmse_rf)
joblib.dump(rf, "rf_model.pkl")
pred_ens = (
    0.3 * pred_rf +
    0.7 * pred_gbr
)
rmse_ens = np.sqrt(mean_squared_error(y_val, pred_ens))
print("Ensemble RMSE:", rmse_ens)