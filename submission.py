import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv(r"C:\coding\MACHINE\store-sales-time-series-forecasting\train.csv")
test = pd.read_csv(r"C:\coding\MACHINE\store-sales-time-series-forecasting\test.csv")
store = pd.read_csv(r"C:\coding\MACHINE\store-sales-time-series-forecasting\stores.csv")
holi = pd.read_csv(r"C:\coding\MACHINE\store-sales-time-series-forecasting\holidays_events.csv")
oil = pd.read_csv(r"C:\coding\MACHINE\store-sales-time-series-forecasting\oil.csv")
trans = pd.read_csv(r"C:\coding\MACHINE\store-sales-time-series-forecasting\transactions.csv")
test_ids = test["id"]
all_data = pd.concat([train, test], axis=0)
all_data = pd.merge(all_data, store, on="store_nbr", how="left")
holi_simple = holi[["date"]].copy()
holi_simple["is_holiday"] = 1
holi_simple = holi_simple.drop_duplicates()
all_data = pd.merge(all_data, holi_simple, on="date", how="left")
all_data["is_holiday"] = all_data["is_holiday"].fillna(0)
all_data = pd.merge(all_data, oil, on="date", how="left")
all_data = pd.merge(all_data, trans, on=["date", "store_nbr"], how="left")
all_data['date'] = pd.to_datetime(all_data['date'])
all_data = all_data.sort_values(by=["store_nbr", "family", "date"])
all_data["dayofweek"] = all_data["date"].dt.day_of_week
all_data["month"] = all_data["date"].dt.month
all_data["day"] = all_data["date"].dt.day
all_data["weekofyear"] = all_data["date"].dt.isocalendar().week.astype(int)
all_data["isweekend"] = (all_data["dayofweek"] >= 5).astype(int)
all_data["sales_lag_1"] = all_data.groupby(["store_nbr","family"])["sales"].shift(1)
all_data["sales_lag_7"] = all_data.groupby(["store_nbr","family"])["sales"].shift(7)
all_data["sales_lag_14"] = all_data.groupby(["store_nbr","family"])["sales"].shift(14)
all_data["sales_lag_30"] = all_data.groupby(["store_nbr","family"])["sales"].shift(30)
all_data["rolling_mean_7"] = all_data.groupby(["store_nbr","family"])["sales"].shift(1).rolling(7).mean()
all_data["rolling_mean_30"] = all_data.groupby(["store_nbr","family"])["sales"].shift(1).rolling(30).mean()
all_data["rolling_std_7"] = all_data.groupby(["store_nbr","family"])["sales"].shift(1).rolling(7).std()
all_data["promo_lag_1"] = all_data.groupby(["store_nbr","family"])["onpromotion"].shift(1)
all_data["promo_sum_7"] = all_data.groupby(["store_nbr","family"])["onpromotion"].shift(1).rolling(7).sum()
all_data["oil_lag_7"] = all_data["dcoilwtico"].shift(7)
all_data["oil_change"] = all_data["dcoilwtico"] - all_data["oil_lag_7"]
all_data["holiday_promo"] = all_data["is_holiday"] * all_data["onpromotion"]
all_data["sales_momentum"] = all_data["rolling_mean_7"] / (all_data["sales_lag_30"] + 1)
all_data["sales_ratio"] = all_data["sales_lag_1"] / (all_data["sales_lag_7"] + 1)
all_data["holiday_lag_1"] = all_data["is_holiday"].shift(1)
all_data["holiday_lead_1"] = all_data["is_holiday"].shift(-1)
all_data["promo_x_sales"] = all_data["promo_lag_1"] * all_data["sales_lag_1"]
all_data["weekend_x_promo"] = all_data["isweekend"] * all_data["onpromotion"]
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(all_data[["family"]])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["family"]), index=all_data.index)
all_data = pd.concat([all_data, encoded_df], axis=1).drop("family", axis=1)
all_data["dcoilwtico"] = all_data["dcoilwtico"].ffill()
all_data["transactions"] = all_data["transactions"].fillna(0)
test_df = all_data.iloc[len(train):]
cols = ["id", "date", "city", "state", "type"]
test_df = test_df.drop(cols, axis=1)
test_df = test_df.fillna(0)
test_df = test_df.drop("sales", axis=1)
gbr = joblib.load("gbr_model.pkl")
rf = joblib.load("rf_model.pkl")
pred_gbr = gbr.predict(test_df)
pred_rf = rf.predict(test_df)
pred_ens = 0.7 * pred_gbr + 0.3 * pred_rf
submission_gbr = pd.DataFrame({
    "id": test_ids,
    "sales": pred_gbr
})

submission_ens = pd.DataFrame({
    "id": test_ids,
    "sales": pred_ens
})

submission_gbr.to_csv("submission_gbr.csv", index=False)
submission_ens.to_csv("submission_ensemble.csv", index=False)
print("Done: submissions created")