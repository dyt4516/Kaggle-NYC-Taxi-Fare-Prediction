import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

pd.set_option('display.max_columns', None)

# 1) 数据加载与清洗
df = pd.read_csv("data/train.csv", nrows=100000)
df = df[(df['fare_amount'] >= 0) & (df['fare_amount'] <= 200)]
df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
df = df[
    (df['pickup_longitude'] > -75) & (df['pickup_longitude'] < -72) &
    (df['pickup_latitude']  >  40) & (df['pickup_latitude']  <  42) &
    (df['dropoff_longitude'] > -75) & (df['dropoff_longitude'] < -72) &
    (df['dropoff_latitude']  >  40) & (df['dropoff_latitude']  <  42)
]

# 2) 时间 & 距离（核心特征）
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['year']    = df['pickup_datetime'].dt.year
df['month']   = df['pickup_datetime'].dt.month
df['day']     = df['pickup_datetime'].dt.day
df['hour']    = df['pickup_datetime'].dt.hour
df['weekday'] = df['pickup_datetime'].dt.weekday  # 0=周一,6=周日

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df['distance_km'] = haversine_distance(
    df['pickup_latitude'],  df['pickup_longitude'],
    df['dropoff_latitude'], df['dropoff_longitude']
)

# 3) 训练/验证
features = [
    'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude',
    'passenger_count', 'year', 'month', 'day', 'hour', 'weekday', 'distance_km'
]
X, y = df[features], df['fare_amount']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4) Linear Regression
lr_model = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_val)
rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))
print(f" Linear Regression RMSE：{rmse_lr:.2f}")

# 5) Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)
rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
print(f" Random Forest Regressor RMSE：{rmse_rf:.2f}")

# 6) LightGBM
lgbm_model = LGBMRegressor(
    n_estimators=500, learning_rate=0.05,
    num_leaves=31, max_depth=-1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1
).fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse"
)
y_pred_lgb = lgbm_model.predict(X_val)
rmse_lgb = np.sqrt(mean_squared_error(y_val, y_pred_lgb))
print(f" LightGBM RMSE：{rmse_lgb:.2f}")
