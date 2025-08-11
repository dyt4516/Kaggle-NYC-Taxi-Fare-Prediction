import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

# 1) 读取数据（仅前 100,000 行）
df = pd.read_csv("data/train.csv", nrows=100000)

# 2) 保留合理车费/乘客数/经纬度范围
df = df[(df['fare_amount'] >= 0)]
df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 7)]
df = df[
    (df['pickup_longitude'] > -75) & (df['pickup_longitude'] < -72) &
    (df['pickup_latitude']  >  40) & (df['pickup_latitude']  <  42) &
    (df['dropoff_longitude'] > -75) & (df['dropoff_longitude'] < -72) &
    (df['dropoff_latitude']  >  40) & (df['dropoff_latitude']  <  42)
]

print("清洗后剩余数据条数：", len(df))

# 3) 车费在 0~50 美元区间的分布
plt.figure(figsize=(10, 6))
sns.histplot(df[df['fare_amount'] < 50]['fare_amount'], bins=100, kde=True)
plt.title("NYC Taxi Fare Distribution ($0–50)")
plt.xlabel("Fare Amount ($)")
plt.ylabel("Number of Rides")
plt.grid(True)
plt.show()

# 4) 把字符串时间转为时间类型
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['year']    = df['pickup_datetime'].dt.year
df['month']   = df['pickup_datetime'].dt.month
df['day']     = df['pickup_datetime'].dt.day
df['hour']    = df['pickup_datetime'].dt.hour
df['weekday'] = df['pickup_datetime'].dt.weekday  # 0=周一, 6=周日
print(df[['pickup_datetime', 'year', 'month', 'day', 'hour', 'weekday']].head())

# 5) 地理特征：计算上下车的 Haversine 球面距离（公里）
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

df['distance_km'] = haversine_distance(
    df['pickup_latitude'],  df['pickup_longitude'],
    df['dropoff_latitude'], df['dropoff_longitude']
)
print(df[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','distance_km']].head())
