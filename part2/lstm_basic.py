"""
基础LSTM犯罪预测模型 - 改进版
处理数据异常值和1月1日录入问题
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据加载 ====================
print("=" * 80)
print("基础LSTM犯罪预测模型 - 数据清洗改进版")
print("=" * 80)

# 加载数据
file_path = '/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/cleaned_data/Crime_Data_2010_to_Present_Cleaned_merged_and_deduped_20250929_add_by_def_fill_some_v3.1.csv'
print(f"\n正在加载数据: {file_path}")
df = pd.read_csv(file_path)
print(f"原始数据: {len(df):,} 条记录")

# ==================== 数据清洗 ====================
print("\n" + "=" * 80)
print("数据清洗流程")
print("=" * 80)

# 转换日期
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')
df = df.dropna(subset=['DATE OCC'])

# 时间范围过滤: 2010-2023 (移除2024和2025不完整数据)
start_date = '2010-01-01'
end_date = '2023-12-31'
df = df[(df['DATE OCC'] >= start_date) & (df['DATE OCC'] <= end_date)]
print(f"筛选2010-2023年数据: {len(df):,} 条记录")

# 按天聚合
daily_crimes = df.groupby('DATE OCC').size().reset_index(name='crime_count')
daily_crimes = daily_crimes.set_index('DATE OCC')

# 确保连续时间序列
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
daily_crimes = daily_crimes.reindex(date_range, fill_value=0)

print(f"生成连续日期序列: {len(daily_crimes)} 天")

# ===== 异常值处理 =====
print("\n处理异常值:")

# 计算正常中位数(排除1月1日)
jan_1st_mask = daily_crimes.index.dayofyear == 1
normal_median = daily_crimes[~jan_1st_mask]['crime_count'].median()
print(f"  - 正常日期中位数: {normal_median:.2f}")

# 处理1月1日异常
jan_1st_count = daily_crimes[jan_1st_mask]['crime_count']
jan_1st_anomaly = (jan_1st_count > 3 * normal_median).sum()
daily_crimes.loc[jan_1st_mask, 'crime_count'] = daily_crimes[jan_1st_mask]['crime_count'].apply(
    lambda x: normal_median if x > 3 * normal_median else x
)
print(f"  - 修正1月1日异常: {jan_1st_anomaly} 个")

# 处理月末异常
month_end_mask = daily_crimes.index.is_month_end
month_end_count = daily_crimes[month_end_mask]['crime_count']
month_end_anomaly = (month_end_count > 3 * normal_median).sum()
daily_crimes.loc[month_end_mask, 'crime_count'] = daily_crimes[month_end_mask]['crime_count'].apply(
    lambda x: normal_median if x > 3 * normal_median else x
)
print(f"  - 修正月末异常: {month_end_anomaly} 个")

# 滚动窗口平滑异常尖峰
rolling_mean = daily_crimes['crime_count'].rolling(window=7, center=True).mean()
rolling_std = daily_crimes['crime_count'].rolling(window=7, center=True).std()
outlier_mask = np.abs(daily_crimes['crime_count'] - rolling_mean) > 3 * rolling_std
outlier_count = outlier_mask.sum()
daily_crimes.loc[outlier_mask, 'crime_count'] = rolling_mean[outlier_mask]
print(f"  - 平滑其他异常点: {outlier_count} 个")

# 填充NaN
daily_crimes = daily_crimes.fillna(method='bfill').fillna(method='ffill')

print("\n清洗后数据统计:")
print(f"  - 日期范围: {daily_crimes.index.min().date()} 至 {daily_crimes.index.max().date()}")
print(f"  - 总天数: {len(daily_crimes)} 天")
print(f"  - 总犯罪数: {daily_crimes['crime_count'].sum():,}")
print(f"  - 日均犯罪: {daily_crimes['crime_count'].mean():.2f}")
print(f"  - 标准差: {daily_crimes['crime_count'].std():.2f}")
print(f"  - 最小值: {daily_crimes['crime_count'].min()}")
print(f"  - 最大值: {daily_crimes['crime_count'].max()}")

# ==================== 特征工程 ====================
print("\n" + "=" * 80)
print("特征工程")
print("=" * 80)

# 重置索引以便添加特征
daily_crimes = daily_crimes.reset_index()
daily_crimes.columns = ['date', 'crime_count']

# 添加时间特征
daily_crimes['year'] = daily_crimes['date'].dt.year
daily_crimes['month'] = daily_crimes['date'].dt.month
daily_crimes['day'] = daily_crimes['date'].dt.day
daily_crimes['dayofweek'] = daily_crimes['date'].dt.dayofweek
daily_crimes['dayofyear'] = daily_crimes['date'].dt.dayofyear
daily_crimes['weekofyear'] = daily_crimes['date'].dt.isocalendar().week
daily_crimes['quarter'] = daily_crimes['date'].dt.quarter

# 周末标记
daily_crimes['is_weekend'] = (daily_crimes['dayofweek'] >= 5).astype(int)

# 滚动统计特征
daily_crimes['rolling_mean_7'] = daily_crimes['crime_count'].rolling(window=7, min_periods=1).mean()
daily_crimes['rolling_std_7'] = daily_crimes['crime_count'].rolling(window=7, min_periods=1).std()
daily_crimes['rolling_mean_30'] = daily_crimes['crime_count'].rolling(window=30, min_periods=1).mean()

# 填充NaN
daily_crimes = daily_crimes.fillna(method='bfill').fillna(method='ffill')

print(f"特征数量: {len(daily_crimes.columns) - 1} 个 (不含日期)")
print(f"特征列: {', '.join([col for col in daily_crimes.columns if col != 'date'])}")

# ==================== 数据准备 ====================
print("\n" + "=" * 80)
print("LSTM数据准备")
print("=" * 80)

lookback_days = 30  # 使用过去30天预测
forecast_days = 7   # 预测未来7天

# 选择特征列
feature_columns = [col for col in daily_crimes.columns if col != 'date']
data = daily_crimes[feature_columns].values

# 数据标准化
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 创建序列数据
X, y = [], []
for i in range(lookback_days, len(scaled_data) - forecast_days):
    X.append(scaled_data[i-lookback_days:i])
    y.append(scaled_data[i:i+forecast_days, 0])  # 只预测crime_count

X = np.array(X)
y = np.array(y)

# 划分训练集和测试集 (80-20)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"回溯天数: {lookback_days} 天")
print(f"预测天数: {forecast_days} 天")
print(f"训练集: {X_train.shape[0]} 个样本")
print(f"测试集: {X_test.shape[0]} 个样本")
print(f"输入形状: {X_train.shape[1:]} (时间步, 特征数)")
print(f"输出形状: {y_train.shape[1]} (预测天数)")

# ==================== 构建LSTM模型 ====================
print("\n" + "=" * 80)
print("构建LSTM模型")
print("=" * 80)

model = Sequential([
    # 第一层LSTM
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    
    # 第二层LSTM
    LSTM(80, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),
    
    # 第三层LSTM
    LSTM(50, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),
    
    # 全连接层
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    
    # 输出层
    Dense(forecast_days)
])

# 编译模型 - 使用Huber损失对异常值更鲁棒
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='huber',
    metrics=['mae', 'mse']
)

print("\n模型结构:")
model.summary()

# ==================== 训练模型 ====================
print("\n" + "=" * 80)
print("训练模型")
print("=" * 80)

# 回调函数
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

# 训练
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n训练完成!")
print(f"训练轮数: {len(history.history['loss'])}")

# ==================== 模型评估 ====================
print("\n" + "=" * 80)
print("模型评估")
print("=" * 80)

# 预测
y_pred = model.predict(X_test, verbose=0)

# 反标准化
# 创建临时数组用于反标准化
temp_pred = np.zeros((y_pred.shape[0], y_pred.shape[1], data.shape[1]))
temp_pred[:, :, 0] = y_pred
temp_pred = temp_pred.reshape(-1, data.shape[1])
y_pred_rescaled = scaler.inverse_transform(temp_pred)[:, 0].reshape(y_pred.shape)

temp_true = np.zeros((y_test.shape[0], y_test.shape[1], data.shape[1]))
temp_true[:, :, 0] = y_test
temp_true = temp_true.reshape(-1, data.shape[1])
y_true_rescaled = scaler.inverse_transform(temp_true)[:, 0].reshape(y_test.shape)

# 计算评估指标
mse = mean_squared_error(y_true_rescaled.flatten(), y_pred_rescaled.flatten())
mae = mean_absolute_error(y_true_rescaled.flatten(), y_pred_rescaled.flatten())
rmse = np.sqrt(mse)
r2 = r2_score(y_true_rescaled.flatten(), y_pred_rescaled.flatten())

# 计算MAPE (避免除以0)
mape = np.mean(np.abs((y_true_rescaled.flatten() - y_pred_rescaled.flatten()) / 
                      np.maximum(y_true_rescaled.flatten(), 1))) * 100

print("\n整体评估指标:")
print(f"  - MSE (均方误差):        {mse:.2f}")
print(f"  - RMSE (均方根误差):     {rmse:.2f}")
print(f"  - MAE (平均绝对误差):    {mae:.2f}")
print(f"  - MAPE (平均绝对百分比): {mape:.2f}%")
print(f"  - R² (决定系数):         {r2:.4f}")

# 按预测天数分别评估
print("\n按预测天数评估:")
for day in range(forecast_days):
    day_mse = mean_squared_error(y_true_rescaled[:, day], y_pred_rescaled[:, day])
    day_mae = mean_absolute_error(y_true_rescaled[:, day], y_pred_rescaled[:, day])
    day_rmse = np.sqrt(day_mse)
    print(f"  第 {day+1} 天: RMSE={day_rmse:.2f}, MAE={day_mae:.2f}")

# ==================== 样本预测展示 ====================
print("\n" + "=" * 80)
print("样本预测展示 (前10个测试样本)")
print("=" * 80)

for i in range(min(10, len(y_test))):
    print(f"\n样本 {i+1}:")
    print(f"  真实值: {y_true_rescaled[i].round(1)}")
    print(f"  预测值: {y_pred_rescaled[i].round(1)}")
    print(f"  误差:   {(y_true_rescaled[i] - y_pred_rescaled[i]).round(1)}")

# ==================== 训练历史 ====================
print("\n" + "=" * 80)
print("训练历史")
print("=" * 80)

final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_train_mae = history.history['mae'][-1]
final_val_mae = history.history['val_mae'][-1]

print(f"\n最终训练损失: {final_train_loss:.4f}")
print(f"最终验证损失: {final_val_loss:.4f}")
print(f"最终训练MAE:  {final_train_mae:.4f}")
print(f"最终验证MAE:  {final_val_mae:.4f}")

# ==================== 总结 ====================
print("\n" + "=" * 80)
print("模型总结")
print("=" * 80)
print(f"\n数据处理:")
print(f"  - 时间范围: 2010-2023 (14年)")
print(f"  - 异常值修正: {jan_1st_anomaly + month_end_anomaly + outlier_count} 个")
print(f"  - 特征数量: {len(feature_columns)}")
print(f"\n模型配置:")
print(f"  - 模型类型: 3层LSTM + BatchNorm + Dropout")
print(f"  - 回溯窗口: {lookback_days} 天")
print(f"  - 预测窗口: {forecast_days} 天")
print(f"  - 训练样本: {len(X_train)}")
print(f"  - 测试样本: {len(X_test)}")
print(f"\n性能指标:")
print(f"  - RMSE: {rmse:.2f}")
print(f"  - MAE:  {mae:.2f}")
print(f"  - R²:   {r2:.4f}")
print(f"  - MAPE: {mape:.2f}%")

print("\n" + "=" * 80)
print("完成!")
print("=" * 80)
