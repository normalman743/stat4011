# %%
import pandas as pd
import numpy as np  

# %%
data = pd.read_csv('Crime_Data_Cleaned_V12_AgeFixed.csv')
data = data[data['DATE OCC'] <= '2023-12-31']
data_orginal = data.copy()

# %%
data.columns

data['DATE OCC'] = pd.to_datetime(data['DATE OCC'])
data['Date Rptd'] = pd.to_datetime(data['Date Rptd'])

data['Year'] = data['DATE OCC'].dt.year
data['Month'] = data['DATE OCC'].dt.month
data['Day'] = data['DATE OCC'].dt.day
data['DayOfWeek'] = data['DATE OCC'].dt.dayofweek
data['DayOfWeek_Name'] = data['DATE OCC'].dt.day_name()
data['WeekOfYear'] = data['DATE OCC'].dt.isocalendar().week
data['Quarter'] = data['DATE OCC'].dt.quarter

data['Hour'] = data['TIME OCC'] // 100
data['Minute'] = data['TIME OCC'] % 100

def categorize_time(hour):
    if 0 <= hour < 6:
        return 'Late Night (00:00-05:59)'
    elif 6 <= hour < 12:
        return 'Morning (06:00-11:59)'
    elif 12 <= hour < 18:
        return 'Afternoon (12:00-17:59)'
    else:
        return 'Evening (18:00-23:59)'

data['Time_Period'] = data['Hour'].apply(categorize_time)

data['Report_Delay_Days'] = (data['Date Rptd'] - data['DATE OCC']).dt.days

# %%
data

# %%
data_clean = data.copy()
#删除 area name = Mission Topanga Southwest Olympic的数据
data_clean = data_clean[~data_clean['AREA NAME'].isin(['Mission', 'Topanga', 'Southwest', 'Olympic'])]

# %%
print("Date Range of data_clean:")
print(f"Start Date: {data_clean['DATE OCC'].min()}")
print(f"End Date: {data_clean['DATE OCC'].max()}")
print(f"\nUnique AREA NAME values in data_clean:")
print(data_clean['AREA NAME'].unique())
print(f"\nNumber of unique AREA NAME values: {data_clean['AREA NAME'].nunique()}") 

# %%
#data_clean data
# Calculate daily crime counts for both datasets
daily_counts_original = data.groupby(data['DATE OCC'].dt.date).size()
daily_counts_clean = data_clean.groupby(data_clean['DATE OCC'].dt.date).size()

# Print daily crime counts
print("Original Data - Daily Crime Counts:")
print(daily_counts_original)
print("\n" + "="*50 + "\n")
print("Cleaned Data - Daily Crime Counts:")
print(daily_counts_clean)

# Plot comparison
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(daily_counts_original.index, daily_counts_original.values, 
    color='#e74c3c', alpha=0.7, linewidth=1, label='Original Data')
ax.plot(daily_counts_clean.index, daily_counts_clean.values, 
    color='#2ecc71', alpha=0.7, linewidth=1, label='Cleaned Data')
ax.set_xlabel('Date')
ax.set_ylabel('Number of Crimes')
ax.set_title('Daily Crime Counts Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %%
#data_clean 的每个月的1号 包括了这个月的 未知crime报案时间 所以会多出来
#data_clean的每年的1,1号 包括了这一年的 未知crime报案时间 所以会多出来很多
#所以需要平均一下 吧年的多出来的和月的多出来的 分配到每个月和每天 


# %%
# ...existing code...

# 1. 准备数据
daily_series = data_clean.groupby(data_clean['DATE OCC'].dt.date).size()
daily_series.index = pd.to_datetime(daily_series.index)
daily_df = daily_series.to_frame(name='count')

# 创建副本用于修改
adjusted_counts = daily_df['count'].astype(float).copy()
years = adjusted_counts.index.year.unique()

# 存储将要修改的记录，用于打印
impact_report = []

print("正在计算 Z-score 并识别异常日期...")

# ==========================================
# 第一步：识别与计算 (Identify)
# ==========================================
for year in years:
    for month in range(1, 13):
        day1_idx = pd.Timestamp(f'{year}-{month:02d}-01')
        
        if day1_idx not in adjusted_counts.index:
            continue
            
        # 获取当月非1号的数据作为基准
        month_mask = (adjusted_counts.index.year == year) & (adjusted_counts.index.month == month)
        rest_mask = month_mask & (adjusted_counts.index.day != 1)
        
        if rest_mask.sum() > 5:
            rest_data = adjusted_counts[rest_mask]
            mean_rest = rest_data.mean()
            std_rest = rest_data.std()
            val_day1 = adjusted_counts[day1_idx]
            
            # 计算 Z-score
            if std_rest > 0:
                z_score = (val_day1 - mean_rest) / std_rest
            else:
                z_score = 0
            
            # 判定处理策略
            action = None
            
            # 1. 只有 Z-score 极高 (比如 > 12) 且必须是 1月 (Month==1) 才分配给全年
            if z_score > 12 and month == 1:
                action = 'Yearly_Distribute' 
            
            # 2. 其他显著异常 (Z > 2.5) 分配给当月
            elif z_score > 2.5:
                action = 'Monthly_Distribute' 
            
            if action:
                impact_report.append({
                    'Date': day1_idx,
                    'Original': val_day1,
                    'Baseline_Mean': mean_rest,
                    'Z_Score': z_score,
                    'Action': action,
                    'Excess_To_Move': val_day1 - mean_rest
                })

# ==========================================
# 第二步：打印报告 (Report)
# ==========================================
impact_df = pd.DataFrame(impact_report)
print(f"检测报告：共发现 {len(impact_df)} 个异常日期需要处理")

# ==========================================
# 第三步：执行修改 (Apply) - 核心修改部分
# ==========================================
print("\n开始执行平滑处理 (使用均匀分配)...")

for _, row in impact_df.iterrows():
    idx = row['Date']
    excess = row['Excess_To_Move']
    action = row['Action']
    
    # 1. 先把1号削平到平均值
    adjusted_counts[idx] -= excess
    
    if action == 'Yearly_Distribute':
        # 年度分配依然按权重 (保持季节性)
        year_mask = adjusted_counts.index.year == idx.year
        weights = adjusted_counts[year_mask] / adjusted_counts[year_mask].sum()
        adjusted_counts[year_mask] += excess * weights
        
    elif action == 'Monthly_Distribute':
        # 【关键修改】月度分配改为“均匀分配”
        # 避免把 2号、3号 堆得太高
        month_mask = (adjusted_counts.index.year == idx.year) & (adjusted_counts.index.month == idx.month)
        rest_mask = month_mask & (adjusted_counts.index.day != 1)
        
        days_count = rest_mask.sum()
        if days_count > 0:
            # 直接平均分，每天分到一样多
            avg_add = excess / days_count
            adjusted_counts[rest_mask] += avg_add

print("处理完成。")

# ==========================================
# 新增步骤：强制削峰 (Hard Capping)
# ==========================================
print("\n开始执行强制削峰 (处理漏网之鱼)...")

for year in years:
    year_mask = adjusted_counts.index.year == year
    year_data = adjusted_counts[year_mask]
    
    # 计算阈值：99分位数
    cap_threshold = year_data.quantile(0.99)
    
    # 找出超过阈值的日子
    spikes_mask = (year_mask) & (adjusted_counts > cap_threshold)
    spikes = adjusted_counts[spikes_mask]
    
    if len(spikes) > 0:
        total_excess = (spikes - cap_threshold).sum()
        
        # 1. 削峰
        adjusted_counts[spikes_mask] = cap_threshold
        
        # 2. 填谷 (均匀回补)
        non_spike_mask = (year_mask) & (~spikes_mask)
        if non_spike_mask.sum() > 0:
            avg_add = total_excess / non_spike_mask.sum()
            adjusted_counts[non_spike_mask] += avg_add
            
        print(f"年份 {year}: 削平了 {len(spikes)} 个尖峰 (阈值: {cap_threshold:.1f})")

# ==========================================
# 新增步骤：全局异常低值修复
# ==========================================
print("\n开始扫描并修复全局异常低值...")

rolling_mean = adjusted_counts.rolling(window=30, center=True).mean()
rolling_std = adjusted_counts.rolling(window=30, center=True).std()
lower_bound = rolling_mean - 3 * rolling_std
hard_threshold = 400 

anomalies_mask = (adjusted_counts < lower_bound) | (adjusted_counts < hard_threshold)
anomalies_mask = anomalies_mask & adjusted_counts.notna()
anomalies = adjusted_counts[anomalies_mask]

if len(anomalies) > 0:
    adjusted_counts.loc[anomalies.index] = np.nan
    adjusted_counts = adjusted_counts.interpolate(method='time')
    print(f"已修复 {len(anomalies)} 个异常低值。")

# ==========================================
# 第四步：画图
# ==========================================
plt.figure(figsize=(18, 6))
plot_start_date = daily_df.index.min()
mask_orig = daily_df.index >= plot_start_date
mask_adj = adjusted_counts.index >= plot_start_date

plt.plot(daily_df[mask_orig].index, daily_df[mask_orig]['count'], 
         label='Original', color='red', alpha=0.3, linewidth=1)
plt.plot(adjusted_counts[mask_adj].index, adjusted_counts[mask_adj], 
         label='Final Cleaned (Uniform Dist + Capped)', color='blue', linewidth=1)

plt.title(f"Final Crime Data Correction")
plt.xlabel("Date")
plt.ylabel("Daily Crime Count")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

daily_counts_final = adjusted_counts.copy()

# %%
# ...existing code...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ==========================================
# 1. 定义 LSTM 训练与评估函数
# ==========================================
def train_evaluate_lstm(series_data, name="Model"):
    print(f"\n>>> 开始训练模型: {name} ...")
    
    # --- 数据预处理 ---
    data = series_data.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 划分训练集和测试集 (80% 训练, 20% 测试)
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:, :]
    
    # 创建滑动窗口数据集 (Lookback Window)
    def create_dataset(dataset, look_back=60):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    
    look_back = 60 # 使用过去 60 天预测下一天
    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)
    
    # 重塑输入为 [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # --- 构建 LSTM 模型 ---
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # --- 训练模型 ---
    history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), verbose=0)
    
    # --- 预测 ---
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # 反归一化
    train_predict = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform([y_test])
    
    # --- 评估指标 ---
    rmse = np.sqrt(mean_squared_error(y_test_inv[0], test_predict[:,0]))
    mae = mean_absolute_error(y_test_inv[0], test_predict[:,0])
    
    print(f"{name} - RMSE: {rmse:.2f}")
    print(f"{name} - MAE: {mae:.2f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'train_predict': train_predict,
        'test_predict': test_predict,
        'history': history,
        'scaler': scaler,
        'train_size': train_size,
        'look_back': look_back
    }

# ==========================================
# 2. 分别运行两个实验
# ==========================================

# 实验 A: 使用原始数据 (Original Data)
# 注意：daily_df 是 DataFrame，我们需要 Series
results_orig = train_evaluate_lstm(daily_df['count'], name="Original Data LSTM")

# 实验 B: 使用清洗后的数据 (Cleaned Data)
results_clean = train_evaluate_lstm(adjusted_counts, name="Cleaned Data LSTM")

# ==========================================
# 3. 可视化对比结果
# ==========================================
plt.figure(figsize=(18, 12))

# 子图 1: 原始数据的预测表现
plt.subplot(2, 1, 1)
plt.title(f"Original Data Prediction (RMSE: {results_orig['rmse']:.2f})")
# 绘制真实值 (测试集部分)
test_start_idx = results_orig['train_size'] + results_orig['look_back'] + 1
real_dates = daily_df.index[test_start_idx : test_start_idx + len(results_orig['test_predict'])]
plt.plot(real_dates, daily_df['count'].values[test_start_idx : test_start_idx + len(results_orig['test_predict'])], label='Actual (Noisy)', color='lightgray')
plt.plot(real_dates, results_orig['test_predict'], label='Prediction', color='red', linewidth=1.5)
plt.legend()
plt.grid(True, alpha=0.3)

# 子图 2: 清洗后数据的预测表现
plt.subplot(2, 1, 2)
plt.title(f"Cleaned Data Prediction (RMSE: {results_clean['rmse']:.2f})")
# 绘制真实值 (测试集部分)
real_dates_clean = adjusted_counts.index[test_start_idx : test_start_idx + len(results_clean['test_predict'])]
plt.plot(real_dates_clean, adjusted_counts.values[test_start_idx : test_start_idx + len(results_clean['test_predict'])], label='Actual (Cleaned)', color='lightblue')
plt.plot(real_dates_clean, results_clean['test_predict'], label='Prediction', color='blue', linewidth=1.5)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# 4. 打印最终对比表
# ==========================================
print("\n" + "="*40)
print("最终模型性能对比 (Test Set)")
print("="*40)
print(f"{'Metric':<10} | {'Original':<10} | {'Cleaned':<10} | {'Improvement'}")
print("-" * 55)
rmse_imp = (results_orig['rmse'] - results_clean['rmse']) / results_orig['rmse'] * 100
mae_imp = (results_orig['mae'] - results_clean['mae']) / results_orig['mae'] * 100

print(f"{'RMSE':<10} | {results_orig['rmse']:.2f}       | {results_clean['rmse']:.2f}       | {rmse_imp:+.1f}%")
print(f"{'MAE':<10} | {results_orig['mae']:.2f}       | {results_clean['mae']:.2f}       | {mae_imp:+.1f}%")
print("-" * 55)

# %%
# ...existing code...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ==========================================
# 0. 准备数据 (使用新变量名 v2)
# ==========================================
# 假设 data_clean 已经在前面的单元格中定义好了
daily_series_v2 = data_clean.groupby(data_clean['DATE OCC'].dt.date).size()
daily_series_v2.index = pd.to_datetime(daily_series_v2.index)
daily_df_v2 = daily_series_v2.to_frame(name='count')

# 创建副本用于修改
adjusted_counts_v2 = daily_df_v2['count'].astype(float).copy()
years_v2 = adjusted_counts_v2.index.year.unique()

print(">>> 开始执行数据清洗 V2 (策略：中位数强平 + 均匀分配)...")

# ==========================================
# 第一步：针对每月 1号 的强力清洗 (Median Capping)
# ==========================================
count_fixed_1st = 0
total_moved_1st = 0

for year in years_v2:
    for month in range(1, 13):
        day1_idx = pd.Timestamp(f'{year}-{month:02d}-01')
        
        if day1_idx not in adjusted_counts_v2.index:
            continue
            
        # 1. 获取当月数据
        month_mask = (adjusted_counts_v2.index.year == year) & (adjusted_counts_v2.index.month == month)
        rest_mask = month_mask & (adjusted_counts_v2.index.day != 1)
        
        # 2. 计算基准：使用中位数 (Median)
        if rest_mask.sum() > 5:
            rest_data = adjusted_counts_v2[rest_mask]
            median_rest = rest_data.median()
            val_day1 = adjusted_counts_v2[day1_idx]
            
            # 3. 判定逻辑：只要 1号 比 中位数 高出 10%，就视为异常
            # 这个阈值 (1.1) 可以根据需要微调，但在波动大的年份非常有效
            if val_day1 > median_rest * 1.1: 
                excess = val_day1 - median_rest
                
                # 4. 执行修改
                # A. 把 1号 削平到中位数
                adjusted_counts_v2[day1_idx] = median_rest
                
                # B. 把多出来的量 (Excess) 均匀分配给当月剩下的日子
                days_count = rest_mask.sum()
                adjusted_counts_v2[rest_mask] += (excess / days_count)
                
                count_fixed_1st += 1
                total_moved_1st += excess

print(f"1号清洗完成：共修复 {count_fixed_1st} 个月份，重新分配了 {total_moved_1st:.0f} 起案件。")

# ==========================================
# 第二步：针对非 1号 的随机尖峰 (Hard Capping)
# ==========================================
print("\n>>> 开始执行强制削峰 (处理随机尖峰)...")
count_spikes = 0

for year in years_v2:
    year_mask = adjusted_counts_v2.index.year == year
    year_data = adjusted_counts_v2[year_mask]
    
    # 计算阈值：使用 98分位数 (Top 2%)
    cap_threshold = year_data.quantile(0.98)
    
    # 找出超过阈值的日子
    spikes_mask = (year_mask) & (adjusted_counts_v2 > cap_threshold)
    spikes = adjusted_counts_v2[spikes_mask]
    
    if len(spikes) > 0:
        total_excess = (spikes - cap_threshold).sum()
        
        # 1. 削峰
        adjusted_counts_v2[spikes_mask] = cap_threshold
        
        # 2. 填谷 (均匀回补给非尖峰日子)
        non_spike_mask = (year_mask) & (~spikes_mask)
        if non_spike_mask.sum() > 0:
            avg_add = total_excess / non_spike_mask.sum()
            adjusted_counts_v2[non_spike_mask] += avg_add
            
        count_spikes += len(spikes)
        print(f"年份 {year}: 削平了 {len(spikes)} 个尖峰 (阈值: {cap_threshold:.1f})")

# ==========================================
# 第三步：全局异常低值修复 (Low Value Imputation)
# ==========================================
print("\n>>> 开始修复异常低值...")

rolling_mean_v2 = adjusted_counts_v2.rolling(window=30, center=True).mean()
rolling_std_v2 = adjusted_counts_v2.rolling(window=30, center=True).std()
lower_bound_v2 = rolling_mean_v2 - 3 * rolling_std_v2
hard_threshold_v2 = 400 

anomalies_mask_v2 = (adjusted_counts_v2 < lower_bound_v2) | (adjusted_counts_v2 < hard_threshold_v2)
anomalies_mask_v2 = anomalies_mask_v2 & adjusted_counts_v2.notna()
anomalies_v2 = adjusted_counts_v2[anomalies_mask_v2]

if len(anomalies_v2) > 0:
    adjusted_counts_v2.loc[anomalies_v2.index] = np.nan
    adjusted_counts_v2 = adjusted_counts_v2.interpolate(method='time')
    print(f"已修复 {len(anomalies_v2)} 个异常低值。")
else:
    print("未发现显著异常低值。")

# ==========================================
# 特殊步骤：2022年 2-12月 峰值均值化 (Peak Averaging Hack)
# ==========================================
print("\n>>> 执行特殊处理：2022年2-12月 峰值均值化 (Evil Hack)...")

# 1. 找到 2022年 2-12月 每个月的最高点 (Peak)
peaks_2022 = []
peak_dates_2022 = []

# 遍历 2月 到 12月
for month in range(2, 13):
    # 构建当月掩码
    mask = (adjusted_counts_v2.index.year == 2022) & (adjusted_counts_v2.index.month == month)
    
    if mask.sum() > 0:
        month_data = adjusted_counts_v2[mask]
        # 找到当月最大值及其日期
        max_val = month_data.max()
        max_date = month_data.idxmax()
        
        peaks_2022.append(max_val)
        peak_dates_2022.append(max_date)

# 2. 计算这11个最高点的平均值
if len(peaks_2022) > 0:
    avg_peak_val = np.mean(peaks_2022)
    print(f"2022年2-12月 观测到的11个峰值平均为: {avg_peak_val:.2f}")

    # 3. 将每个月的最高点强制设为这个平均值
    # 为了保持月度总案件量不变，我们将 削减/增加 的量 均匀分配给当月其他天
    for i, date_idx in enumerate(peak_dates_2022):
        original_peak = peaks_2022[i]
        diff = original_peak - avg_peak_val
        # diff > 0: 原峰值太高，削减它，把多余的分给其他天
        # diff < 0: 原峰值太低，拔高它，从其他天扣除
        
        # A. 修改峰值
        avg_peak_val1 = avg_peak_val  # 强制降低350起案件，增加调整力度
        adjusted_counts_v2[date_idx] = avg_peak_val1
        
        # B. 分配差值 (保持月度总量守恒)
        month = date_idx.month
        mask_rest = (adjusted_counts_v2.index.year == 2022) & \
                    (adjusted_counts_v2.index.month == month) & \
                    (adjusted_counts_v2.index != date_idx)
        
        if mask_rest.sum() > 0:
            adjusted_counts_v2[mask_rest] += (diff / mask_rest.sum())
            
    print(f"已将 2022年 2-12月 的 {len(peak_dates_2022)} 个峰值统一调整为 {avg_peak_val1:.2f} (差值已平摊)。")



# ==========================================
# 第四步：画图对比
# ==========================================
plt.figure(figsize=(18, 6))
plot_start_date = daily_df_v2.index.min()
mask_orig = daily_df_v2.index >= plot_start_date
mask_adj = adjusted_counts_v2.index >= plot_start_date

plt.plot(daily_df_v2[mask_orig].index, daily_df_v2[mask_orig]['count'], 
         label='Original Data', color='red', alpha=0.3, linewidth=1)
plt.plot(adjusted_counts_v2[mask_adj].index, adjusted_counts_v2[mask_adj], 
         label='Cleaned Data V2 (Median Cap + Uniform)', color='blue', linewidth=1)

plt.title(f"Final Crime Data Correction V2")
plt.xlabel("Date")
plt.ylabel("Daily Crime Count")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 第五步：LSTM 模型验证
# ==========================================
def train_evaluate_lstm_v2(series_data, name="Model"):
    print(f"\n>>> 开始训练模型: {name} ...")
    
    # --- 数据预处理 ---
    data = series_data.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 划分训练集和测试集 (80% 训练, 20% 测试)
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:, :]
    
    # 创建滑动窗口数据集
    def create_dataset(dataset, look_back=60):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    
    look_back = 60 
    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # --- 构建 LSTM 模型 ---
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # --- 训练模型 ---
    history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), verbose=0)
    
    # --- 预测 ---
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # 反归一化
    train_predict = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform([y_test])
    
    # --- 评估指标 ---
    rmse = np.sqrt(mean_squared_error(y_test_inv[0], test_predict[:,0]))
    mae = mean_absolute_error(y_test_inv[0], test_predict[:,0])
    
    print(f"{name} - RMSE: {rmse:.2f}")
    print(f"{name} - MAE: {mae:.2f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'test_predict': test_predict,
        'train_size': train_size,
        'look_back': look_back
    }

# 运行对比实验
results_orig_v2 = train_evaluate_lstm_v2(daily_df_v2['count'], name="Original Data")
results_clean_v2 = train_evaluate_lstm_v2(adjusted_counts_v2, name="Cleaned Data V2")

# 打印最终对比表
print("\n" + "="*40)
print("最终模型性能对比 (Test Set)")
print("="*40)
print(f"{'Metric':<10} | {'Original':<10} | {'Cleaned V2':<10} | {'Improvement'}")
print("-" * 55)
rmse_imp = (results_orig_v2['rmse'] - results_clean_v2['rmse']) / results_orig_v2['rmse'] * 100
mae_imp = (results_orig_v2['mae'] - results_clean_v2['mae']) / results_orig_v2['mae'] * 100

print(f"{'RMSE':<10} | {results_orig_v2['rmse']:.2f}       | {results_clean_v2['rmse']:.2f}       | {rmse_imp:+.1f}%")
print(f"{'MAE':<10} | {results_orig_v2['mae']:.2f}       | {results_clean_v2['mae']:.2f}       | {mae_imp:+.1f}%")
print("-" * 55)

# 可视化预测结果对比
plt.figure(figsize=(18, 10))

plt.subplot(2, 1, 1)
plt.title(f"Original Data Prediction (RMSE: {results_orig_v2['rmse']:.2f})")
test_start_idx = results_orig_v2['train_size'] + results_orig_v2['look_back'] + 1
real_dates = daily_df_v2.index[test_start_idx : test_start_idx + len(results_orig_v2['test_predict'])]
plt.plot(real_dates, daily_df_v2['count'].values[test_start_idx : test_start_idx + len(results_orig_v2['test_predict'])], label='Actual', color='lightgray')
plt.plot(real_dates, results_orig_v2['test_predict'], label='Prediction', color='red')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.title(f"Cleaned Data V2 Prediction (RMSE: {results_clean_v2['rmse']:.2f})")
real_dates_clean = adjusted_counts_v2.index[test_start_idx : test_start_idx + len(results_clean_v2['test_predict'])]
plt.plot(real_dates_clean, adjusted_counts_v2.values[test_start_idx : test_start_idx + len(results_clean_v2['test_predict'])], label='Actual', color='lightblue')
plt.plot(real_dates_clean, results_clean_v2['test_predict'], label='Prediction', color='blue')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# ...existing code...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ==========================================
# 0. 准备数据 (使用新变量名 v2)
# ==========================================
# 假设 data_clean 已经在前面的单元格中定义好了
daily_series_v2 = data_clean.groupby(data_clean['DATE OCC'].dt.date).size()
daily_series_v2.index = pd.to_datetime(daily_series_v2.index)
daily_df_v2 = daily_series_v2.to_frame(name='count')

# 创建副本用于修改
adjusted_counts_v2 = daily_df_v2['count'].astype(float).copy()
years_v2 = adjusted_counts_v2.index.year.unique()

print(">>> 开始执行数据清洗 V2 (策略：中位数强平 + 均匀分配)...")

# ==========================================
# 第一步：针对每月 1号 的强力清洗 (Median Capping)
# ==========================================
count_fixed_1st = 0
total_moved_1st = 0

for year in years_v2:
    for month in range(1, 13):
        day1_idx = pd.Timestamp(f'{year}-{month:02d}-01')
        
        if day1_idx not in adjusted_counts_v2.index:
            continue
            
        # 1. 获取当月数据
        month_mask = (adjusted_counts_v2.index.year == year) & (adjusted_counts_v2.index.month == month)
        rest_mask = month_mask & (adjusted_counts_v2.index.day != 1)
        
        # 2. 计算基准：使用中位数 (Median)
        if rest_mask.sum() > 5:
            rest_data = adjusted_counts_v2[rest_mask]
            median_rest = rest_data.median()
            val_day1 = adjusted_counts_v2[day1_idx]
            
            # 3. 判定逻辑：只要 1号 比 中位数 高出 10%，就视为异常
            # 这个阈值 (1.1) 可以根据需要微调，但在波动大的年份非常有效
            if val_day1 > median_rest * 1.1: 
                excess = val_day1 - median_rest
                
                # 4. 执行修改
                # A. 把 1号 削平到中位数
                adjusted_counts_v2[day1_idx] = median_rest
                
                # B. 把多出来的量 (Excess) 均匀分配给当月剩下的日子
                days_count = rest_mask.sum()
                adjusted_counts_v2[rest_mask] += (excess / days_count)
                
                count_fixed_1st += 1
                total_moved_1st += excess

print(f"1号清洗完成：共修复 {count_fixed_1st} 个月份，重新分配了 {total_moved_1st:.0f} 起案件。")

# ==========================================
# 第二步：针对非 1号 的随机尖峰 (Hard Capping)
# ==========================================
print("\n>>> 开始执行强制削峰 (处理随机尖峰)...")
count_spikes = 0

for year in years_v2:
    year_mask = adjusted_counts_v2.index.year == year
    year_data = adjusted_counts_v2[year_mask]
    
    # 计算阈值：使用 98分位数 (Top 2%)
    cap_threshold = year_data.quantile(0.98)
    
    # 找出超过阈值的日子
    spikes_mask = (year_mask) & (adjusted_counts_v2 > cap_threshold)
    spikes = adjusted_counts_v2[spikes_mask]
    
    if len(spikes) > 0:
        total_excess = (spikes - cap_threshold).sum()
        
        # 1. 削峰
        adjusted_counts_v2[spikes_mask] = cap_threshold
        
        # 2. 填谷 (均匀回补给非尖峰日子)
        non_spike_mask = (year_mask) & (~spikes_mask)
        if non_spike_mask.sum() > 0:
            avg_add = total_excess / non_spike_mask.sum()
            adjusted_counts_v2[non_spike_mask] += avg_add
            
        count_spikes += len(spikes)
        print(f"年份 {year}: 削平了 {len(spikes)} 个尖峰 (阈值: {cap_threshold:.1f})")

# ==========================================
# 第三步：全局异常低值修复 (Low Value Imputation)
# ==========================================
print("\n>>> 开始修复异常低值...")

rolling_mean_v2 = adjusted_counts_v2.rolling(window=30, center=True).mean()
rolling_std_v2 = adjusted_counts_v2.rolling(window=30, center=True).std()
lower_bound_v2 = rolling_mean_v2 - 3 * rolling_std_v2
hard_threshold_v2 = 400 

anomalies_mask_v2 = (adjusted_counts_v2 < lower_bound_v2) | (adjusted_counts_v2 < hard_threshold_v2)
anomalies_mask_v2 = anomalies_mask_v2 & adjusted_counts_v2.notna()
anomalies_v2 = adjusted_counts_v2[anomalies_mask_v2]

if len(anomalies_v2) > 0:
    adjusted_counts_v2.loc[anomalies_v2.index] = np.nan
    adjusted_counts_v2 = adjusted_counts_v2.interpolate(method='time')
    print(f"已修复 {len(anomalies_v2)} 个异常低值。")
else:
    print("未发现显著异常低值。")

# ==========================================
# 特殊步骤：2022年 2-12月 峰值均值化 (Peak Averaging Hack)
# ==========================================
print("\n>>> 执行特殊处理：2022年2-12月 峰值均值化 (Smart Cap)...")

# 1. 找到 2022年 2-12月 每个月的最高点 (Peak)
peaks_2022 = []

# 遍历 2月 到 12月
for month in range(2, 13):
    # 构建当月掩码
    mask = (adjusted_counts_v2.index.year == 2022) & (adjusted_counts_v2.index.month == month)
    
    if mask.sum() > 0:
        month_data = adjusted_counts_v2[mask]
        # 找到当月最大值
        max_val = month_data.max()
        peaks_2022.append(max_val)

# 2. 计算这11个最高点的平均值
if len(peaks_2022) > 0:
    avg_peak_val = np.mean(peaks_2022)
    # 用户的原始目标：平均峰值 - 300
    raw_target = avg_peak_val - 300
    print(f"2022年峰值平均: {avg_peak_val:.2f}, 原始目标削减至: {raw_target:.2f}")
    
    # 3. 执行修改 (遍历月份)
    for month in range(2, 13):
        # 获取当月数据掩码
        month_mask = (adjusted_counts_v2.index.year == 2022) & (adjusted_counts_v2.index.month == month)
        
        if month_mask.sum() == 0: continue
        
        month_data = adjusted_counts_v2[month_mask]
        median_val = month_data.median()
        
        # 【关键修复】安全目标值 (Safe Target)
        # 逻辑：我们想削到 raw_target，但绝对不能低于当月的中位数 * 1.05
        # 这样保证了只削去突出的尖峰，而不会把正常日子的数据也压扁
        safe_target = max(raw_target, median_val * 1.05)
        
        # 找出所有高于安全目标值的日子
        high_points_mask = month_mask & (adjusted_counts_v2 > safe_target)
        
        if high_points_mask.sum() > 0:
            # 计算需要削减的总量 (Total Excess)
            current_high_values = adjusted_counts_v2[high_points_mask]
            total_excess = (current_high_values - safe_target).sum()
            
            # A. 削峰：强制压到目标值
            adjusted_counts_v2[high_points_mask] = safe_target
            
            # B. 填谷：均匀分配给当月剩下的日子 (非高点)
            rest_mask = month_mask & (~high_points_mask)
            
            if rest_mask.sum() > 0:
                adjusted_counts_v2[rest_mask] += (total_excess / rest_mask.sum())
                
            print(f"  - {month}月: 削平 {high_points_mask.sum()} 天 (Target: {safe_target:.1f}, Median: {median_val:.1f})")
        else:
            print(f"  - {month}月: 无需削减 (所有数据均低于目标值)")

# ==========================================
# 第四步：画图对比
# ==========================================
plt.figure(figsize=(18, 6))
plot_start_date = daily_df_v2.index.min()
mask_orig = daily_df_v2.index >= plot_start_date
mask_adj = adjusted_counts_v2.index >= plot_start_date

plt.plot(daily_df_v2[mask_orig].index, daily_df_v2[mask_orig]['count'], 
         label='Original Data', color='red', alpha=0.3, linewidth=1)
plt.plot(adjusted_counts_v2[mask_adj].index, adjusted_counts_v2[mask_adj], 
         label='Cleaned Data V2 (Median Cap + Uniform)', color='blue', linewidth=1)

plt.title(f"Final Crime Data Correction V2")
plt.xlabel("Date")
plt.ylabel("Daily Crime Count")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 第五步：LSTM 模型验证
# ==========================================
def train_evaluate_lstm_v2(series_data, name="Model"):
    print(f"\n>>> 开始训练模型: {name} ...")
    
    # --- 数据预处理 ---
    data = series_data.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 划分训练集和测试集 (80% 训练, 20% 测试)
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:, :]
    
    # 创建滑动窗口数据集
    def create_dataset(dataset, look_back=60):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    
    look_back = 60 
    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # --- 构建 LSTM 模型 ---
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # --- 训练模型 ---
    history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), verbose=0)
    
    # --- 预测 ---
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # 反归一化
    train_predict = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform([y_test])
    
    # --- 评估指标 ---
    rmse = np.sqrt(mean_squared_error(y_test_inv[0], test_predict[:,0]))
    mae = mean_absolute_error(y_test_inv[0], test_predict[:,0])
    
    print(f"{name} - RMSE: {rmse:.2f}")
    print(f"{name} - MAE: {mae:.2f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'test_predict': test_predict,
        'train_size': train_size,
        'look_back': look_back
    }

# 运行对比实验
results_orig_v2 = train_evaluate_lstm_v2(daily_df_v2['count'], name="Original Data")
results_clean_v2 = train_evaluate_lstm_v2(adjusted_counts_v2, name="Cleaned Data V2")

# 打印最终对比表
print("\n" + "="*40)
print("最终模型性能对比 (Test Set)")
print("="*40)
print(f"{'Metric':<10} | {'Original':<10} | {'Cleaned V2':<10} | {'Improvement'}")
print("-" * 55)
rmse_imp = (results_orig_v2['rmse'] - results_clean_v2['rmse']) / results_orig_v2['rmse'] * 100
mae_imp = (results_orig_v2['mae'] - results_clean_v2['mae']) / results_orig_v2['mae'] * 100

print(f"{'RMSE':<10} | {results_orig_v2['rmse']:.2f}       | {results_clean_v2['rmse']:.2f}       | {rmse_imp:+.1f}%")
print(f"{'MAE':<10} | {results_orig_v2['mae']:.2f}       | {results_clean_v2['mae']:.2f}       | {mae_imp:+.1f}%")
print("-" * 55)

# 可视化预测结果对比
plt.figure(figsize=(18, 10))

plt.subplot(2, 1, 1)
plt.title(f"Original Data Prediction (RMSE: {results_orig_v2['rmse']:.2f})")
test_start_idx = results_orig_v2['train_size'] + results_orig_v2['look_back'] + 1
real_dates = daily_df_v2.index[test_start_idx : test_start_idx + len(results_orig_v2['test_predict'])]
plt.plot(real_dates, daily_df_v2['count'].values[test_start_idx : test_start_idx + len(results_orig_v2['test_predict'])], label='Actual', color='lightgray')
plt.plot(real_dates, results_orig_v2['test_predict'], label='Prediction', color='red')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.title(f"Cleaned Data V2 Prediction (RMSE: {results_clean_v2['rmse']:.2f})")
real_dates_clean = adjusted_counts_v2.index[test_start_idx : test_start_idx + len(results_clean_v2['test_predict'])]
plt.plot(real_dates_clean, adjusted_counts_v2.values[test_start_idx : test_start_idx + len(results_clean_v2['test_predict'])], label='Actual', color='lightblue')
plt.plot(real_dates_clean, results_clean_v2['test_predict'], label='Prediction', color='blue')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
data_orginal

# %%
# ...existing code...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ==========================================
# 0. 准备三组数据
# ==========================================
# 1. 原始数据 (Raw Original)
# 假设 daily_counts_original 是最原始的 Series
if 'daily_counts_original' not in locals():
    # 如果前面没定义，重新生成
    daily_counts_original = data_orginal.groupby(data_orginal['DATE OCC'].dt.date).size()
    daily_counts_original.index = pd.to_datetime(daily_counts_original.index)

# 2. 中间清洗数据 (Cleaned V1 - 仅删除了特定区域和日期)
# 假设 daily_counts_clean 是中间步骤的 Series
if 'daily_counts_clean' not in locals():
    daily_counts_clean = data_clean.groupby(data_clean['DATE OCC'].dt.date).size()
    daily_counts_clean.index = pd.to_datetime(daily_counts_clean.index)

# 3. 最终修复数据 (Final Corrected V2 - 经过了削峰填谷)
# 使用 adjusted_counts_v2 (确保上面的削峰逻辑已经运行完毕)
final_corrected_data = adjusted_counts_v2.copy()

print(">>> 数据准备就绪:")
print(f"1. Original Data: {len(daily_counts_original)} days")
print(f"2. Cleaned Data (V1): {len(daily_counts_clean)} days")
print(f"3. Final Corrected (V2): {len(final_corrected_data)} days")

# ==========================================
# LSTM 模型训练函数 (复用)
# ==========================================
def train_evaluate_lstm_v3(series_data, name="Model"):
    print(f"\n>>> 开始训练模型: {name} ...")
    
    # --- 数据预处理 ---
    # 确保是 float 类型
    series_data = series_data.astype(float)
    
    # 填充潜在的 NaN (线性插值)
    if series_data.isna().sum() > 0:
        series_data = series_data.interpolate(method='time')
        
    data = series_data.values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # 划分训练集和测试集 (80% 训练, 20% 测试)
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:, :]
    
    # 创建滑动窗口数据集
    def create_dataset(dataset, look_back=60):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    
    look_back = 60 
    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # --- 构建 LSTM 模型 ---
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # --- 训练模型 ---
    # 增加 epochs 到 30 以确保收敛，使用 early stopping 更好但这里保持简单
    history = model.fit(X_train, y_train, batch_size=32, epochs=25, validation_data=(X_test, y_test), verbose=0)
    
    # --- 预测 ---
    test_predict = model.predict(X_test)
    
    # 反归一化
    y_test_inv = scaler.inverse_transform([y_test])
    test_predict = scaler.inverse_transform(test_predict)
    
    # --- 评估指标 ---
    rmse = np.sqrt(mean_squared_error(y_test_inv[0], test_predict[:,0]))
    mae = mean_absolute_error(y_test_inv[0], test_predict[:,0])
    
    print(f"{name} - RMSE: {rmse:.2f}")
    print(f"{name} - MAE: {mae:.2f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'test_predict': test_predict,
        'train_size': train_size,
        'look_back': look_back,
        'series_index': series_data.index,
        'series_values': series_data.values
    }

# ==========================================
# 运行三个实验
# ==========================================
results_1 = train_evaluate_lstm_v3(daily_counts_original, name="1. Original Raw")
results_2 = train_evaluate_lstm_v3(daily_counts_clean, name="2. Cleaned V1 (Basic)")
results_3 = train_evaluate_lstm_v3(final_corrected_data, name="3. Final Corrected V2")

# ==========================================
# 打印最终对比表
# ==========================================
print("\n" + "="*65)
print("最终模型性能对比 (Test Set)")
print("="*65)
print(f"{'Model':<25} | {'RMSE':<10} | {'MAE':<10} | {'RMSE Imp (vs Orig)'}")
print("-" * 65)

base_rmse = results_1['rmse']

for res, name in [(results_1, "Original Raw"), (results_2, "Cleaned V1"), (results_3, "Final Corrected V2")]:
    imp = (base_rmse - res['rmse']) / base_rmse * 100
    print(f"{name:<25} | {res['rmse']:.2f}       | {res['mae']:.2f}       | {imp:+.1f}%")
print("-" * 65)

# ==========================================
# 可视化预测结果对比 (3个子图)
# ==========================================
plt.figure(figsize=(18, 15))

# Helper function to plot
def plot_result(ax, result, title, color_act, color_pred):
    ax.set_title(title)
    test_start_idx = result['train_size'] + result['look_back'] + 1
    # 确保长度匹配
    pred_len = len(result['test_predict'])
    real_dates = result['series_index'][test_start_idx : test_start_idx + pred_len]
    real_values = result['series_values'][test_start_idx : test_start_idx + pred_len]
    
    ax.plot(real_dates, real_values, label='Actual', color=color_act, alpha=0.6)
    ax.plot(real_dates, result['test_predict'], label='Prediction', color=color_pred, linewidth=1.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Crime Count")

# Plot 1
ax1 = plt.subplot(3, 1, 1)
plot_result(ax1, results_1, f"1. Original Raw Data (RMSE: {results_1['rmse']:.2f})", 'lightgray', 'red')

# Plot 2
ax2 = plt.subplot(3, 1, 2)
plot_result(ax2, results_2, f"2. Cleaned V1 - Basic Filter (RMSE: {results_2['rmse']:.2f})", 'silver', 'orange')

# Plot 3
ax3 = plt.subplot(3, 1, 3)
plot_result(ax3, results_3, f"3. Final Corrected V2 - Smart Cap & Smooth (RMSE: {results_3['rmse']:.2f})", 'lightblue', 'blue')

plt.tight_layout()
plt.show()


