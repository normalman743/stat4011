import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# 设置中文显示和绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

# 读取数据
df = pd.read_csv('/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/Crime_Data_from_2020_to_Present_20250929.csv')

print("=== 数据集基本信息 ===")
print(f"数据形状: {df.shape}")
print(f"数据时间范围: {df['DATE OCC'].min()} 到 {df['DATE OCC'].max()}")
print("\n=== 数据列名 ===")
print(df.columns.tolist())

# 创建分析函数
def analyze_single_row(row_index, row_data):
    """分析单行数据的详细统计信息"""
    print(f"\n{'='*60}")
    print(f"=== 第 {row_index} 行数据分析 ===")
    print(f"{'='*60}")

    # 基本信息
    print("\n【基本信息】")
    print(f"案件编号: {row_data['DR_NO']}")
    print(f"报告日期: {row_data['Date Rptd']}")
    print(f"犯罪发生日期: {row_data['DATE OCC']}")
    print(f"犯罪发生时间: {row_data['TIME OCC']:04d}")
    print(f"区域代码: {row_data['AREA']} ({row_data['AREA NAME']})")
    print(f"报告地区编号: {row_data['Rpt Dist No']}")
    print(f"犯罪分类: {'重罪' if row_data['Part 1-2'] == 1 else '轻罪'}")

    # 犯罪信息
    print("\n【犯罪信息】")
    print(f"犯罪代码: {row_data['Crm Cd']}")
    print(f"犯罪描述: {row_data['Crm Cd Desc']}")
    if pd.notna(row_data['Mocodes']) and row_data['Mocodes'] != '':
        print(f"犯罪方式代码: {row_data['Mocodes']}")

    # 受害人信息
    print("\n【受害人信息】")
    print(f"受害人年龄: {row_data['Vict Age']}")
    if pd.notna(row_data['Vict Sex']):
        print(f"受害人性别: {row_data['Vict Sex']}")
    if pd.notna(row_data['Vict Descent']):
        print(f"受害人种族: {row_data['Vict Descent']}")

    # 地点信息
    print("\n【地点信息】")
    if pd.notna(row_data['Premis Desc']):
        print(f"犯罪地点: {row_data['Premis Desc']}")
    print(f"位置: {row_data['LOCATION']}")
    if pd.notna(row_data['Cross Street']) and row_data['Cross Street'] != '':
        print(f"十字路口: {row_data['Cross Street']}")
    print(f"坐标: ({row_data['LAT']}, {row_data['LON']})")

    # 武器和状态
    print("\n【案件处理】")
    if pd.notna(row_data['Weapon Used Cd']) and row_data['Weapon Used Cd'] != '':
        print(f"武器代码: {row_data['Weapon Used Cd']}")
        if pd.notna(row_data['Weapon Desc']):
            print(f"武器描述: {row_data['Weapon Desc']}")
    print(f"状态代码: {row_data['Status']} ({row_data['Status Desc']})")

    # 时间特征分析
    print("\n【时间特征分析】")
    crime_time = int(row_data['TIME OCC'])
    crime_hour = crime_time // 100
    crime_minute = crime_time % 100

    # 时间段分类
    if 5 <= crime_hour < 12:
        time_period = "早晨"
    elif 12 <= crime_hour < 17:
        time_period = "下午"
    elif 17 <= crime_hour < 22:
        time_period = "傍晚"
    else:
        time_period = "深夜"

    print(f"犯罪时间: {crime_hour:02d}:{crime_minute:02d} ({time_period})")


    return row_data

# 分析前10行数据作为示例
print("\n" + "="*80)
print("=== 详细单行数据分析 (前10行) ===")
print("="*80)

for i in range(10):
    row_data = df.iloc[i]
    analyze_single_row(i+1, row_data)

# ============================================================================
# === 特征工程:创建分类组合字段 ===
# ============================================================================

print("\n" + "="*80)
print("=== 特征工程:创建分类组合 ===")
print("="*80)

# 1. 地理维度
df['UAC'] = df.apply(lambda x: int(str(x['AREA']) + str(x['Rpt Dist No'])), axis=1)  # 唯一地区编码

# 2. 时间维度
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
df['Year'] = df['DATE OCC'].dt.year
df['Month'] = df['DATE OCC'].dt.month
df['Quarter'] = df['DATE OCC'].dt.quarter
df['DayOfWeek'] = df['DATE OCC'].dt.dayofweek  # 0=Monday, 6=Sunday
df['DayName'] = df['DATE OCC'].dt.day_name()
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# 时间段分类
df['Hour'] = df['TIME OCC'] // 100
df['TimePeriod'] = pd.cut(df['Hour'], 
                          bins=[-1, 5, 11, 17, 21, 24],
                          labels=['深夜(0-5)', '早晨(6-11)', '下午(12-17)', '傍晚(18-21)', '夜晚(22-24)'])

# 3. 犯罪类型维度
df['CrimeCategory'] = df['Part 1-2'].map({1: '重罪', 2: '轻罪'})
df['Crime_Type_Code'] = df['CrimeCategory'] + '_' + df['Crm Cd'].astype(str)

# 4. 受害人维度
df['VictimProfile'] = (df['Vict Sex'].fillna('U') + '_' + 
                       df['Vict Descent'].fillna('U') + '_' + 
                       pd.cut(df['Vict Age'], bins=[0, 18, 35, 60, 100], 
                              labels=['少年', '青年', '中年', '老年']).astype(str))

# 5. 地理+犯罪类型组合
df['Area_Crime'] = df['AREA NAME'] + '_' + df['CrimeCategory']

# 6. 时间+犯罪类型组合
df['Time_Crime'] = df['TimePeriod'].astype(str) + '_' + df['CrimeCategory']
df['Month_Crime'] = df['Month'].astype(str) + '月_' + df['CrimeCategory']

print("✓ 特征工程完成!")
print(f"新增字段: Year, Month, Quarter, DayOfWeek, Hour, TimePeriod, CrimeCategory, Crime_Type_Code, VictimProfile, Area_Crime, Time_Crime, Month_Crime")

# ============================================================================
# === 统计分析:各维度分布 ===
# ============================================================================

def analyze_distribution(df, column, title, top_n=10):
    """分析某个字段的分布"""
    print(f"\n{'='*80}")
    print(f"=== {title} ===")
    print(f"{'='*80}")
    
    # 统计
    stats = df[column].value_counts().head(top_n)
    pct = df[column].value_counts(normalize=True).head(top_n) * 100
    
    result = pd.DataFrame({
        '数量': stats,
        '占比(%)': pct.round(2)
    })
    
    print(result)
    print(f"\n总类别数: {df[column].nunique()}")
    
    return result

# 1. 时间维度分析
print("\n" + "🕐"*40)
print("【时间维度分析】")
print("🕐"*40)

analyze_distribution(df, 'Year', '年度犯罪分布')
analyze_distribution(df, 'Month', '月度犯罪分布', top_n=12)
analyze_distribution(df, 'Quarter', '季度犯罪分布', top_n=4)
analyze_distribution(df, 'DayName', '星期分布', top_n=7)
analyze_distribution(df, 'TimePeriod', '时段犯罪分布', top_n=5)

# 2. 地理维度分析
print("\n" + "🗺️"*40)
print("【地理维度分析】")
print("🗺️"*40)

analyze_distribution(df, 'AREA NAME', '区域犯罪分布', top_n=15)
analyze_distribution(df, 'UAC', '唯一地区编码(UAC)犯罪分布', top_n=20)

# 3. 犯罪类型分析
print("\n" + "🚨"*40)
print("【犯罪类型分析】")
print("🚨"*40)

analyze_distribution(df, 'CrimeCategory', '重罪/轻罪分布', top_n=2)
analyze_distribution(df, 'Crm Cd Desc', '具体犯罪类型分布', top_n=20)

# 4. 组合维度分析
print("\n" + "🔗"*40)
print("【组合维度分析】")
print("🔗"*40)

analyze_distribution(df, 'Area_Crime', '区域+犯罪类别组合', top_n=20)
analyze_distribution(df, 'Time_Crime', '时段+犯罪类别组合', top_n=10)
analyze_distribution(df, 'Month_Crime', '月份+犯罪类别组合', top_n=20)

# 5. 交叉统计分析
print("\n" + "="*80)
print("=== 交叉统计分析 ===")
print("="*80)

# 犯罪类别 × 时段
print("\n【犯罪类别 × 时段】")
crosstab1 = pd.crosstab(df['CrimeCategory'], df['TimePeriod'], margins=True)
print(crosstab1)

# 犯罪类别 × 月份
print("\n【犯罪类别 × 月份】")
crosstab2 = pd.crosstab(df['CrimeCategory'], df['Month'], margins=True)
print(crosstab2)

# 区域 × 犯罪类别
print("\n【Top 10 区域 × 犯罪类别】")
top_areas = df['AREA NAME'].value_counts().head(10).index
crosstab3 = pd.crosstab(df[df['AREA NAME'].isin(top_areas)]['AREA NAME'], 
                        df[df['AREA NAME'].isin(top_areas)]['CrimeCategory'])
print(crosstab3)

# 6. 数值型字段的基本统计
print("\n" + "="*80)
print("=== 数值型字段统计摘要 ===")
print("="*80)

numeric_cols = ['Vict Age', 'Hour', 'AREA', 'Rpt Dist No', 'Crm Cd', 'UAC']
print(df[numeric_cols].describe())

# 7. 保存处理后的数据
output_path = '/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/crime_data_processed.csv'
df.to_csv(output_path, index=False)
print(f"\n✓ 处理后的数据已保存至: {output_path}")