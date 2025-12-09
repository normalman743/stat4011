import pandas as pd
import numpy as np
from datetime import datetime

# 读取数据
csvfile = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/Crime_Data_from_2020_to_Present_20250929.csv"
data = pd.read_csv(csvfile)

# 输出文件路径
report_txt = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/EDA_Report.txt"
stats_csv = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/EDA_Statistics.csv"

# 创建报告文件
with open(report_txt, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("EXPLORATORY DATA ANALYSIS (EDA) REPORT\n")
    f.write("Crime Data from 2020 to Present\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    
    # 1. 数据集概览
    f.write("1. DATASET OVERVIEW\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total Rows: {data.shape[0]}\n")
    f.write(f"Total Columns: {data.shape[1]}\n")
    f.write(f"Column Names: {', '.join(data.columns.tolist())}\n\n")
    
    # 2. 数据类型和缺失值
    f.write("2. DATA TYPES AND MISSING VALUES\n")
    f.write("-" * 80 + "\n")
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        missing_pct = (missing_count / len(data)) * 100
        f.write(f"  {col:20s} | Type: {str(data[col].dtype):12s} | Missing: {missing_count:6d} ({missing_pct:5.2f}%)\n")
    f.write("\n")
    
    # 3. 基本统计信息
    f.write("3. DESCRIPTIVE STATISTICS (Numerical Variables)\n")
    f.write("-" * 80 + "\n")
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        f.write(f"\n{col}:\n")
        f.write(f"  Count:  {data[col].count()}\n")
        f.write(f"  Mean:   {data[col].mean():.4f}\n")
        f.write(f"  Std:    {data[col].std():.4f}\n")
        f.write(f"  Min:    {data[col].min()}\n")
        f.write(f"  Q1:     {data[col].quantile(0.25)}\n")
        f.write(f"  Median: {data[col].median()}\n")
        f.write(f"  Q3:     {data[col].quantile(0.75)}\n")
        f.write(f"  Max:    {data[col].max()}\n")
    f.write("\n")
    
    # 4. 分类变量分析
    f.write("4. CATEGORICAL VARIABLES ANALYSIS\n")
    f.write("-" * 80 + "\n")
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        f.write(f"\n{col} (Unique: {data[col].nunique()}):\n")
        value_counts = data[col].value_counts().head(10)
        for val, count in value_counts.items():
            pct = (count / len(data)) * 100
            f.write(f"  {str(val):30s}: {count:6d} ({pct:5.2f}%)\n")
    f.write("\n")
    
    # 5. 异常值检测
    f.write("5. OUTLIERS DETECTION (Numerical Variables)\n")
    f.write("-" * 80 + "\n")
    for col in numerical_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        f.write(f"\n{col}:\n")
        f.write(f"  IQR Range: [{lower_bound:.2f}, {upper_bound:.2f}]\n")
        f.write(f"  Outliers Count: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)\n")
    f.write("\n")
    
    # 6. 数据质量检查
    f.write("6. DATA QUALITY CHECK\n")
    f.write("-" * 80 + "\n")
    f.write(f"Duplicate Rows: {data.duplicated().sum()}\n")
    f.write(f"Complete Cases (no missing values): {data.dropna().shape[0]} ({data.dropna().shape[0]/len(data)*100:.2f}%)\n")
    f.write(f"Rows with at least one missing value: {len(data) - data.dropna().shape[0]}\n\n")
    
    # 7. 关键发现
    f.write("7. KEY FINDINGS & INSIGHTS\n")
    f.write("-" * 80 + "\n")
    
    # 受害者年龄分析
    if 'Vict Age' in data.columns:
        vict_age = data['Vict Age'].dropna()
        f.write(f"\n• Victim Age:\n")
        f.write(f"  - Age range: {vict_age.min()} to {vict_age.max()} years\n")
        f.write(f"  - Average age: {vict_age.mean():.1f} years\n")
        f.write(f"  - Unusual ages detected: {len(vict_age[vict_age < 0])} negative values\n")
    
    # 犯罪区域分析
    if 'AREA NAME' in data.columns:
        top_areas = data['AREA NAME'].value_counts().head(5)
        f.write(f"\n• Top 5 Crime Areas:\n")
        for area, count in top_areas.items():
            f.write(f"  - {area}: {count} incidents\n")
    
    # 犯罪类型分析
    if 'Crm Cd Desc' in data.columns:
        top_crimes = data['Crm Cd Desc'].value_counts().head(5)
        f.write(f"\n• Top 5 Crime Types:\n")
        for crime, count in top_crimes.items():
            f.write(f"  - {crime}: {count} incidents\n")
    
    # 案件状态分析
    if 'Status Desc' in data.columns:
        status_dist = data['Status Desc'].value_counts()
        f.write(f"\n• Case Status Distribution:\n")
        for status, count in status_dist.items():
            pct = (count / len(data)) * 100
            f.write(f"  - {status}: {count} ({pct:.1f}%)\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 80 + "\n")

print(f"✓ 文本报告已生成: {report_txt}")

# 生成统计汇总 CSV
stats_summary = []

# 数值变量统计
for col in numerical_cols:
    stats_summary.append({
        'Variable': col,
        'Type': 'Numerical',
        'Count': data[col].count(),
        'Missing': data[col].isnull().sum(),
        'Mean': data[col].mean(),
        'Std': data[col].std(),
        'Min': data[col].min(),
        'Max': data[col].max(),
        'Q1': data[col].quantile(0.25),
        'Median': data[col].median(),
        'Q3': data[col].quantile(0.75)
    })

# 分类变量统计
for col in categorical_cols:
    stats_summary.append({
        'Variable': col,
        'Type': 'Categorical',
        'Count': data[col].count(),
        'Missing': data[col].isnull().sum(),
        'Unique Values': data[col].nunique(),
        'Most Common': data[col].value_counts().index[0] if len(data[col].value_counts()) > 0 else None,
        'Most Common Count': data[col].value_counts().values[0] if len(data[col].value_counts()) > 0 else 0
    })

stats_df = pd.DataFrame(stats_summary)
stats_df.to_csv(stats_csv, index=False, encoding='utf-8')
print(f"✓ 统计汇总已生成: {stats_csv}")

# 打印摘要到控制台
print("\n" + "=" * 80)
print("EDA SUMMARY")
print("=" * 80)
print(f"Total Records: {len(data)}")
print(f"Total Features: {len(data.columns)}")
print(f"Missing Data: {data.isnull().sum().sum()} cells ({data.isnull().sum().sum()/(len(data)*len(data.columns))*100:.2f}%)")
print(f"Duplicate Rows: {data.duplicated().sum()}")
print(f"\nTop Crime Area: {data['AREA NAME'].value_counts().index[0]}")
print(f"Top Crime Type: {data['Crm Cd Desc'].value_counts().index[0]}")
print("=" * 80)
