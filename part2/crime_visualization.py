import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

print("正在读取数据...")

# 读取两个CSV文件
df1 = pd.read_csv('/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/cleaned_data/Crime_Data_Cleaned_Final-1.csv')
df2 = pd.read_csv('/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/cleaned_data/Crime_Data_Cleaned_Final-2.csv')

print(f"文件1数据量: {len(df1)}")
print(f"文件2数据量: {len(df2)}")

# 合并数据
data = pd.concat([df1, df2], ignore_index=True)
print(f"合并后总数据量: {len(data)}")
print(f"\n数据列名: {data.columns.tolist()}")
print(f"\n数据概览:")
print(data.head())
print(f"\n数据信息:")
print(data.info())

# 转换日期格式
data['DATE OCC'] = pd.to_datetime(data['DATE OCC'], errors='coerce')
data['Date Rptd'] = pd.to_datetime(data['Date Rptd'], errors='coerce')

# 提取年份和月份
data['Year'] = data['DATE OCC'].dt.year
data['Month'] = data['DATE OCC'].dt.month
data['Year_Month'] = data['DATE OCC'].dt.to_period('M')

# 创建输出目录
import os
output_dir = '/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/crime_visualization_output'
os.makedirs(output_dir, exist_ok=True)

print("\n开始生成可视化图表...")

# 1. 按犯罪类型统计 (Top 15)
plt.figure(figsize=(14, 8))
crime_counts = data['Crm Cd Desc'].value_counts().head(15)
crime_counts.plot(kind='barh', color='steelblue')
plt.title('Top 15 Crime Types', fontsize=16, fontweight='bold')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Crime Type', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/01_top15_crime_types.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: Top 15 犯罪类型")
plt.close()

# 2. 按地区统计 (AREA NAME)
plt.figure(figsize=(14, 8))
area_counts = data['AREA NAME'].value_counts()
area_counts.plot(kind='barh', color='coral')
plt.title('Crime Count by Area', fontsize=16, fontweight='bold')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Area', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_crime_by_area.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 按地区的犯罪统计")
plt.close()

# 3. 按年份统计
plt.figure(figsize=(12, 6))
yearly_counts = data['Year'].value_counts().sort_index()
yearly_counts.plot(kind='bar', color='teal')
plt.title('Crime Count by Year', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/03_crime_by_year.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 按年份的犯罪统计")
plt.close()

# 4. 按月份统计 (所有年份汇总)
plt.figure(figsize=(12, 6))
monthly_counts = data['Month'].value_counts().sort_index()
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_counts.plot(kind='bar', color='mediumpurple')
plt.title('Crime Count by Month (All Years Combined)', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(range(12), month_names, rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/04_crime_by_month.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 按月份的犯罪统计")
plt.close()

# 5. 按受害者性别统计
plt.figure(figsize=(10, 6))
sex_counts = data['Vict Sex'].value_counts()
colors_sex = ['skyblue', 'pink', 'gray']
plt.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%', 
        colors=colors_sex, startangle=90)
plt.title('Crime Distribution by Victim Sex', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/05_crime_by_victim_sex.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 按受害者性别的犯罪统计")
plt.close()

# 6. 按受害者种族统计
plt.figure(figsize=(12, 6))
descent_counts = data['Vict Descent'].value_counts().head(10)
descent_counts.plot(kind='bar', color='salmon')
plt.title('Top 10 Crime Count by Victim Descent', fontsize=16, fontweight='bold')
plt.xlabel('Descent Code', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/06_crime_by_victim_descent.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 按受害者种族的犯罪统计")
plt.close()

# 7. 按案件状态统计
plt.figure(figsize=(10, 6))
status_counts = data['Status Desc'].value_counts()
status_counts.plot(kind='bar', color='lightgreen')
plt.title('Crime Count by Status', fontsize=16, fontweight='bold')
plt.xlabel('Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/07_crime_by_status.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 按案件状态的犯罪统计")
plt.close()

# 8. 按场所类型统计 (Top 15)
plt.figure(figsize=(14, 8))
premis_counts = data['Premis Desc'].value_counts().head(15)
premis_counts.plot(kind='barh', color='gold')
plt.title('Top 15 Crime Count by Premise Type', fontsize=16, fontweight='bold')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Premise Type', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/08_crime_by_premise.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 按场所类型的犯罪统计")
plt.close()

# 9. 按武器类型统计 (排除无武器)
plt.figure(figsize=(14, 8))
weapon_data = data[data['Weapon Desc'].notna() & (data['Weapon Desc'] != 'NO WEAPON')]
weapon_counts = weapon_data['Weapon Desc'].value_counts().head(15)
weapon_counts.plot(kind='barh', color='crimson')
plt.title('Top 15 Weapon Types Used (Excluding "NO WEAPON")', fontsize=16, fontweight='bold')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Weapon Type', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/09_crime_by_weapon.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 按武器类型的犯罪统计")
plt.close()

# 10. 时间序列趋势图 - 整体趋势
fig, axes = plt.subplots(3, 1, figsize=(18, 14))

# 10a. 月度趋势
time_series = data.groupby('Year_Month').size()
ax1 = axes[0]
time_series.plot(kind='line', linewidth=2.5, color='navy', marker='o', markersize=4, ax=ax1)
ax1.fill_between(range(len(time_series)), time_series.values, alpha=0.3, color='navy')
ax1.set_title('Monthly Crime Trend', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Time (Year-Month)', fontsize=12)
ax1.set_ylabel('Crime Count', fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(axis='x', rotation=45)

# 添加移动平均线
rolling_mean = time_series.rolling(window=12, center=True).mean()
ax1.plot(range(len(rolling_mean)), rolling_mean.values, 
         color='red', linewidth=2, linestyle='--', label='12-Month Moving Avg', alpha=0.7)
ax1.legend(fontsize=10)

# 10b. 年度趋势
yearly_series = data.groupby('Year').size()
ax2 = axes[1]
yearly_series.plot(kind='line', linewidth=3, color='darkgreen', marker='D', 
                   markersize=8, ax=ax2)
ax2.fill_between(range(len(yearly_series)), yearly_series.values, alpha=0.3, color='darkgreen')
ax2.set_title('Yearly Crime Trend', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Crime Count', fontsize=12)
ax2.set_xticks(range(len(yearly_series)))
ax2.set_xticklabels(yearly_series.index, rotation=45)
ax2.grid(True, alpha=0.3, linestyle='--')

# 添加数值标签
for i, v in enumerate(yearly_series.values):
    ax2.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)

# 10c. 按季度统计
data['Quarter'] = data['DATE OCC'].dt.to_period('Q')
quarterly_series = data.groupby('Quarter').size()
ax3 = axes[2]
quarterly_series.plot(kind='line', linewidth=2.5, color='purple', marker='s', 
                      markersize=5, ax=ax3)
ax3.fill_between(range(len(quarterly_series)), quarterly_series.values, 
                 alpha=0.3, color='purple')
ax3.set_title('Quarterly Crime Trend', fontsize=16, fontweight='bold', pad=20)
ax3.set_xlabel('Quarter', fontsize=12)
ax3.set_ylabel('Crime Count', fontsize=12)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'{output_dir}/10_crime_trend_timeline.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 犯罪趋势时间序列(月度/年度/季度)")
plt.close()

# 10d. 单独的年度对比图 - 带增长率
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 年度犯罪数量
yearly_series.plot(kind='bar', color='steelblue', ax=ax1, width=0.7)
ax1.set_title('Annual Crime Count with Year-over-Year Change', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Crime Count', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for i, v in enumerate(yearly_series.values):
    ax1.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 年度变化率
yearly_change = yearly_series.pct_change() * 100
colors = ['green' if x >= 0 else 'red' for x in yearly_change.values[1:]]
yearly_change.iloc[1:].plot(kind='bar', color=colors, ax=ax2, width=0.7)
ax2.set_title('Year-over-Year Crime Change Rate (%)', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Change Rate (%)', fontsize=12)
ax2.tick_params(axis='x', rotation=45)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(True, alpha=0.3, axis='y')

# 添加百分比标签
for i, v in enumerate(yearly_change.iloc[1:].values):
    ax2.text(i, v, f'{v:.1f}%', ha='center', 
            va='bottom' if v >= 0 else 'top', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/10d_yearly_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 年度对比及增长率")
plt.close()

# 10e. Top犯罪类型的时间趋势
top_crimes = data['Crm Cd Desc'].value_counts().head(5).index
plt.figure(figsize=(16, 8))

for crime in top_crimes:
    crime_data = data[data['Crm Cd Desc'] == crime]
    crime_time_series = crime_data.groupby('Year_Month').size()
    plt.plot(range(len(crime_time_series)), crime_time_series.values, 
            marker='o', markersize=3, linewidth=2, label=crime, alpha=0.8)

plt.title('Top 5 Crime Types - Trend Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Time (Year-Month)', fontsize=12)
plt.ylabel('Crime Count', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/10e_top_crimes_trend.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: Top 5犯罪类型时间趋势")
plt.close()

# 11. 热力图: 年份 vs 地区
plt.figure(figsize=(16, 10))
heatmap_data = data.groupby(['Year', 'AREA NAME']).size().unstack(fill_value=0)
sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt='d', cbar_kws={'label': 'Crime Count'})
plt.title('Crime Heatmap: Year vs Area', fontsize=16, fontweight='bold')
plt.xlabel('Area', fontsize=12)
plt.ylabel('Year', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{output_dir}/11_heatmap_year_area.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 年份vs地区热力图")
plt.close()

# 12. 按Part 1-2统计
plt.figure(figsize=(8, 6))
part_counts = data['Part 1-2'].value_counts()
part_counts.plot(kind='bar', color='orchid')
plt.title('Crime Count by Part 1-2 Classification', fontsize=16, fontweight='bold')
plt.xlabel('Part Classification', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{output_dir}/12_crime_by_part.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 按Part 1-2分类的犯罪统计")
plt.close()

# 生成统计摘要
summary = f"""
================== 犯罪数据统计摘要 ==================

数据总量: {len(data):,} 条记录

时间范围: {data['Year'].min()} - {data['Year'].max()}

Top 5 犯罪类型:
{crime_counts.head().to_string()}

Top 5 犯罪地区:
{area_counts.head().to_string()}

受害者性别分布:
{sex_counts.to_string()}

案件状态分布:
{status_counts.to_string()}

年度犯罪统计:
{yearly_counts.to_string()}

所有图表已保存至: {output_dir}
=====================================================
"""

print(summary)

# 保存摘要到文件
with open(f'{output_dir}/statistics_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"\n✅ 所有可视化图表已生成完成!")
print(f"📁 输出目录: {output_dir}")
