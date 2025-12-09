import pandas as pd
import numpy as np

csvfile = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/Crime_Data_from_2020_to_Present_20250929.csv"

data = pd.read_csv(csvfile)

row_name = data.columns.tolist()
print("="*80)
print("📊 数据集列名和数据类型分析")
print("="*80)
print(f"\n总列数: {len(row_name)}")
print(f"总行数: {data.shape[0]:,}")

print("\n" + "="*80)
print("列名和数据类型:")
print("="*80)

# Print each column name and its data type
for col in row_name:
    print(f"{col}: {data[col].dtype}")

# New functionality: Check unique values for each column
print("\n" + "="*80)
print("【唯一值分析】- 唯一值数量 < 总行数50%的列")
print("="*80)

total_rows = data.shape[0]

for col in row_name:
    unique_count = data[col].nunique()
    unique_pct = unique_count / total_rows
    
    # 只显示唯一值少于50%的列
    if unique_count < total_rows * 0.5:
        print(f"\n📌 列名: '{col}'")
        print(f"   数据类型: {data[col].dtype}")
        print(f"   唯一值数量: {unique_count:,}")
        print(f"   唯一值占比: {unique_pct:.2%}")
        print(f"   缺失值: {data[col].isnull().sum():,}")
        
        # 获取前N个最常见的值（按频率排序）
        top_n = min(10, unique_count)  # 显示前10个或全部（如果少于10个）
        value_counts = data[col].value_counts()
        
        print(f"   前{top_n}个最常见的值:")
        for i, (val, count) in enumerate(value_counts.head(top_n).items(), 1):
            pct = (count / total_rows) * 100
            print(f"      {i}. {val}: {count:,} ({pct:.2f}%)")
        
        # 如果还有更多唯一值
        if unique_count > top_n:
            print(f"   ... 还有 {unique_count - top_n:,} 个唯一值未显示")

print("\n" + "="*80)
print("✅ 分析完成")
print("="*80)