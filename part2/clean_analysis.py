import pandas as pd

# 读取数据
csvfile = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/Crime_Data_from_2010_to_2019_20250929.csv"
data = pd.read_csv(csvfile)

print("="*80)
print("📊 有缺失值的列分析")
print("="*80)

# 获取有缺失值的列
missing_cols = data.columns[data.isnull().any()].tolist()

print(f"\n有缺失值的列数量: {len(missing_cols)}")
print(f"列名: {missing_cols}")

# 遍历有缺失值的列
for col in missing_cols:
    unique_count = data[col].nunique()
    missing_count = data[col].isnull().sum()
    total_rows = data.shape[0]
    unique_pct = unique_count / total_rows * 100

    print(f"\n📌 列名: '{col}'")
    print(f"   数据类型: {data[col].dtype}")
    print(f"   缺失值数量: {missing_count:,}")
    print(f"   唯一值数量: {unique_count:,}")
    print(f"   唯一值占比: {unique_pct:.2f}%")

    # 如果唯一值数量小于100，打印所有唯一值以及占比
    if unique_count < 100:
        print(f"   所有唯一值及其占比:")
        value_counts = data[col].value_counts(dropna=True)
        for val, count in value_counts.items():
            pct = count / total_rows * 100
            print(f"      {val}: {count:,} ({pct:.2f}%)")

print("\n" + "="*80)
print("✅ 分析完成")
print("="*80)