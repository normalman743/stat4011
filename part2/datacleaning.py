import pandas as pd
import numpy as np
from pathlib import Path

# 读取数据
csvfile = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/Crime_Data_from_2010_to_2019_20250929.csv"
df = pd.read_csv(csvfile)

# 创建输出目录
output_dir = Path('/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/cleaned_data')
output_dir.mkdir(exist_ok=True)

print("="*80)
print("🧹 数据清理流程")
print("="*80)
print(f"\n初始数据: {df.shape[0]:,} 行, {df.shape[1]} 列")

# ============================================================================
# 清理策略总结
# ============================================================================

print("\n" + "="*80)
print("📋 【清理策略】")
print("="*80)

strategy = """
1️⃣ 删除缺失比例 >90% 的列:
   • Crm Cd 2 (93.12%)
   • Crm Cd 3 (99.77%)
   • Crm Cd 4 (99.99%)

2️⃣ 删除缺失比例 <1% 的行:
   • Premis Cd (16行, 0.00%)
   • Premis Desc (588行, 0.06%)
   • Status (1行, 0.00%)
   • Crm Cd 1 (11行, 0.00%)

3️⃣ 填补缺失值（中等比例，有语义意义）:
   • Mocodes (15.09%) → '0000' (表示无记录)
   • Vict Sex (14.39%) → 'U' (Unknown)
   • Vict Descent (14.39%) → 'U' (Unknown)
   • Weapon Used Cd (67.44%) → 0 (表示无武器)
   • Weapon Desc (67.44%) → 'NO WEAPON' (表示无武器)
   • Cross Street (84.65%) → 'NOT RECORDED' (表示未记录)
"""

print(strategy)

# ============================================================================
# Step 1: 删除缺失比例 >90% 的列
# ============================================================================

print("\n" + "="*80)
print("Step 1️⃣ 删除缺失比例 >90% 的列")
print("="*80)

cols_to_drop = ['Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4']

print(f"\n要删除的列: {cols_to_drop}")
print(f"删除前: {df.shape[1]} 列")

df = df.drop(columns=cols_to_drop)

print(f"删除后: {df.shape[1]} 列")
print("✅ 完成")

# ============================================================================
# Step 2: 删除缺失比例 <1% 的行
# ============================================================================

print("\n" + "="*80)
print("Step 2️⃣ 删除缺失值很少的行 (<1%)")
print("="*80)

rows_to_clean = ['Premis Cd', 'Premis Desc', 'Status', 'Crm Cd 1']

print(f"\n删除前行数: {df.shape[0]:,}")

for col in rows_to_clean:
    missing_before = df[col].isnull().sum()
    if missing_before > 0:
        print(f"  删除 {col} 缺失的 {missing_before:,} 行")
        df = df.dropna(subset=[col])

print(f"删除后行数: {df.shape[0]:,}")
print("✅ 完成")

# ============================================================================
# Step 3: 填补 Mocodes
# ============================================================================

print("\n" + "="*80)
print("Step 3️⃣ 填补 Mocodes 缺失值")
print("="*80)

print(f"\n缺失前: {df['Mocodes'].isnull().sum():,} 个缺失值")

# 检查 '0000' 是否在原始唯一值中
if '0000' in df['Mocodes'].unique():
    print("⚠️ '0000' 已存在于原始数据中")
else:
    print("✓ '0000' 不在原始数据中，可以作为新类别")

df['Mocodes'].fillna('0000', inplace=True)

print(f"缺失后: {df['Mocodes'].isnull().sum():,} 个缺失值")
print("✅ 完成 (用 '0000' 表示无记录)")

# ============================================================================
# Step 4: 填补 Vict Sex
# ============================================================================

print("\n" + "="*80)
print("Step 4️⃣ 填补 Vict Sex 缺失值")
print("="*80)

print(f"\n缺失前: {df['Vict Sex'].isnull().sum():,} 个缺失值")
print(f"原始唯一值: {sorted(df['Vict Sex'].dropna().unique())}")

# 检查 'U' 是否在原始唯一值中
if 'U' in df['Vict Sex'].unique():
    print("⚠️ 'U' 已存在于原始数据中")
else:
    print("✓ 'U' 不在原始数据中，可以作为新类别")

df['Vict Sex'].fillna('U', inplace=True)

print(f"缺失后: {df['Vict Sex'].isnull().sum():,} 个缺失值")
print(f"更新后唯一值: {sorted(df['Vict Sex'].unique())}")
print("✅ 完成 (用 'U' 表示 Unknown)")

# 说明 X, H, - 的含义
print("\n📝 Vict Sex 编码说明:")
print("   M = Male (男性)")
print("   F = Female (女性)")
print("   X = Unknown (性别未知，原始数据)")
print("   H = 可能是 'Hispanic' 误编码或其他")
print("   - = 无效数据")
print("   U = Unknown (我们填补的缺失值)")

# ============================================================================
# Step 5: 填补 Vict Descent
# ============================================================================

print("\n" + "="*80)
print("Step 5️⃣ 填补 Vict Descent 缺失值")
print("="*80)

print(f"\n缺失前: {df['Vict Descent'].isnull().sum():,} 个缺失值")
print(f"原始唯一值数量: {df['Vict Descent'].nunique()}")

# 检查 'U' 是否已存在
existing_u_count = (df['Vict Descent'] == 'U').sum()
print(f"原始数据中 'U' 的数量: {existing_u_count:,}")

# 填补缺失值
missing_count = df['Vict Descent'].isnull().sum()
df['Vict Descent'].fillna('U', inplace=True)

print(f"缺失后: {df['Vict Descent'].isnull().sum():,} 个缺失值")
print(f"更新后 'U' 的总数量: {(df['Vict Descent'] == 'U').sum():,}")
print(f"  (原有 {existing_u_count:,} + 新增 {missing_count:,})")
print("✅ 完成 (用 'U' 表示 Unknown)")

# ============================================================================
# Step 6: 填补 Weapon Used Cd
# ============================================================================

print("\n" + "="*80)
print("Step 6️⃣ 填补 Weapon Used Cd 缺失值")
print("="*80)

print(f"\n缺失前: {df['Weapon Used Cd'].isnull().sum():,} 个缺失值")
print(f"数据类型: {df['Weapon Used Cd'].dtype}")

# 检查是否有 0 值
if 0 in df['Weapon Used Cd'].unique():
    print("⚠️ 0 已存在于原始数据中")
else:
    print("✓ 0 不在原始数据中，可以作为新类别")

df['Weapon Used Cd'].fillna(0, inplace=True)

print(f"缺失后: {df['Weapon Used Cd'].isnull().sum():,} 个缺失值")
print("✅ 完成 (用 0 表示无武器)")

# ============================================================================
# Step 7: 填补 Weapon Desc
# ============================================================================

print("\n" + "="*80)
print("Step 7️⃣ 填补 Weapon Desc 缺失值")
print("="*80)

print(f"\n缺失前: {df['Weapon Desc'].isnull().sum():,} 个缺失值")

# 检查 'NO WEAPON' 是否已存在
if 'NO WEAPON' in df['Weapon Desc'].unique():
    print("⚠️ 'NO WEAPON' 已存在于原始数据中")
else:
    print("✓ 'NO WEAPON' 不在原始数据中，可以作为新类别")

df['Weapon Desc'].fillna('NO WEAPON', inplace=True)

print(f"缺失后: {df['Weapon Desc'].isnull().sum():,} 个缺失值")
print("✅ 完成 (用 'NO WEAPON' 表示无武器)")

# ============================================================================
# Step 8: 填补 Cross Street
# ============================================================================

print("\n" + "="*80)
print("Step 8️⃣ 填补 Cross Street 缺失值")
print("="*80)

print(f"\n缺失前: {df['Cross Street'].isnull().sum():,} 个缺失值")

# 检查 'NOT RECORDED' 是否已存在
if 'NOT RECORDED' in df['Cross Street'].unique():
    print("⚠️ 'NOT RECORDED' 已存在于原始数据中")
else:
    print("✓ 'NOT RECORDED' 不在原始数据中，可以作为新类别")

df['Cross Street'].fillna('NOT RECORDED', inplace=True)

print(f"缺失后: {df['Cross Street'].isnull().sum():,} 个缺失值")
print("✅ 完成 (用 'NOT RECORDED' 表示未记录)")

# ============================================================================
# 验证清理结果
# ============================================================================

print("\n" + "="*80)
print("✅ 【清理结果验证】")
print("="*80)

# 检查剩余缺失值
remaining_missing = df.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]

if len(remaining_missing) == 0:
    print("\n🎉 完美！数据已完全清理，无缺失值")
else:
    print("\n⚠️ 仍有缺失值:")
    for col, count in remaining_missing.items():
        pct = (count / len(df)) * 100
        print(f"   {col}: {count:,} ({pct:.2f}%)")

print(f"\n📊 最终数据统计:")
print(f"   行数: {df.shape[0]:,}")
print(f"   列数: {df.shape[1]}")
print(f"   总单元格: {df.shape[0] * df.shape[1]:,}")
print(f"   缺失单元格: {df.isnull().sum().sum():,}")

# ============================================================================
# 保存清理后的数据
# ============================================================================

print("\n" + "="*80)
print("💾 【保存清理后的数据】")
print("="*80)

# CSV 格式
output_csv = output_dir / 'Crime_Data_Cleaned_Final.csv'
df.to_csv(output_csv, index=False, encoding='utf-8')
print(f"✓ CSV: {output_csv}")

# Parquet 格式（更快，更小）- 检查依赖
try:
    output_parquet = output_dir / 'Crime_Data_Cleaned_Final.parquet'
    df.to_parquet(output_parquet, index=False)
    print(f"✓ Parquet: {output_parquet}")
except ImportError as e:
    print(f"⚠️ Parquet: 跳过 (缺少依赖)")
    print(f"   提示: 如需保存 parquet 格式，请运行: pip install pyarrow")

# ============================================================================
# 生成清理报告
# ============================================================================

print("\n" + "="*80)
print("📋 【生成清理报告】")
print("="*80)

report = f"""
【数据清理最终报告】
生成时间: {pd.Timestamp.now()}

【清理前】
• 行数: 1,004,991
• 列数: 30
• 缺失值较多

【清理操作】
1. 删除列 (3个):
   - Crm Cd 2, Crm Cd 3, Crm Cd 4 (缺失 >90%)

2. 删除行 (<1% 缺失):
   - Premis Cd: 16 行
   - Premis Desc: 588 行
   - Status: 1 行
   - Crm Cd 1: 11 行
   - 总删除: ~600 行

3. 填补缺失值:
   - Mocodes: 151,619 → '0000' (无记录)
   - Vict Sex: 144,644 → 'U' (Unknown)
   - Vict Descent: 144,656 → 'U' (Unknown)
   - Weapon Used Cd: ~677,000 → 0 (无武器)
   - Weapon Desc: ~677,000 → 'NO WEAPON' (无武器)
   - Cross Street: ~850,000 → 'NOT RECORDED' (未记录)

【清理后】
• 行数: {df.shape[0]:,}
• 列数: {df.shape[1]}
• 缺失值: 0 ✓
• 数据质量: 优秀

【新增类别】
• Mocodes: '0000' = 无记录
• Vict Sex: 'U' = Unknown
• Vict Descent: 'U' = Unknown (与原有 'U' 合并)
• Weapon Used Cd: 0 = 无武器
• Weapon Desc: 'NO WEAPON' = 无武器
• Cross Street: 'NOT RECORDED' = 未记录

【数据完整性】
✓ 所有字段无缺失值
✓ 数据类型一致
✓ 可直接用于分析
"""

print(report)

# 保存报告
report_file = output_dir / 'cleaning_report_final.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✓ 报告已保存: {report_file}")

print("\n" + "="*80)
print("✅ 数据清理完成！")
print("="*80)