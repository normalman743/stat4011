import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Create output folder
output_dir = Path('/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/data_enhancement_output')
output_dir.mkdir(exist_ok=True)

# Read data
df = pd.read_csv('/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/Crime_Data_from_2020_to_Present_20250929.csv')

print("="*80)
print("📊 数据增强与缺失值分析")
print("="*80)

print(f"\n初始数据形状: {df.shape}")
print(f"行数: {df.shape[0]:,}, 列数: {df.shape[1]}")

# ============================================================================
# === 1. 缺失值分析 ===
# ============================================================================

print("\n" + "="*80)
print("📋 【缺失值分析】")
print("="*80)

# 计算缺失值
missing_data = pd.DataFrame({
    '列名': df.columns,
    '缺失数量': df.isnull().sum().values,
    '缺失比例(%)': (df.isnull().sum().values / len(df) * 100).round(2),
    '数据类型': df.dtypes.values
})

# 按缺失数量排序
missing_data = missing_data[missing_data['缺失数量'] > 0].sort_values('缺失数量', ascending=False)

print("\n缺失值详细统计:")
print(missing_data.to_string(index=False))

# 保存缺失值报告
missing_data.to_csv(output_dir / 'missing_values_report.csv', index=False, encoding='utf-8-sig')

# 统计总结
total_cells = df.shape[0] * df.shape[1]
missing_cells = df.isnull().sum().sum()
missing_pct = (missing_cells / total_cells * 100)

print(f"\n📈 总体统计:")
print(f"   总单元格数: {total_cells:,}")
print(f"   缺失单元格数: {missing_cells:,}")
print(f"   整体缺失比例: {missing_pct:.2f}%")

# ============================================================================
# === 2. 按缺失比例分类 ===
# ============================================================================

print("\n" + "="*80)
print("🔍 【按缺失程度分类】")
print("="*80)

# 分类标准
def classify_missing(pct):
    if pct == 0:
        return '完整'
    elif pct < 1:
        return '微小缺失'
    elif pct < 5:
        return '轻微缺失'
    elif pct < 20:
        return '中度缺失'
    else:
        return '严重缺失'

missing_data['缺失程度'] = missing_data['缺失比例(%)'].apply(classify_missing)

# 统计各类别
classification = missing_data['缺失程度'].value_counts()
print("\n缺失程度分布:")
print(classification)

# 打印各类别的字段
for level in ['完整', '微小缺失', '轻微缺失', '中度缺失', '严重缺失']:
    fields = missing_data[missing_data['缺失程度'] == level]['列名'].tolist()
    if fields:
        print(f"\n✓ {level} ({len(fields)}个字段):")
        for field in fields:
            missing_pct = missing_data[missing_data['列名'] == field]['缺失比例(%)'].values[0]
            missing_count = missing_data[missing_data['列名'] == field]['缺失数量'].values[0]
            print(f"   - {field}: {missing_count:,} ({missing_pct:.2f}%)")

# ============================================================================
# === 3. 关键字段缺失值详细分析 ===
# ============================================================================

print("\n" + "="*80)
print("🎯 【关键字段缺失值详细分析】")
print("="*80)

# 定义关键字段
key_fields = {
    'DATE OCC': '犯罪日期',
    'TIME OCC': '犯罪时间',
    'AREA NAME': '区域名称',
    'Crm Cd Desc': '犯罪描述',
    'Vict Age': '受害者年龄',
    'Vict Sex': '受害者性别',
    'Vict Descent': '受害者种族',
    'Premis Desc': '事发地点类型',
    'Weapon Desc': '武器描述',
    'Status Desc': '案件状态',
    'Part 1-2': '犯罪分类'
}

print("\n关键字段缺失值详情:")
for field, desc in key_fields.items():
    if field in df.columns:
        missing_count = df[field].isnull().sum()
        missing_pct = (missing_count / len(df) * 100)
        non_null = len(df) - missing_count
        
        print(f"\n📌 {field} ({desc}):")
        print(f"   缺失: {missing_count:,} ({missing_pct:.2f}%)")
        print(f"   有效: {non_null:,} ({100-missing_pct:.2f}%)")
        
        # 显示样本值
        sample_values = df[field].dropna().unique()[:5]
        print(f"   样本值: {sample_values}")

# ============================================================================
# === 4. 缺失值可视化 ===
# ============================================================================

print("\n" + "="*80)
print("🎨 【生成缺失值可视化】")
print("="*80)

# 创建缺失值柱状图
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('数据缺失值分析', fontsize=16, fontweight='bold')

# 绘图1: 缺失数量（降序）
missing_data_sorted = missing_data.sort_values('缺失数量', ascending=True).tail(20)
ax1 = axes[0]
bars = ax1.barh(range(len(missing_data_sorted)), missing_data_sorted['缺失数量'], 
                 color=plt.cm.Reds(np.linspace(0.3, 0.9, len(missing_data_sorted))))
ax1.set_yticks(range(len(missing_data_sorted)))
ax1.set_yticklabels(missing_data_sorted['列名'], fontsize=9)
ax1.set_xlabel('缺失数量')
ax1.set_title('Top 20 缺失数量最多的字段', fontsize=12)
ax1.grid(True, alpha=0.3, axis='x')

# 添加数值标签
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2.,
            f' {int(width):,}',
            ha='left', va='center', fontsize=8)

# 绘图2: 缺失比例
missing_data_sorted2 = missing_data[missing_data['缺失比例(%)'] > 0].sort_values('缺失比例(%)', ascending=True).tail(20)
ax2 = axes[1]
colors_pct = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(missing_data_sorted2)))
bars2 = ax2.barh(range(len(missing_data_sorted2)), missing_data_sorted2['缺失比例(%)'],
                  color=colors_pct)
ax2.set_yticks(range(len(missing_data_sorted2)))
ax2.set_yticklabels(missing_data_sorted2['列名'], fontsize=9)
ax2.set_xlabel('缺失比例 (%)')
ax2.set_title('Top 20 缺失比例最高的字段', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')

# 添加数值标签
for i, bar in enumerate(bars2):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2.,
            f' {width:.2f}%',
            ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'missing_values_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ 缺失值可视化已保存: missing_values_visualization.png")

# ============================================================================
# === 5. 数据增强建议 ===
# ============================================================================

print("\n" + "="*80)
print("💡 【数据补充建议】")
print("="*80)

recommendations = {
    '删除法': {
        '条件': '缺失比例 > 50%',
        '说明': '直接删除这些字段（信息量太低）',
        '字段': []
    },
    '前向填充': {
        '条件': '时间序列字段，缺失 < 5%',
        '说明': '用前面的值填充',
        '字段': []
    },
    '众数填充': {
        '条件': '分类字段，缺失 < 10%',
        '说明': '用该字段最常出现的值填充',
        '字段': []
    },
    '平均值填充': {
        '条件': '数值字段，缺失 < 10%',
        '说明': '用该字段平均值填充',
        '字段': []
    },
    '特殊值填充': {
        '条件': '缺失有特殊含义',
        '说明': '用特殊标记（如"未知"）填充',
        '字段': []
    }
}

# 分类字段
for idx, row in missing_data.iterrows():
    field = row['列名']
    pct = row['缺失比例(%)']
    dtype = row['数据类型']
    
    if pct > 50:
        recommendations['删除法']['字段'].append(field)
    elif pct > 0 and pct < 5:
        if 'float' in str(dtype) or 'int' in str(dtype):
            recommendations['平均值填充']['字段'].append(field)
        else:
            recommendations['众数填充']['字段'].append(field)
    elif pct >= 5 and pct < 10:
        recommendations['众数填充']['字段'].append(field)
    elif pct >= 10:
        recommendations['特殊值填充']['字段'].append(field)

# 打印建议
for method, info in recommendations.items():
    if info['字段']:
        print(f"\n🔧 {method}:")
        print(f"   条件: {info['条件']}")
        print(f"   说明: {info['说明']}")
        print(f"   字段 ({len(info['字段'])}个):")
        for field in info['字段']:
            pct = missing_data[missing_data['列名'] == field]['缺失比例(%)'].values[0]
            print(f"      • {field} ({pct:.2f}%)")

# ============================================================================
# === 6. 生成修复建议表 ===
# ============================================================================

print("\n" + "="*80)
print("📋 【生成修复建议表】")
print("="*80)

repair_suggestions = []

for idx, row in missing_data.iterrows():
    field = row['列名']
    missing_count = row['缺失数量']
    missing_pct = row['缺失比例(%)']
    
    # 确定修复策略
    if missing_pct > 50:
        strategy = '删除字段'
        reason = '信息量太低（缺失>50%）'
        priority = '低'
    elif missing_pct == 0:
        strategy = '保留'
        reason = '无缺失值'
        priority = '无需处理'
    elif field in ['DATE OCC', 'TIME OCC']:
        strategy = '前向填充'
        reason = '时间序列字段'
        priority = '高'
    elif missing_pct < 1:
        strategy = '众数/平均值填充'
        reason = '缺失极少（<1%）'
        priority = '低'
    elif missing_pct < 5:
        strategy = '众数/平均值填充'
        reason = '缺失轻微（<5%）'
        priority = '中'
    elif missing_pct < 20:
        strategy = '特殊值填充（"未知"）'
        reason = '缺失中等（5-20%）'
        priority = '高'
    else:
        strategy = '特殊值填充（"未知"）'
        reason = '缺失较多（>20%）'
        priority = '高'
    
    repair_suggestions.append({
        '字段': field,
        '缺失数量': missing_count,
        '缺失比例(%)': missing_pct,
        '修复策略': strategy,
        '原因': reason,
        '优先级': priority
    })

repair_df = pd.DataFrame(repair_suggestions)
repair_df = repair_df.sort_values('优先级', key=lambda x: x.map({'高': 0, '中': 1, '低': 2, '无需处理': 3}))

print("\n修复建议表 (优先级排序):")
print(repair_df.to_string(index=False))

# 保存修复建议
repair_df.to_csv(output_dir / 'repair_suggestions.csv', index=False, encoding='utf-8-sig')
print("\n✓ 修复建议已保存: repair_suggestions.csv")

# ============================================================================
# === 7. 总结 ===
# ============================================================================

print("\n" + "="*80)
print("✅ 【分析总结】")
print("="*80)

print(f"""
📊 数据质量概览:
   • 总记录数: {len(df):,}
   • 总字段数: {df.shape[1]}
   • 完整字段: {len(missing_data[missing_data['缺失数量']==0]) + (df.shape[1] - len(missing_data))}
   • 有缺失的字段: {len(missing_data)}
   • 整体缺失比例: {missing_pct:.2f}%

🎯 下一步操作:
   1. 检查 repair_suggestions.csv 了解修复策略
   2. 根据优先级逐步处理缺失值
   3. 对于"高"优先级字段立即处理
   4. 运行数据修复脚本补充缺失值

📁 输出文件:
   • missing_values_report.csv - 缺失值详细报告
   • repair_suggestions.csv - 修复建议
   • missing_values_visualization.png - 可视化图表
""")

print("="*80)
print("分析完成！📈")
print("="*80)