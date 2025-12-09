import pandas as pd
import numpy as np
from pathlib import Path

# 文件路径列表
file_paths = [
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/AGGRESSIVE_AGGRESSIVE_VOTING_REAL_F1_0.7521489971346705.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/GRADIENT_TUNE_10PCT_REAL_F1_0.7611336032388665.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/GRADIENT_TUNE_7PCT_REAL_F1_0.7531847133757962.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/GRADIENT_TUNE_8PCT_REAL_F1_0.7528174305033809.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/GRADIENT_TUNE_9PCT_REAL_F1_0.7533759772565743.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/result.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/submit.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold1_bad_f1_0.7778_good_0.9765_bad_0.7778_macro_0.8771_weighted_0.9570_seed_13_REAL_F1_0.7549378200438918.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold1_bad_f1_0.8083_good_0.9803_bad_0.8083_macro_0.8943_weighted_0.9634_seed_13_REAL_F1_0.7628549501151188.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold4_bad_f1_0.8250_good_0.9814_bad_0.8250_macro_0.9032_weighted_0.9661_seed_13_REAL_F1_0.7525325615050651_REAL_F1_0.7525325615050651.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold5_bad_f1_0.8401_good_0.9838_bad_0.8401_macro_0.9119_weighted_0.9697_seed_13_REAL_F1_0.7579273008507347.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/best.csv"
]

# 读取best.csv作为基准
best_df = pd.read_csv("/Users/mannormal/Desktop/课程/y4t1/stat 4011/best.csv")
print("=" * 80)
print("BEST.CSV 分析")
print("=" * 80)
print(f"总样本数: {len(best_df)}")
print(f"0的数量: {(best_df['Predict'] == 0).sum()}")
print(f"1的数量: {(best_df['Predict'] == 1).sum()}")
print(f"0的比例: {(best_df['Predict'] == 0).sum() / len(best_df) * 100:.2f}%")
print(f"1的比例: {(best_df['Predict'] == 1).sum() / len(best_df) * 100:.2f}%")
print()

# 分析每个文件
results = []

for file_path in file_paths:
    file_name = Path(file_path).name
    
    # 跳过best.csv本身
    if file_name == "best.csv":
        continue
    
    try:
        # 读取文件
        df = pd.read_csv(file_path)
        
        # 确保ID列对齐
        merged = best_df.merge(df, on='ID', suffixes=('_best', '_current'))
        
        # 计算一致性
        consistent = (merged['Predict_best'] == merged['Predict_current']).sum()
        total = len(merged)
        consistency_rate = consistent / total * 100 if total > 0 else 0
        
        # 统计0和1的数量
        count_0 = (df['Predict'] == 0).sum()
        count_1 = (df['Predict'] == 1).sum()
        
        # 计算差异
        diff_count = total - consistent
        
        # 找出不一致的ID
        diff_ids = merged[merged['Predict_best'] != merged['Predict_current']]['ID'].tolist()
        
        # 详细分析：预测为0和1中的一致性
        # 当前文件预测为0的样本
        current_pred_0 = merged[merged['Predict_current'] == 0]
        current_pred_0_consistent = (current_pred_0['Predict_best'] == current_pred_0['Predict_current']).sum()
        current_pred_0_total = len(current_pred_0)
        pred_0_consistency_rate = current_pred_0_consistent / current_pred_0_total * 100 if current_pred_0_total > 0 else 0
        
        # 当前文件预测为1的样本
        current_pred_1 = merged[merged['Predict_current'] == 1]
        current_pred_1_consistent = (current_pred_1['Predict_best'] == current_pred_1['Predict_current']).sum()
        current_pred_1_total = len(current_pred_1)
        pred_1_consistency_rate = current_pred_1_consistent / current_pred_1_total * 100 if current_pred_1_total > 0 else 0
        
        results.append({
            'file_name': file_name,
            'total_samples': total,
            'consistent_samples': consistent,
            'different_samples': diff_count,
            'consistency_rate': consistency_rate,
            'count_0': count_0,
            'count_1': count_1,
            'ratio_0': count_0 / total * 100 if total > 0 else 0,
            'ratio_1': count_1 / total * 100 if total > 0 else 0,
            'diff_ids': diff_ids[:10],  # 只保存前10个不一致的ID
            'pred_0_total': current_pred_0_total,
            'pred_0_consistent': current_pred_0_consistent,
            'pred_0_consistency_rate': pred_0_consistency_rate,
            'pred_1_total': current_pred_1_total,
            'pred_1_consistent': current_pred_1_consistent,
            'pred_1_consistency_rate': pred_1_consistency_rate
        })
        
    except Exception as e:
        print(f"读取文件 {file_name} 时出错: {e}")
        continue

# 按一致性排序
results.sort(key=lambda x: x['consistency_rate'], reverse=True)

print("=" * 80)
print("文件一致性分析 (与 best.csv 比较)")
print("=" * 80)
print()

for i, result in enumerate(results, 1):
    print(f"{i}. {result['file_name']}")
    print(f"   一致性: {result['consistency_rate']:.2f}% ({result['consistent_samples']}/{result['total_samples']})")
    print(f"   不同样本数: {result['different_samples']}")
    print(f"   预测为0: {result['count_0']} ({result['ratio_0']:.2f}%)")
    print(f"      - 预测为0中一致的: {result['pred_0_consistent']}/{result['pred_0_total']} ({result['pred_0_consistency_rate']:.2f}%)")
    print(f"   预测为1: {result['count_1']} ({result['ratio_1']:.2f}%)")
    print(f"      - 预测为1中一致的: {result['pred_1_consistent']}/{result['pred_1_total']} ({result['pred_1_consistency_rate']:.2f}%)")
    
    if result['diff_ids']:
        print(f"   部分不一致的ID: {', '.join(result['diff_ids'][:5])}")
    print()

# 保存详细结果到CSV
results_df = pd.DataFrame([{
    '文件名': r['file_name'],
    '总样本数': r['total_samples'],
    '一致样本数': r['consistent_samples'],
    '不同样本数': r['different_samples'],
    '一致性率(%)': f"{r['consistency_rate']:.2f}",
    '预测为0': r['count_0'],
    '预测为1': r['count_1'],
    '0的比例(%)': f"{r['ratio_0']:.2f}",
    '1的比例(%)': f"{r['ratio_1']:.2f}",
    '预测0中一致': r['pred_0_consistent'],
    '预测0总数': r['pred_0_total'],
    '预测0一致率(%)': f"{r['pred_0_consistency_rate']:.2f}",
    '预测1中一致': r['pred_1_consistent'],
    '预测1总数': r['pred_1_total'],
    '预测1一致率(%)': f"{r['pred_1_consistency_rate']:.2f}"
} for r in results])

output_path = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/consistency_analysis.csv"
results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"详细分析结果已保存到: {output_path}")

# 找出最一致的文件
print("\n" + "=" * 80)
print("最一致的文件 (前3名)")
print("=" * 80)
for i, result in enumerate(results[:3], 1):
    print(f"{i}. {result['file_name']}: {result['consistency_rate']:.2f}% 一致")

# 分析预测分布差异
print("\n" + "=" * 80)
print("预测分布分析")
print("=" * 80)
best_ratio_1 = (best_df['Predict'] == 1).sum() / len(best_df) * 100
print(f"best.csv 中1的比例: {best_ratio_1:.2f}%")
print()

for result in results[:5]:  # 显示前5个
    ratio_diff = result['ratio_1'] - best_ratio_1
    print(f"{result['file_name'][:50]:50s} | 1的比例: {result['ratio_1']:6.2f}% | 差异: {ratio_diff:+6.2f}%")
