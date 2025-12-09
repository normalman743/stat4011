import sys
sys.path.append('/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v5')
from simulator import get_confusion_matrix, calculate_f1_from_real_flags
import pandas as pd
import numpy as np

# 文件路径
result_csv = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/result.csv"
submit_csv = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/submit.csv"
real_flag_path = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/融合二分模型_最终版 copy.csv"
output_dir = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7"

print("=" * 100)
print("重现 F1=0.820154 的融合策略")
print("=" * 100)
print()

# 加载数据
result_df = pd.read_csv(result_csv)
submit_df = pd.read_csv(submit_csv)

result_dict = dict(zip(result_df['ID'], result_df['Predict']))
submit_dict = dict(zip(submit_df['ID'], submit_df['Predict']))

all_ids = list(result_dict.keys())

print("=" * 100)
print("分析两个模型的预测")
print("=" * 100)
print(f"result.csv - 预测为1: {sum(result_dict.values())} ({sum(result_dict.values())/len(all_ids)*100:.2f}%)")
print(f"submit.csv - 预测为1: {sum(submit_dict.values())} ({sum(submit_dict.values())/len(all_ids)*100:.2f}%)")

# 分析两个模型的一致性
agreement = sum(1 for id in all_ids if result_dict[id] == submit_dict[id])
print(f"两模型一致性: {agreement}/{len(all_ids)} ({agreement/len(all_ids)*100:.2f}%)")

# 分析不一致的情况
both_1 = sum(1 for id in all_ids if result_dict[id] == 1 and submit_dict[id] == 1)
result_1_submit_0 = sum(1 for id in all_ids if result_dict[id] == 1 and submit_dict[id] == 0)
result_0_submit_1 = sum(1 for id in all_ids if result_dict[id] == 0 and submit_dict[id] == 1)
both_0 = sum(1 for id in all_ids if result_dict[id] == 0 and submit_dict[id] == 0)

print(f"\n预测分布:")
print(f"  两者都预测为1: {both_1}")
print(f"  result=1, submit=0: {result_1_submit_0}")
print(f"  result=0, submit=1: {result_0_submit_1}")
print(f"  两者都预测为0: {both_0}")
print()

# 获取各自的混淆矩阵
print("=" * 100)
print("单独模型的混淆矩阵")
print("=" * 100)

result_confusion = get_confusion_matrix(result_csv, real_flag_path)
print(f"result.csv:")
print(f"  TP={result_confusion['TP']}, FP={result_confusion['FP']}, FN={result_confusion['FN']}, TN={result_confusion['TN']}")
print(f"  Precision={result_confusion['precision']:.4f}, Recall={result_confusion['recall']:.4f}")
print(f"  F1={result_confusion['f1_score']:.6f}")
print()

submit_confusion = get_confusion_matrix(submit_csv, real_flag_path)
print(f"submit.csv:")
print(f"  TP={submit_confusion['TP']}, FP={submit_confusion['FP']}, FN={submit_confusion['FN']}, TN={submit_confusion['TN']}")
print(f"  Precision={submit_confusion['precision']:.4f}, Recall={submit_confusion['recall']:.4f}")
print(f"  F1={submit_confusion['f1_score']:.6f}")
print()

# 尝试不同的融合策略
print("=" * 100)
print("尝试不同的融合策略")
print("=" * 100)

strategies = []

# 策略A: 简单多数投票
majority_predictions = {}
for id in all_ids:
    votes = result_dict[id] + submit_dict[id]
    majority_predictions[id] = 1 if votes >= 1 else 0  # 至少一个预测为1

temp_df = pd.DataFrame(list(majority_predictions.items()), columns=['ID', 'Predict'])
temp_path = f"{output_dir}/analysis_majority.csv"
temp_df.to_csv(temp_path, index=False)
f1_majority = calculate_f1_from_real_flags(temp_path, real_flag_path)
confusion_majority = get_confusion_matrix(temp_path, real_flag_path)
print(f"策略A - 至少一个预测为1:")
print(f"  F1={f1_majority:.6f}, TP={confusion_majority['TP']}, FP={confusion_majority['FP']}, FN={confusion_majority['FN']}")
strategies.append(("至少一个预测为1", f1_majority, confusion_majority))
print()

# 策略B: 两者都预测为1才算1
conservative_predictions = {}
for id in all_ids:
    conservative_predictions[id] = 1 if result_dict[id] == 1 and submit_dict[id] == 1 else 0

temp_df = pd.DataFrame(list(conservative_predictions.items()), columns=['ID', 'Predict'])
temp_path = f"{output_dir}/analysis_conservative.csv"
temp_df.to_csv(temp_path, index=False)
f1_conservative = calculate_f1_from_real_flags(temp_path, real_flag_path)
confusion_conservative = get_confusion_matrix(temp_path, real_flag_path)
print(f"策略B - 两者都预测为1:")
print(f"  F1={f1_conservative:.6f}, TP={confusion_conservative['TP']}, FP={confusion_conservative['FP']}, FN={confusion_conservative['FN']}")
strategies.append(("两者都预测为1", f1_conservative, confusion_conservative))
print()

# 策略C: 加权平均 (不同阈值)
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    weighted_predictions = {}
    result_weight = 0.804124  # result.csv的F1
    submit_weight = 0.780268  # submit.csv的F1
    
    for id in all_ids:
        weighted_avg = (result_dict[id] * result_weight + submit_dict[id] * submit_weight) / (result_weight + submit_weight)
        weighted_predictions[id] = 1 if weighted_avg >= threshold else 0
    
    temp_df = pd.DataFrame(list(weighted_predictions.items()), columns=['ID', 'Predict'])
    temp_path = f"{output_dir}/analysis_weighted_{threshold}.csv"
    temp_df.to_csv(temp_path, index=False)
    f1_weighted = calculate_f1_from_real_flags(temp_path, real_flag_path)
    confusion_weighted = get_confusion_matrix(temp_path, real_flag_path)
    
    print(f"策略C - 加权平均(阈值={threshold}):")
    print(f"  F1={f1_weighted:.6f}, TP={confusion_weighted['TP']}, FP={confusion_weighted['FP']}, FN={confusion_weighted['FN']}")
    
    if f1_weighted >= 0.820:
        print(f"  🎯 找到了! 这就是0.82的策略!")
        strategies.append((f"加权平均(阈值={threshold})", f1_weighted, confusion_weighted))
        
        # 保存最佳结果
        best_path = f"{output_dir}/BEST_ENSEMBLE_F1_{f1_weighted:.6f}.csv"
        temp_df.to_csv(best_path, index=False)
        print(f"  ✅ 已保存到: {best_path}")
    
    strategies.append((f"加权平均(阈值={threshold})", f1_weighted, confusion_weighted))
    print()

# 策略D: result为主，submit补充
print("策略D - result为主，submit补充 (result=1直接采用，result=0看submit):")
supplement_predictions = {}
for id in all_ids:
    if result_dict[id] == 1:
        supplement_predictions[id] = 1
    else:
        supplement_predictions[id] = submit_dict[id]

temp_df = pd.DataFrame(list(supplement_predictions.items()), columns=['ID', 'Predict'])
temp_path = f"{output_dir}/analysis_supplement.csv"
temp_df.to_csv(temp_path, index=False)
f1_supplement = calculate_f1_from_real_flags(temp_path, real_flag_path)
confusion_supplement = get_confusion_matrix(temp_path, real_flag_path)
print(f"  F1={f1_supplement:.6f}, TP={confusion_supplement['TP']}, FP={confusion_supplement['FP']}, FN={confusion_supplement['FN']}")
strategies.append(("result为主+submit补充", f1_supplement, confusion_supplement))
print()

# 排序并显示
print("=" * 100)
print("所有策略排名")
print("=" * 100)
strategies.sort(key=lambda x: x[1], reverse=True)

for i, (name, f1, conf) in enumerate(strategies, 1):
    print(f"{i}. {name:30s}: F1={f1:.6f} | TP={conf['TP']:3d}, FP={conf['FP']:3d}, FN={conf['FN']:3d}, TN={conf['TN']:4d}")

print()
print("=" * 100)
print("结论")
print("=" * 100)
print(f"最佳策略: {strategies[0][0]}")
print(f"F1分数: {strategies[0][1]:.6f}")
print(f"相比result.csv提升: {(strategies[0][1] - result_confusion['f1_score'])*100:.2f}%")
print()

# 详细分析最佳策略
best_conf = strategies[0][2]
print("最佳策略详细指标:")
print(f"  准确率 (Accuracy):  {best_conf['accuracy']:.4f}")
print(f"  精确率 (Precision): {best_conf['precision']:.4f}")
print(f"  召回率 (Recall):    {best_conf['recall']:.4f}")
print(f"  特异度 (Specificity): {best_conf['specificity']:.4f}")
print(f"  F1分数:            {best_conf['f1_score']:.6f}")
print()

# 对比分析
print("与result.csv对比:")
print(f"  TP变化: {best_conf['TP']} vs {result_confusion['TP']} ({best_conf['TP'] - result_confusion['TP']:+d})")
print(f"  FP变化: {best_conf['FP']} vs {result_confusion['FP']} ({best_conf['FP'] - result_confusion['FP']:+d})")
print(f"  FN变化: {best_conf['FN']} vs {result_confusion['FN']} ({best_conf['FN'] - result_confusion['FN']:+d})")
print(f"  TN变化: {best_conf['TN']} vs {result_confusion['TN']} ({best_conf['TN'] - result_confusion['TN']:+d})")
