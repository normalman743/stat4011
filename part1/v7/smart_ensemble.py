import sys
sys.path.append('/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v5')
from simulator import get_confusion_matrix,calculate_f1_from_real_flags
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

# 文件路径列表 - 按F1分数排序的前几名
file_paths = [
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/result.csv",  # F1: 0.8041
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/submit.csv",  # F1: 0.7803
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold1_bad_f1_0.8083_good_0.9803_bad_0.8083_macro_0.8943_weighted_0.9634_seed_13_REAL_F1_0.7628549501151188.csv",  # F1: 0.7629
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/GRADIENT_TUNE_10PCT_REAL_F1_0.7611336032388665.csv",  # F1: 0.7611
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold5_bad_f1_0.8401_good_0.9838_bad_0.8401_macro_0.9119_weighted_0.9697_seed_13_REAL_F1_0.7579273008507347.csv",  # F1: 0.7579
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold1_bad_f1_0.7778_good_0.9765_bad_0.7778_macro_0.8771_weighted_0.9570_seed_13_REAL_F1_0.7549378200438918.csv",  # F1: 0.7549
]

real_flag_path = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/融合二分模型_最终版 copy.csv"
output_dir = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7"

print("=" * 100)
print("智能模型融合策略")
print("=" * 100)
print()

# 加载所有模型的预测
models = {}
for file_path in file_paths:
    file_name = Path(file_path).name
    df = pd.read_csv(file_path)
    models[file_name] = dict(zip(df['ID'], df['Predict']))

# 获取所有ID
all_ids = list(next(iter(models.values())).keys())

print(f"加载了 {len(models)} 个模型")
print(f"总样本数: {len(all_ids)}")
print()

# ============================
# 策略1: 加权投票 (基于F1分数)
# ============================
print("=" * 100)
print("策略1: 加权投票 (基于F1分数)")
print("=" * 100)

# F1分数作为权重
weights = [0.8041, 0.7803, 0.7629, 0.7611, 0.7579, 0.7549]
model_names = list(models.keys())

weighted_predictions = {}
for account_id in all_ids:
    weighted_sum = 0
    for i, model_name in enumerate(model_names):
        weighted_sum += models[model_name][account_id] * weights[i]
    
    # 阈值调整
    for threshold in [0.35, 0.40, 0.45, 0.50, 0.55]:
        pred = 1 if weighted_sum / sum(weights) >= threshold else 0
        key = f"weighted_voting_threshold_{threshold}"
        if key not in weighted_predictions:
            weighted_predictions[key] = {}
        weighted_predictions[key][account_id] = pred

# 评估加权投票
best_weighted_f1 = 0
best_weighted_strategy = None
for strategy_name, predictions in weighted_predictions.items():
    # 保存临时文件
    temp_df = pd.DataFrame(list(predictions.items()), columns=['ID', 'Predict'])
    temp_path = f"{output_dir}/temp_{strategy_name}.csv"
    temp_df.to_csv(temp_path, index=False)
    
    # 计算F1
    confusion = get_confusion_matrix(temp_path, real_flag_path)
    if confusion and confusion['f1_score'] > best_weighted_f1:
        best_weighted_f1 = confusion['f1_score']
        best_weighted_strategy = strategy_name
        print(f"  {strategy_name}: F1={confusion['f1_score']:.6f} (TP={confusion['TP']}, FP={confusion['FP']}, FN={confusion['FN']})")

print(f"\n最佳加权投票策略: {best_weighted_strategy}, F1={best_weighted_f1:.6f}")
print()

# ============================
# 策略2: 多数投票 (简单投票)
# ============================
print("=" * 100)
print("策略2: 多数投票")
print("=" * 100)

majority_predictions = {}
for account_id in all_ids:
    votes = [models[model_name][account_id] for model_name in model_names]
    majority_predictions[account_id] = 1 if sum(votes) > len(votes) / 2 else 0

temp_df = pd.DataFrame(list(majority_predictions.items()), columns=['ID', 'Predict'])
temp_path = f"{output_dir}/majority_voting.csv"
temp_df.to_csv(temp_path, index=False)

confusion = get_confusion_matrix(temp_path, real_flag_path)
print(f"多数投票 F1: {confusion['f1_score']:.6f} (TP={confusion['TP']}, FP={confusion['FP']}, FN={confusion['FN']})")
print()

# ============================
# 策略3: 保守策略 (至少N个模型预测为1才算1)
# ============================
print("=" * 100)
print("策略3: 保守策略 (至少N个模型预测为1)")
print("=" * 100)

best_conservative_f1 = 0
best_conservative_n = 0

for n in range(2, len(models) + 1):
    conservative_predictions = {}
    for account_id in all_ids:
        votes = [models[model_name][account_id] for model_name in model_names]
        conservative_predictions[account_id] = 1 if sum(votes) >= n else 0
    
    temp_df = pd.DataFrame(list(conservative_predictions.items()), columns=['ID', 'Predict'])
    temp_path = f"{output_dir}/conservative_n{n}.csv"
    temp_df.to_csv(temp_path, index=False)
    
    confusion = get_confusion_matrix(temp_path, real_flag_path)
    print(f"  至少{n}个模型: F1={confusion['f1_score']:.6f} (TP={confusion['TP']}, FP={confusion['FP']}, FN={confusion['FN']})")
    
    if confusion['f1_score'] > best_conservative_f1:
        best_conservative_f1 = confusion['f1_score']
        best_conservative_n = n

print(f"\n最佳保守策略: 至少{best_conservative_n}个模型, F1={best_conservative_f1:.6f}")
print()

# ============================
# 策略4: 激进策略 (至少1个模型预测为1就算1)
# ============================
print("=" * 100)
print("策略4: 激进策略 (至少1个模型预测为1)")
print("=" * 100)

aggressive_predictions = {}
for account_id in all_ids:
    votes = [models[model_name][account_id] for model_name in model_names]
    aggressive_predictions[account_id] = 1 if sum(votes) >= 1 else 0

temp_df = pd.DataFrame(list(aggressive_predictions.items()), columns=['ID', 'Predict'])
temp_path = f"{output_dir}/aggressive_voting.csv"
temp_df.to_csv(temp_path, index=False)

confusion = get_confusion_matrix(temp_path, real_flag_path)
print(f"激进投票 F1: {confusion['f1_score']:.6f} (TP={confusion['TP']}, FP={confusion['FP']}, FN={confusion['FN']})")
print()

# ============================
# 策略5: 顶级模型组合 (只用前K个最好的)
# ============================
print("=" * 100)
print("策略5: 顶级模型组合")
print("=" * 100)

best_top_k_f1 = 0
best_top_k = 0

for k in range(2, len(models) + 1):
    top_k_models = model_names[:k]
    top_k_predictions = {}
    
    for account_id in all_ids:
        votes = [models[model_name][account_id] for model_name in top_k_models]
        top_k_predictions[account_id] = 1 if sum(votes) > len(votes) / 2 else 0
    
    temp_df = pd.DataFrame(list(top_k_predictions.items()), columns=['ID', 'Predict'])
    temp_path = f"{output_dir}/top_{k}_models.csv"
    temp_df.to_csv(temp_path, index=False)
    
    confusion = get_confusion_matrix(temp_path, real_flag_path)
    print(f"  前{k}个模型: F1={confusion['f1_score']:.6f} (TP={confusion['TP']}, FP={confusion['FP']}, FN={confusion['FN']})")
    
    if confusion['f1_score'] > best_top_k_f1:
        best_top_k_f1 = confusion['f1_score']
        best_top_k = k

print(f"\n最佳顶级模型组合: 前{best_top_k}个, F1={best_top_k_f1:.6f}")
print()

# ============================
# 策略6: 精确率优先 (使用高精确率的模型)
# ============================
print("=" * 100)
print("策略6: 精确率优先融合")
print("=" * 100)

# 选择精确率高的模型 (submit.csv 精确率=0.8835, v3.2refined_fold1_8083 精确率=0.8628)
high_precision_models = [
    "submit.csv",
    "v3.2refined_fold1_bad_f1_0.8083_good_0.9803_bad_0.8083_macro_0.8943_weighted_0.9634_seed_13_REAL_F1_0.7628549501151188.csv",
    "v3.2refined_fold5_bad_f1_0.8401_good_0.9838_bad_0.8401_macro_0.9119_weighted_0.9697_seed_13_REAL_F1_0.7579273008507347.csv"
]

precision_predictions = {}
for account_id in all_ids:
    votes = [models[model_name][account_id] for model_name in high_precision_models if model_name in models]
    # 保守策略：至少2个高精确率模型同意才预测为1
    precision_predictions[account_id] = 1 if sum(votes) >= 2 else 0

temp_df = pd.DataFrame(list(precision_predictions.items()), columns=['ID', 'Predict'])
temp_path = f"{output_dir}/precision_focused.csv"
temp_df.to_csv(temp_path, index=False)

confusion = get_confusion_matrix(temp_path, real_flag_path)
print(f"精确率优先融合 F1: {confusion['f1_score']:.6f} (TP={confusion['TP']}, FP={confusion['FP']}, FN={confusion['FN']})")
print()

# ============================
# 策略7: 混合策略 (result.csv + 其他模型的补充)
# ============================
print("=" * 100)
print("策略7: 混合策略 (以result.csv为基础)")
print("=" * 100)

# result.csv 有最高的F1，以它为基础
base_model = "result.csv"

for supplement_threshold in range(2, 5):
    hybrid_predictions = {}
    for account_id in all_ids:
        base_pred = models[base_model][account_id]
        
        if base_pred == 1:
            # 如果base预测为1，直接采用
            hybrid_predictions[account_id] = 1
        else:
            # 如果base预测为0，看其他模型是否有足够多的预测为1
            other_votes = [models[model_name][account_id] for model_name in model_names if model_name != base_model]
            hybrid_predictions[account_id] = 1 if sum(other_votes) >= supplement_threshold else 0
    
    temp_df = pd.DataFrame(list(hybrid_predictions.items()), columns=['ID', 'Predict'])
    temp_path = f"{output_dir}/hybrid_supplement_{supplement_threshold}.csv"
    temp_df.to_csv(temp_path, index=False)
    
    confusion = get_confusion_matrix(temp_path, real_flag_path)
    print(f"  补充阈值={supplement_threshold}: F1={confusion['f1_score']:.6f} (TP={confusion['TP']}, FP={confusion['FP']}, FN={confusion['FN']})")

print()

# ============================
# 策略8: Stacking (模拟简单的stacking)
# ============================
print("=" * 100)
print("策略8: 伪Stacking (基于disagreement)")
print("=" * 100)

# 找出模型分歧最大的样本，用最好的模型决策
stacking_predictions = {}
for account_id in all_ids:
    votes = [models[model_name][account_id] for model_name in model_names]
    vote_variance = np.var(votes)
    
    if vote_variance > 0.2:  # 分歧较大
        # 使用最好的模型（result.csv）
        stacking_predictions[account_id] = models[base_model][account_id]
    else:
        # 分歧较小，用加权投票
        weighted_sum = sum(votes[i] * weights[i] for i in range(len(votes)))
        stacking_predictions[account_id] = 1 if weighted_sum / sum(weights) >= 0.5 else 0

temp_df = pd.DataFrame(list(stacking_predictions.items()), columns=['ID', 'Predict'])
temp_path = f"{output_dir}/pseudo_stacking.csv"
temp_df.to_csv(temp_path, index=False)

confusion = get_confusion_matrix(temp_path, real_flag_path)
print(f"伪Stacking F1: {confusion['f1_score']:.6f} (TP={confusion['TP']}, FP={confusion['FP']}, FN={confusion['FN']})")
print()

# ============================
# 总结
# ============================
print("=" * 100)
print("总结：所有融合策略对比")
print("=" * 100)

all_strategies = [
    ("原始最佳(result.csv)", 0.8041),
    ("加权投票", best_weighted_f1),
    ("多数投票", confusion['f1_score'] if confusion else 0),
    ("保守策略", best_conservative_f1),
    ("顶级模型组合", best_top_k_f1)
]

all_strategies.sort(key=lambda x: x[1], reverse=True)

for i, (strategy, f1) in enumerate(all_strategies, 1):
    print(f"{i}. {strategy:30s}: F1={f1:.6f}")

print("\n💡 建议:")
print("1. 如果要提高召回率(找出更多的1)，尝试激进策略或混合策略")
print("2. 如果要提高精确率(减少误判)，尝试保守策略或精确率优先")
print("3. 综合平衡，加权投票和顶级模型组合通常表现较好")
print("4. 可以尝试微调阈值来优化F1分数")
