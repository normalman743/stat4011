import sys
sys.path.append('/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v5')
from simulator import get_confusion_matrix, calculate_f1_from_real_flags
import pandas as pd
import numpy as np
from pathlib import Path

# 文件路径列表
file_paths = [
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/result.csv",  # F1: 0.8041 (基线)
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/submit.csv",  # F1: 0.7803
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold1_bad_f1_0.8083_good_0.9803_bad_0.8083_macro_0.8943_weighted_0.9634_seed_13_REAL_F1_0.7628549501151188.csv",  # F1: 0.7629
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/GRADIENT_TUNE_10PCT_REAL_F1_0.7611336032388665.csv",  # F1: 0.7611
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold5_bad_f1_0.8401_good_0.9838_bad_0.8401_macro_0.9119_weighted_0.9697_seed_13_REAL_F1_0.7579273008507347.csv",  # F1: 0.7579
]

real_flag_path = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/融合二分模型_最终版 copy.csv"
output_dir = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7"

print("=" * 100)
print("基于 result.csv 的加权融合策略")
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
model_names = list(models.keys())
base_model = "result.csv"

print(f"基线模型: {base_model}")
print(f"辅助模型数量: {len(models) - 1}")
print(f"总样本数: {len(all_ids)}")
print()

# 先验证基线模型的F1
baseline_f1 = calculate_f1_from_real_flags(file_paths[0], real_flag_path)
print(f"基线模型 F1: {baseline_f1:.6f}")
print()

# ============================
# 策略1: result.csv权重 + 其他模型加权补充
# ============================
print("=" * 100)
print("策略1: 基线权重 + 其他模型加权补充")
print("=" * 100)

best_f1 = baseline_f1
best_strategy = "baseline"
best_weights = None

# 尝试不同的权重组合
for base_weight in [0.5, 0.6, 0.7, 0.8, 0.9]:
    # 其他模型的权重按F1分数分配
    other_weights = [0.7803, 0.7629, 0.7611, 0.7579]
    other_weights = [w / sum(other_weights) * (1 - base_weight) for w in other_weights]
    
    # 尝试不同的阈值
    for threshold in np.arange(0.35, 0.65, 0.05):
        weighted_predictions = {}
        
        for account_id in all_ids:
            # 基线模型的投票
            weighted_sum = models[base_model][account_id] * base_weight
            
            # 其他模型的加权投票
            for i, model_name in enumerate(model_names[1:]):
                weighted_sum += models[model_name][account_id] * other_weights[i]
            
            weighted_predictions[account_id] = 1 if weighted_sum >= threshold else 0
        
        # 保存并评估
        temp_df = pd.DataFrame(list(weighted_predictions.items()), columns=['ID', 'Predict'])
        temp_path = f"{output_dir}/temp_weighted_base_{base_weight}_threshold_{threshold:.2f}.csv"
        temp_df.to_csv(temp_path, index=False)
        
        f1 = calculate_f1_from_real_flags(temp_path, real_flag_path)
        
        if f1 > best_f1:
            best_f1 = f1
            best_strategy = f"base_weight={base_weight:.1f}, threshold={threshold:.2f}"
            best_weights = (base_weight, other_weights, threshold)
            print(f"  ✨ 新最佳: base_weight={base_weight:.1f}, threshold={threshold:.2f}, F1={f1:.6f}")

print(f"\n最佳策略1: {best_strategy}, F1={best_f1:.6f}")
print()

# ============================
# 策略2: 智能补充 (base预测0时，其他模型加权投票)
# ============================
print("=" * 100)
print("策略2: 智能补充策略")
print("=" * 100)

best_smart_f1 = baseline_f1
best_smart_threshold = 0

for other_threshold in np.arange(0.3, 0.8, 0.05):
    smart_predictions = {}
    
    for account_id in all_ids:
        base_pred = models[base_model][account_id]
        
        if base_pred == 1:
            # 基线预测为1，直接采用
            smart_predictions[account_id] = 1
        else:
            # 基线预测为0，看其他模型的加权投票
            other_weights = [0.7803, 0.7629, 0.7611, 0.7579]
            weighted_sum = sum(models[model_names[i+1]][account_id] * other_weights[i] 
                             for i in range(len(other_weights)))
            weighted_avg = weighted_sum / sum(other_weights)
            
            smart_predictions[account_id] = 1 if weighted_avg >= other_threshold else 0
    
    # 保存并评估
    temp_df = pd.DataFrame(list(smart_predictions.items()), columns=['ID', 'Predict'])
    temp_path = f"{output_dir}/temp_smart_supplement_{other_threshold:.2f}.csv"
    temp_df.to_csv(temp_path, index=False)
    
    f1 = calculate_f1_from_real_flags(temp_path, real_flag_path)
    
    if f1 > best_smart_f1:
        best_smart_f1 = f1
        best_smart_threshold = other_threshold
        print(f"  ✨ 补充阈值={other_threshold:.2f}, F1={f1:.6f}")

print(f"\n最佳智能补充: threshold={best_smart_threshold:.2f}, F1={best_smart_f1:.6f}")
print()

# ============================
# 策略3: 自适应权重 (根据样本一致性调整)
# ============================
print("=" * 100)
print("策略3: 自适应权重策略")
print("=" * 100)

best_adaptive_f1 = baseline_f1
best_adaptive_params = None

for agreement_threshold in [0.6, 0.7, 0.8, 0.9]:
    for vote_threshold in np.arange(0.4, 0.7, 0.1):
        adaptive_predictions = {}
        
        for account_id in all_ids:
            # 计算所有模型的投票
            all_votes = [models[model_name][account_id] for model_name in model_names]
            vote_mean = np.mean(all_votes)
            vote_std = np.std(all_votes)
            
            # 如果模型一致性高，增加基线权重
            if vote_std < (1 - agreement_threshold):
                # 高一致性，相信基线
                adaptive_predictions[account_id] = models[base_model][account_id]
            else:
                # 低一致性，用加权投票
                weights = [0.8041, 0.7803, 0.7629, 0.7611, 0.7579]
                weighted_sum = sum(all_votes[i] * weights[i] for i in range(len(weights)))
                weighted_avg = weighted_sum / sum(weights)
                adaptive_predictions[account_id] = 1 if weighted_avg >= vote_threshold else 0
        
        # 保存并评估
        temp_df = pd.DataFrame(list(adaptive_predictions.items()), columns=['ID', 'Predict'])
        temp_path = f"{output_dir}/temp_adaptive_{agreement_threshold}_{vote_threshold:.1f}.csv"
        temp_df.to_csv(temp_path, index=False)
        
        f1 = calculate_f1_from_real_flags(temp_path, real_flag_path)
        
        if f1 > best_adaptive_f1:
            best_adaptive_f1 = f1
            best_adaptive_params = (agreement_threshold, vote_threshold)
            print(f"  ✨ agreement={agreement_threshold}, vote_threshold={vote_threshold:.1f}, F1={f1:.6f}")

if best_adaptive_params:
    print(f"\n最佳自适应策略: agreement={best_adaptive_params[0]}, vote_threshold={best_adaptive_params[1]:.1f}, F1={best_adaptive_f1:.6f}")
print()

# ============================
# 策略4: 渐进式融合 (逐步添加模型)
# ============================
print("=" * 100)
print("策略4: 渐进式融合")
print("=" * 100)

best_progressive_f1 = baseline_f1
best_progressive_combo = [base_model]

current_combo = [base_model]
remaining_models = model_names[1:]

while remaining_models:
    best_add_f1 = best_progressive_f1
    best_add_model = None
    
    for candidate in remaining_models:
        test_combo = current_combo + [candidate]
        
        # 用这个组合做加权投票
        for threshold in np.arange(0.4, 0.6, 0.1):
            combo_predictions = {}
            
            for account_id in all_ids:
                votes = [models[m][account_id] for m in test_combo]
                combo_predictions[account_id] = 1 if np.mean(votes) >= threshold else 0
            
            temp_df = pd.DataFrame(list(combo_predictions.items()), columns=['ID', 'Predict'])
            temp_path = f"{output_dir}/temp_progressive.csv"
            temp_df.to_csv(temp_path, index=False)
            
            f1 = calculate_f1_from_real_flags(temp_path, real_flag_path)
            
            if f1 > best_add_f1:
                best_add_f1 = f1
                best_add_model = candidate
    
    if best_add_model and best_add_f1 > best_progressive_f1:
        current_combo.append(best_add_model)
        remaining_models.remove(best_add_model)
        best_progressive_f1 = best_add_f1
        best_progressive_combo = current_combo.copy()
        print(f"  ✨ 添加 {best_add_model[:40]}..., F1={best_add_f1:.6f}")
    else:
        break

print(f"\n最佳渐进组合: {len(best_progressive_combo)} 个模型, F1={best_progressive_f1:.6f}")
print()

# ============================
# 总结
# ============================
print("=" * 100)
print("所有策略总结")
print("=" * 100)

strategies = [
    ("基线 (result.csv)", baseline_f1),
    ("策略1: 基线权重+补充", best_f1),
    ("策略2: 智能补充", best_smart_f1),
    ("策略3: 自适应权重", best_adaptive_f1),
    ("策略4: 渐进式融合", best_progressive_f1)
]

strategies.sort(key=lambda x: x[1], reverse=True)

for i, (name, f1) in enumerate(strategies, 1):
    improvement = (f1 - baseline_f1) * 100
    print(f"{i}. {name:30s}: F1={f1:.6f} (提升: {improvement:+.2f}%)")

print("\n" + "=" * 100)
print("最终建议")
print("=" * 100)
print(f"最佳策略: {strategies[0][0]}")
print(f"最佳F1分数: {strategies[0][1]:.6f}")
print(f"相比基线提升: {(strategies[0][1] - baseline_f1)*100:.2f}%")

# 保存最佳结果
if strategies[0][1] > baseline_f1:
    print(f"\n✅ 找到了比基线更好的融合策略!")
else:
    print(f"\n⚠️ 基线模型 result.csv 已经是最优的")
