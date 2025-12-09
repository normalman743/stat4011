import sys
sys.path.append('/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v5')
from simulator import calculate_f1_from_real_flags
import pandas as pd

# 输入文件
result_csv = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/result.csv"
submit_csv = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/submit.csv"

# 输出文件
output_csv = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/FINAL_BEST_ENSEMBLE.csv"

print("=" * 80)
print("最佳融合策略：至少一个预测为1")
print("=" * 80)
print()

# 加载两个模型的预测
result_df = pd.read_csv(result_csv)
submit_df = pd.read_csv(submit_csv)

print(f"✓ 加载 result.csv: {len(result_df)} 条预测")
print(f"✓ 加载 submit.csv: {len(submit_df)} 条预测")
print()

# 转换为字典以便快速查找
result_dict = dict(zip(result_df['ID'], result_df['Predict']))
submit_dict = dict(zip(submit_df['ID'], submit_df['Predict']))

# 融合策略：至少一个预测为1，就预测为1
ensemble_predictions = {}
for account_id in result_dict.keys():
    # OR逻辑：只要有一个模型预测为1，就预测为1
    ensemble_predictions[account_id] = 1 if (result_dict[account_id] == 1 or submit_dict[account_id] == 1) else 0

print("融合策略执行完成")
print(f"预测为1的数量: {sum(ensemble_predictions.values())} / {len(ensemble_predictions)}")
print()

# 保存结果
ensemble_df = pd.DataFrame(list(ensemble_predictions.items()), columns=['ID', 'Predict'])
ensemble_df.to_csv(output_csv, index=False)

print(f"✓ 融合结果已保存到: {output_csv}")
print()


print("=" * 80)
print("验证结果")
print("=" * 80)
print(f"F1分数: {f1_score:.6f}")
print()

# 对比单独模型
result_f1 = calculate_f1_from_real_flags(result_csv, real_flag_path)
submit_f1 = calculate_f1_from_real_flags(submit_csv, real_flag_path)

print("对比:")
print(f"  result.csv F1: {result_f1:.6f}")
print(f"  submit.csv F1: {submit_f1:.6f}")
print(f"  融合后 F1:     {f1_score:.6f}")
print(f"  提升:         {(f1_score - result_f1)*100:+.2f}%")
print()

print("=" * 80)
print("✅ 完成！最佳融合文件已生成")
print("=" * 80)
