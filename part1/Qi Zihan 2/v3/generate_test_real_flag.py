import pandas as pd
import numpy as np

df = pd.read_csv('/Users/mannormal/4011/account_scores.csv')
print(f"总账户数: {len(df)}")

# 按predict分三层
high_good_df = df[df['predict'] < 0.1].copy()
mid_df = df[(df['predict'] >= 0.1) & (df['predict'] <= 0.8)].copy()
high_bad_df = df[df['predict'] > 0.8].copy()

# 分配真实标签
def assign_RealFlag(df, real_good, real_bad, seed=42):
    df = df.copy()
    df['RealFlag'] = 0
    np.random.seed(seed)
    bad_indices = np.random.choice(len(df), real_bad, replace=False)
    df.iloc[bad_indices, df.columns.get_loc('RealFlag')] = 1
    return df

high_good_df = assign_RealFlag(high_good_df, real_good=6626, real_bad=154)
mid_df = assign_RealFlag(mid_df, real_good=168, real_bad=124)
high_bad_df = assign_RealFlag(high_bad_df, real_good=37, real_bad=449)

# 验证每一层的good/bad数量
print("\n各层分布：")
for name, part_df in zip(
    ['high_good', 'mid', 'high_bad'],
    [high_good_df, mid_df, high_bad_df]
):
    good_count = (part_df['RealFlag'] == 0).sum()
    bad_count = (part_df['RealFlag'] == 1).sum()
    print(f"{name}: Good账户数={good_count}, Bad账户数={bad_count}, 总数={len(part_df)}")

# 合并
result_df = pd.concat([high_good_df, mid_df, high_bad_df], ignore_index=True)
result_df = result_df[['ID', 'RealFlag']]
result_df.columns = ['ID', 'RealFlag']

output_path = '/Users/mannormal/4011/Qi Zihan/v3/test_real_flag.csv'
result_df.to_csv(output_path, index=False)
print(f"\n结果已保存到: {output_path}")

# 验证总数
print(f"输出文件包含 {len(result_df)} 个账户")
print(f"Bad账户数: {(result_df['RealFlag'] == 1).sum()}")
print(f"Good账户数: {(result_df['RealFlag'] == 0).sum()}")
