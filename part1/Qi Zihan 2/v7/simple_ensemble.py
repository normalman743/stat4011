import pandas as pd

# 读取两个模型的预测
result = pd.read_csv("/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/result.csv")
submit = pd.read_csv("/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/submit.csv")

# 融合：至少一个预测为1就是1 (OR逻辑)
result['Predict'] = (result['Predict'] | submit['Predict']).astype(int)

# 保存
result.to_csv("/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/FINAL_BEST.csv", index=False)
print(f"✓ 完成！预测为1: {result['Predict'].sum()}/{len(result)}")
