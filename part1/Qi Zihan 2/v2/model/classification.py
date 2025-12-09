import pandas as pd
import glob
import os

# 获取所有csv文件路径
csv_files = glob.glob('Qi Zihan/v2/results/high_score_predictions/*.csv')

dfs = []
for file in csv_files:
    # 获取文件名
    basename = os.path.basename(file)
    # 跳过前5个字符
    basename = basename[5:]
    # 按.分割
    parts = basename.split('.')
    # 取第一个和最后一个部分拼接
    col_name = f"{parts[0]}_{parts[-1]}"
    print(f"处理文件: {file}，列名: {col_name}")
    # 读取csv
    df = pd.read_csv(file)
    # 只保留ID和Predict
    df = df[['ID', 'Predict']]
    # 改列名
    df = df.rename(columns={'Predict': col_name})
    dfs.append(df)

# 按ID合并
result = dfs[0]
for df in dfs[1:]:
    result = pd.merge(result, df, on='ID', how='outer')

# 保存结果
result.to_csv('merged_result.csv', index=False)