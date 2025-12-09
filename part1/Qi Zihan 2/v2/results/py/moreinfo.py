import pandas as pd  # 添加 pandas 库
import os

def count_labels(csv_file_path):
    df = pd.read_csv(csv_file_path)
    good_count = (df['Predict'] == 1).sum()  # 假设标签列名为 'label'
    bad_count = (df['Predict'] == 0).sum()
    return good_count, bad_count

# 使用
if __name__ == "__main__":
    path = "/Users/mannormal/4011/Qi Zihan/v2/results/py/votes1/"
    
    # 获取目录中所有的 CSV 文件
    all_files = [f for f in os.listdir(path) if f.endswith('.csv') and f.startswith('vote_')]
    
    # 存储结果的列表
    results = []
    
    for filename in all_files:
        csv_file = os.path.join(path, filename)
        
        # 从文件名提取 F1 分数
        f1_score = None
        if "_REAL_F1_" in filename:
            try:
                f1_score = float(filename.split("_REAL_F1_")[-1].split(".csv")[0])
            except ValueError:
                print(f"无法解析 F1 分数：{filename}")
                continue
        
        # 计算每个文件中的 1 和 0 的数量
        try:
            good_count, bad_count = count_labels(csv_file)
            
            # 添加到结果列表
            results.append({
                'filename': filename,
                'good_count': good_count,
                'bad_count': bad_count,
                'f1_score': f1_score,
                'total_count': good_count + bad_count
            })
            
            print(f"File: {filename}, Good: {good_count}, Bad: {bad_count}, F1 Score: {f1_score}")
        except Exception as e:
            print(f"处理文件 {filename} 时出错：{e}")
    
    # 将结果保存到CSV文件
    if results:
        results_df = pd.DataFrame(results)
        # 按照filename排序
        results_df = results_df.sort_values('filename')
        output_path = "/Users/mannormal/4011/Qi Zihan/v2/results/py/file_analysis_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n结果已保存到: {output_path}")
        print(f"共处理 {len(results)} 个文件")
    else:
        print("没有找到有效的文件结果")