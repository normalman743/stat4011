import pandas as pd
import os
import json
from pathlib import Path
import re  # 新增导入
import numpy as np  # 新增导入

def process_results_to_csv():
    # Define paths
    results_path = "/Users/mannormal/4011/Qi Zihan/v2/results/"
    json_output_path = "/Users/mannormal/4011/Qi Zihan/v2/results/json/"
    
    # Ensure json directory exists
    os.makedirs(json_output_path, exist_ok=True)
    
    # Get all CSV files in results directory
    csv_files = [f for f in os.listdir(results_path) if f.endswith('.csv') and not f.startswith('.')]
    
    results_data = []
    
    for csv_file in csv_files:
        try:
            # Read CSV file
            df = pd.read_csv(os.path.join(results_path, csv_file))
            
            # Extract F1 score from filename if present
            f1_score = None
            if 'F1_' in csv_file:
                # 提取所有F1分数，取最后一个
                f1_matches = re.findall(r'F1_(\d+\.\d+)', csv_file)
                if f1_matches:
                    f1_score = float(f1_matches[-1])  # 取最后一个F1分数
            
            # 确定预测列
            if 'Predict' in df.columns:
                predict_col = df['Predict']
            elif len(df.columns) > 0:
                predict_col = df.iloc[:, -1]  # 使用最后一列
            else:
                predict_col = pd.Series()  # 空Series
            
            # Get basic statistics
            stats = {
                'filename': csv_file,
                'f1_score': f1_score,
                'total_rows': len(df),
                'columns': list(df.columns) if not df.empty else [],
                'good_count': len(df[predict_col == 0]) if len(predict_col) > 0 else 0,
                'bad_count': len(df[predict_col == 1]) if len(predict_col) > 0 else 0
            }
            
            # 添加更多有用信息
            stats['bad_ratio'] = stats['bad_count'] / stats['total_rows'] if stats['total_rows'] > 0 else 0
            
            results_data.append(stats)
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results_data)
    
    # Sort by F1 score if available
    if 'f1_score' in summary_df.columns:
        summary_df = summary_df.sort_values('f1_score', ascending=False, na_position='last')
    
    # Save to CSV in json directory
    output_csv = os.path.join(json_output_path, 'results_summary.csv')
    summary_df.to_csv(output_csv, index=False)
    
    # Also save as JSON for easier processing
    output_json = os.path.join(json_output_path, 'results_summary.json')
    summary_df.to_json(output_json, orient='records', indent=2)
    
    print(f"Processed {len(csv_files)} CSV files")
    print(f"Results saved to:")
    print(f"  CSV: {output_csv}")
    print(f"  JSON: {output_json}")
    
    # 打印一些统计信息
    print(f"\nF1 Score Statistics:")
    print(f"  Average F1: {summary_df['f1_score'].mean():.4f}")
    print(f"  Best F1: {summary_df['f1_score'].max():.4f}")
    print(f"  Models with F1 > 0.75: {len(summary_df[summary_df['f1_score'] > 0.75])}")
    
    return summary_df

def create_voting_matrix():
    predictions_path = "/Users/mannormal/4011/Qi Zihan/v2/results/high_score_predictions/"
    
    # 读取所有CSV文件
    all_predictions = {}
    for csv_file in os.listdir(predictions_path):
        if csv_file.endswith('.csv'):
            df = pd.read_csv(os.path.join(predictions_path, csv_file))
            # 假设ID列和Predict列
            all_predictions[csv_file] = df.set_index('ID')['Predict']
    
    # 创建投票矩阵
    voting_df = pd.DataFrame(all_predictions)
    voting_df['bad_votes'] = voting_df.sum(axis=1)
    
    # 确保ID列对齐
    voting_df.reset_index(inplace=True)
    voting_df.rename(columns={'index': 'ID'}, inplace=True)
    
    # 计算投票比例
    voting_df['vote_ratio'] = voting_df['bad_votes'] / len(all_predictions)
    
    # 投票分布统计
    vote_distribution = voting_df['bad_votes'].value_counts().sort_index()
    
    # 找出分歧最大的样本
    voting_df['vote_std'] = voting_df.iloc[:, 1:-1].std(axis=1)  # 只计算预测列的标准差
    
    return voting_df, vote_distribution

def analyze_disagreements(voting_df):
    # 分层分析
    layers = {
        'unanimous_bad': voting_df[voting_df['bad_votes'] == 36],  # 全票Bad
        'high_conf_bad': voting_df[voting_df['bad_votes'] >= 30],   # 高置信Bad
        'disputed': voting_df[voting_df['bad_votes'].between(15, 25)],  # 争议区
        'high_conf_good': voting_df[voting_df['bad_votes'] <= 5],   # 高置信Good
        'unanimous_good': voting_df[voting_df['bad_votes'] == 0]    # 全票Good
    }
    
    print("样本分层统计：")
    for name, group in layers.items():
        print(f"{name}: {len(group)}个样本")
    
    return layers

def analyze_model_correlation(voting_df):
    # 计算模型间的相关性
    model_cols = [col for col in voting_df.columns if col.endswith('.csv')]
    correlation_matrix = voting_df[model_cols].corr()
    
    # 找出独立性强的模型
    avg_correlation = correlation_matrix.mean()
    independent_models = avg_correlation[avg_correlation < 0.7].index
    
    print(f"独立性强的模型: {list(independent_models)[:5]}")
    
    return correlation_matrix

def generate_probing_strategy(voting_df, vote_distribution):
    # 找出关键投票阈值
    cumsum = vote_distribution.cumsum()
    
    # 找到累计约755个Bad的投票阈值
    threshold_755 = None
    for votes, count in cumsum.items():
        if count >= 755:
            threshold_755 = votes
            break
    
    print(f"预测755个Bad的投票阈值约为: {threshold_755}票")
    
    # 生成探测序列
    probing_samples = {
        f'vote_{i}': voting_df[voting_df['bad_votes'] == i].index.tolist()
        for i in range(threshold_755-2, threshold_755+3)
    }
    
    return probing_samples, threshold_755

def create_submission_plan(voting_df, threshold):
    submissions = []
    
    # 第1-5次：阈值探测
    for t in [36, 30, 25, 20, threshold]:
        pred = (voting_df['bad_votes'] >= t).astype(int)
        submissions.append({
            'name': f'threshold_{t}',
            'description': f'投票数>={t}的预测为Bad',
            'predictions': pred
        })
    
    # 第6-10次：反转探测
    for votes in [threshold-1, threshold, threshold+1]:
        samples = voting_df[voting_df['bad_votes'] == votes].index
        # 反转这些样本
        base_pred = (voting_df['bad_votes'] >= threshold).astype(int)
        flip_pred = base_pred.copy()
        flip_pred[samples] = 1 - flip_pred[samples]
        
        submissions.append({
            'name': f'flip_vote_{votes}',
            'description': f'反转{votes}票的样本',
            'predictions': flip_pred
        })
    
    return submissions

if __name__ == "__main__":
    summary = process_results_to_csv()
    print("\nTop 10 results by F1 score:")
    print(summary.head(10)[['filename', 'f1_score', 'bad_count', 'bad_ratio']])
    
    # 调用投票矩阵分析
    voting_df, vote_distribution = create_voting_matrix()
    
    # 分析样本分歧
    layers = analyze_disagreements(voting_df)
    
    # 分析模型相关性
    correlation_matrix = analyze_model_correlation(voting_df)
    
    # 生成探测策略
    probing_samples, threshold_755 = generate_probing_strategy(voting_df, vote_distribution)
    
    # 创建提交计划
    submissions = create_submission_plan(voting_df, threshold_755)
    
    print("\nGenerated submissions:")
    for submission in submissions:
        print(submission['name'], submission['description'])