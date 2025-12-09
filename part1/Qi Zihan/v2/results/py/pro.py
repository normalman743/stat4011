import pandas as pd
import numpy as np
import os
from pathlib import Path

def calculate_confusion_matrix(pred_bad, pred_good, bad_f1, true_bad=727, true_good=6831):
    """根据预测分布和F1计算混淆矩阵
    注意：bad=1, good=0
    """
    if bad_f1 == 0:
        return {'TP': 0, 'FP': pred_bad, 'FN': true_bad, 'TN': true_good, 'precision': 0, 'recall': 0}
    
    # 通过F1反推TP
    best_tp = 0
    best_f1_diff = float('inf')
    
    for tp in range(min(pred_bad, true_bad) + 1):
        if pred_bad == 0:
            precision = 0
        else:
            precision = tp / pred_bad
        
        if true_bad == 0:
            recall = 0
        else:
            recall = tp / true_bad
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        f1_diff = abs(f1 - bad_f1)
        if f1_diff < best_f1_diff:
            best_f1_diff = f1_diff
            best_tp = tp
    
    tp = best_tp
    fp = pred_bad - tp
    fn = true_bad - tp
    tn = true_good - fp
    
    precision = tp / pred_bad if pred_bad > 0 else 0
    recall = tp / true_bad if true_bad > 0 else 0
    
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'precision': precision, 'recall': recall}

def create_account_scores(models_dir=None, reference_file=None, true_bad=727, true_good=6831):
    """生成账户概率评分
    注意：在数据中 bad=1, good=0
    """
    # 默认路径
    if models_dir is None:
        models_dir = "/Users/mannormal/4011/Qi Zihan/v2/results/high_score_predictions/0.75+"
    if reference_file is None:
        reference_file = "/Users/mannormal/4011/Qi Zihan/v2/results/test_acc_predict_REAL_F1_0.17549788774894384_REAL_F1_0.0.csv"
    
    # 检查文件是否存在
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"模型目录不存在: {models_dir}")
    if not os.path.exists(reference_file):
        raise FileNotFoundError(f"参考文件不存在: {reference_file}")
    
    # 获取所有账户ID
    ref_df = pd.read_csv(reference_file)
    all_accounts = ref_df['ID'].tolist()
    
    # 初始化每个账户的bad概率列表
    account_bad_probs = {acc_id: [] for acc_id in all_accounts}
    
    # 处理每个模型文件
    csv_files = [f for f in os.listdir(models_dir) if f.endswith('.csv') and 'REAL_F1' in f]
    
    if len(csv_files) == 0:
        raise ValueError(f"在目录 {models_dir} 中没有找到符合条件的CSV文件")
    
    print(f"处理 {len(csv_files)} 个模型文件:")
    print("注意：数据中 bad=1, good=0")
    print("-" * 80)
    
    for filename in csv_files:
        filepath = os.path.join(models_dir, filename)
        df = pd.read_csv(filepath)
        
        # 修正：在数据中 bad=1, good=0
        pred_bad = len(df[df['Predict'] == 1])   # 预测为bad(1)的数量
        pred_good = len(df[df['Predict'] == 0])  # 预测为good(0)的数量
        
        # 从文件名提取F1分数
        try:
            f1_str = filename.split('REAL_F1_')[1].split('.csv')[0].split('_')[0]
            bad_f1 = float(f1_str)
        except (IndexError, ValueError) as e:
            print(f"警告: 无法从文件名 {filename} 解析F1分数，跳过")
            continue
        
        # 计算混淆矩阵
        cm = calculate_confusion_matrix(pred_bad, pred_good, bad_f1, true_bad, true_good)
        
        # 计算每个类别的准确率
        bad_precision = cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) > 0 else 0  # 预测为bad时的准确率
        good_precision = cm['TN'] / (cm['TN'] + cm['FN']) if (cm['TN'] + cm['FN']) > 0 else 0  # 预测为good时的准确率
        
        print(f"{filename}")
        print(f"  预测分布: Bad(1)={pred_bad}, Good(0)={pred_good}")
        print(f"  混淆矩阵: TP={cm['TP']}, FP={cm['FP']}, FN={cm['FN']}, TN={cm['TN']}")
        print(f"  Bad_precision={bad_precision:.6f}, Good_precision={good_precision:.6f}")
        print()
        
        # 为每个账户计算bad概率
        for _, row in df.iterrows():
            acc_id = row['ID']
            if acc_id not in account_bad_probs:
                print(f"警告: 账户 {acc_id} 不在参考文件中")
                continue
                
            if row['Predict'] == 1:  # 预测为bad(1)
                # 该账户为bad的概率 = bad的准确率
                prob_bad = bad_precision
            else:  # 预测为good(0)
                # 该账户为bad的概率 = 1 - good的准确率
                prob_bad = 1 - good_precision
            
            account_bad_probs[acc_id].append(prob_bad)
    
    # 计算每个账户的最终概率（取平均值）
    results = []
    for acc_id in all_accounts:
        if len(account_bad_probs[acc_id]) > 0:
            predict_prob = np.mean(account_bad_probs[acc_id])
        else:
            predict_prob = 0.5  # 如果没有任何模型预测，给默认值
        results.append({'ID': acc_id, 'predict': predict_prob})
    
    # 转换为DataFrame并排序
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('predict', ascending=False)
    
    # 保存结果
    result_df.to_csv('account_scores.csv', index=False, float_format='%.6f')
    
    # 统计信息
    print("=" * 80)
    print(f"生成 account_scores.csv:")
    print(f"总账户数: {len(result_df)}")
    print(f"predict=0.0的账户: {sum(result_df['predict'] == 0.0)}")
    print(f"predict>0.5的账户: {sum(result_df['predict'] > 0.5)}")
    print(f"predict>0.8的账户: {sum(result_df['predict'] > 0.8)}")
    print(f"最高概率: {result_df['predict'].max():.6f}")
    print(f"最低概率: {result_df['predict'].min():.6f}")
    
    # 显示概率分布
    unique_probs = result_df['predict'].unique()
    print(f"不同概率值的数量: {len(unique_probs)}")
    print(f"概率分布:")
    print(f"  [0.0-0.1): {sum((result_df['predict'] >= 0.0) & (result_df['predict'] < 0.1))}")
    print(f"  [0.1-0.2): {sum((result_df['predict'] >= 0.1) & (result_df['predict'] < 0.2))}")
    print(f"  [0.2-0.5): {sum((result_df['predict'] >= 0.2) & (result_df['predict'] < 0.5))}")
    print(f"  [0.5-0.8): {sum((result_df['predict'] >= 0.5) & (result_df['predict'] < 0.8))}")
    print(f"  [0.8-1.0]: {sum(result_df['predict'] >= 0.8)}")
    
    # 显示前10个高风险账户
    print("\n前10个高风险账户:")
    print(result_df.head(10).to_string(index=False))
    
    return result_df

# 使用示例
if __name__ == "__main__":
    try:
        result_df = create_account_scores()
    except Exception as e:
        print(f"错误: {e}")
        print("\n请检查文件路径是否正确，或使用自定义路径:")
        print("create_account_scores(models_dir='你的模型目录', reference_file='你的参考文件')")