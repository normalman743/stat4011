import pandas as pd
import numpy as np

def calculate_confusion_matrix_from_stats(pred_bad, pred_good, bad_f1, true_bad=727, true_good=6832):
    """
    根据预测统计和bad F1分数计算混淆矩阵
    """
    
    if bad_f1 == 0:
        TP = 0
        FN = true_bad
        FP = pred_bad
        TN = true_good - FP
    else:
        TP = int(bad_f1 * (true_bad + pred_bad) / 2)
        TP = min(TP, pred_bad, true_bad)
    
    FN = true_bad - TP        
    FP = pred_bad - TP        
    TN = true_good - FP       
    
    return TP, TN, FP, FN

def analyze_voting_results(csv_data, true_bad=727, true_good=6832):
    """
    分析投票结果的混淆矩阵
    """
    results = []
    
    for _, row in csv_data.iterrows():
        filename = row['filename']
        pred_good = row['good_count'] 
        pred_bad = row['bad_count']
        bad_f1 = row['f1_score']
        
        parts = filename.replace('.csv', '').split('_')
        threshold = int(parts[1]) if len(parts) > 1 else None
        inverse = int(parts[2]) if len(parts) > 2 else None
        
        TP, TN, FP, FN = calculate_confusion_matrix_from_stats(
            pred_bad, pred_good, bad_f1, true_bad, true_good
        )
        
        # 修正计算公式
        precision_bad = TP / pred_bad if pred_bad > 0 else 0
        recall_bad = TP / true_bad if true_bad > 0 else 0
        
        # Good类精确率 = TN / (预测为Good的总数)
        # 预测为Good的总数 = pred_good = TN + FN
        precision_good = TN / pred_good if pred_good > 0 else 0
        recall_good = TN / true_good if true_good > 0 else 0
        
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        calculated_f1 = (2 * precision_bad * recall_bad / (precision_bad + recall_bad) 
                        if (precision_bad + recall_bad) > 0 else 0)
        
        results.append({
            'filename': filename,
            'threshold': threshold,
            'inverse': inverse,
            'TP': TP,
            'TN': TN,
            'FP': FP, 
            'FN': FN,
            'bad_precision': precision_bad,
            'bad_recall': recall_bad,
            'good_precision': precision_good,
            'good_recall': recall_good,
            'bad_f1_given': bad_f1,
            'bad_f1_calculated': calculated_f1,
            'accuracy': accuracy,
            'f1_diff': abs(bad_f1 - calculated_f1)
        })
    
    return pd.DataFrame(results)

def print_confusion_matrix(TP, TN, FP, FN):
    """打印混淆矩阵"""
    print("混淆矩阵:")
    print(f"              预测")
    print(f"           Bad    Good")
    print(f"真实 Bad   {TP:4d}   {FN:4d}")
    print(f"    Good   {FP:4d}   {TN:4d}")
    print()

# 使用示例
if __name__ == "__main__":
    csv_file_path = "/Users/mannormal/4011/Qi Zihan/v2/results/py/file_analysis_results.csv"
    
    try:
        df = pd.read_csv(csv_file_path)
        results_df = analyze_voting_results(df)
        
        top_f1_results = results_df.nlargest(len(results_df), 'bad_f1_given')
        for _, row in top_f1_results.iterrows():
            print(f"文件: {row['filename']}")
            print(f"阈值: {row['threshold']}, 逆向: {row['inverse']}")
            print_confusion_matrix(row['TP'], row['TN'], row['FP'], row['FN'])
            print(f"Bad类 - 精确率: {row['bad_precision']:.4f}, 召回率: {row['bad_recall']:.4f}, F1: {row['bad_f1_given']:.4f}")
            print(f"Good类 - 精确率: {row['good_precision']:.4f}, 召回率: {row['good_recall']:.4f}")
            print(f"总体准确率: {row['accuracy']:.4f}")
            print(f"F1验证差异: {row['f1_diff']:.6f}")
            print("-" * 60)
            
    except FileNotFoundError:
        print(f"找不到文件: {csv_file_path}")
    except Exception as e:
        print(f"读取文件时出错: {e}")