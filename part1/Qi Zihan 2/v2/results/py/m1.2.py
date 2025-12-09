import pandas as pd
import numpy as np

def calculate_confusion_matrix_from_stats(pred_bad, pred_good, bad_f1, true_bad=727, true_good=6831):
    """
    根据预测统计和bad F1分数计算混淆矩阵
    """
    
    if bad_f1 == 0:
        TP = 0
    else:
        TP = bad_f1 * (true_bad + pred_bad) / 2
        TP = min(TP, true_bad, pred_bad)
        TP = int(round(TP))
    
    FN = true_bad - TP          
    FP = pred_bad - TP          
    TN = pred_good - FN         
    
    return TP, TN, FP, FN

def analyze_voting_results(csv_data, true_bad=727, true_good=6831):
    """分析投票结果的混淆矩阵"""
    results = []
    
    for _, row in csv_data.iterrows():
        filename = row['filename']
        pred_good = row['good_count'] 
        pred_bad = row['bad_count']
        bad_f1 = row['f1_score']
        
        parts = filename.replace('.csv', '').split('_')
        threshold = int(parts[1]) if len(parts) > 1 else None
        inverse = int(parts[2]) if len(parts) > 2 else None
        
        try:
            TP, TN, FP, FN = calculate_confusion_matrix_from_stats(
                pred_bad, pred_good, bad_f1, true_bad, true_good
            )
            
            precision_bad = TP / pred_bad if pred_bad > 0 else 0
            recall_bad = TP / true_bad if true_bad > 0 else 0
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
                'pred_bad': pred_bad,
                'pred_good': pred_good,
                'bad_precision': precision_bad,
                'bad_recall': recall_bad,
                'good_precision': precision_good,
                'good_recall': recall_good,
                'bad_f1_given': bad_f1,
                'bad_f1_calculated': calculated_f1,
                'accuracy': accuracy,
                'f1_diff': abs(bad_f1 - calculated_f1)
            })
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            continue
    
    return pd.DataFrame(results)

def print_confusion_matrix(TP, TN, FP, FN):
    """打印混淆矩阵"""
    print(f"    混淆矩阵:        预测")
    print(f"              Bad    Good")
    print(f"    真实 Bad  {TP:4d}   {FN:4d}")
    print(f"        Good  {FP:4d}   {TN:4d}")

def print_threshold_comparison(threshold_group):
    """打印同一阈值下两个方向的对比"""
    print("=" * 80)
    print(f"阈值 {threshold_group['threshold'].iloc[0]} 的对比分析")
    print("=" * 80)
    
    # 按inverse排序，确保0在前1在后
    threshold_group = threshold_group.sort_values('inverse')
    
    for _, row in threshold_group.iterrows():
        direction = "正向 (≤threshold → bad)" if row['inverse'] == 0 else "逆向 (≤threshold → good)"
        print(f"\n【{direction}】 - {row['filename']}")
        print(f"预测分布: Bad={row['pred_bad']}, Good={row['pred_good']}")
        print_confusion_matrix(row['TP'], row['TN'], row['FP'], row['FN'])
        
        print(f"指标:")
        print(f"  Bad类  - 精确率: {row['bad_precision']:.4f}, 召回率: {row['bad_recall']:.4f}, F1: {row['bad_f1_given']:.4f}")
        print(f"  Good类 - 精确率: {row['good_precision']:.4f}, 召回率: {row['good_recall']:.4f}")
        print(f"  总体准确率: {row['accuracy']:.4f}")
        print(f"  F1验证差异: {row['f1_diff']:.6f}")
    
    # 对比总结
    if len(threshold_group) == 2:
        normal = threshold_group[threshold_group['inverse'] == 0].iloc[0]
        inverse = threshold_group[threshold_group['inverse'] == 1].iloc[0]
        
        print(f"\n【对比总结】")
        print(f"  正向 vs 逆向:")
        print(f"    Bad F1:  {normal['bad_f1_given']:.4f} vs {inverse['bad_f1_given']:.4f}")
        print(f"    准确率:  {normal['accuracy']:.4f} vs {inverse['accuracy']:.4f}")
        print(f"    更佳选择: {'正向' if normal['bad_f1_given'] > inverse['bad_f1_given'] else '逆向'} (基于Bad F1)")

# 使用示例
if __name__ == "__main__":
    csv_file_path = "/Users/mannormal/4011/Qi Zihan/v2/results/py/file_analysis_results.csv"
    
    try:
        df = pd.read_csv(csv_file_path)
        print(f"读取到 {len(df)} 个文件的结果")
        print(f"真实分布: Bad={727}, Good={6831}, 总计={7558}")
        
        results_df = analyze_voting_results(df)
        
        if len(results_df) > 0:
            # 按阈值分组显示
            print("\n" + "="*80)
            print("所有阈值的投票策略对比分析")
            print("="*80)
            
            # 按阈值分组
            for threshold in sorted(results_df['threshold'].unique()):
                threshold_group = results_df[results_df['threshold'] == threshold]
                print_threshold_comparison(threshold_group)
            
            # 保存完整结果
            output_path = "/Users/mannormal/4011/Qi Zihan/v2/results/py/confusion_matrix_analysis.csv"
            results_df.to_csv(output_path, index=False)
            print(f"\n完整分析结果已保存到: {output_path}")
            
            # 全局最佳结果
            print("\n" + "="*80)
            print("全局最佳结果总结")
            print("="*80)
            
            best_f1_row = results_df.loc[results_df['bad_f1_given'].idxmax()]
            best_acc_row = results_df.loc[results_df['accuracy'].idxmax()]
            
            print(f"最佳Bad F1: {best_f1_row['bad_f1_given']:.4f}")
            print(f"  文件: {best_f1_row['filename']}")
            print(f"  阈值: {best_f1_row['threshold']}, 方向: {'正向' if best_f1_row['inverse']==0 else '逆向'}")
            print(f"  准确率: {best_f1_row['accuracy']:.4f}")
            
            print(f"\n最佳准确率: {best_acc_row['accuracy']:.4f}")
            print(f"  文件: {best_acc_row['filename']}")
            print(f"  阈值: {best_acc_row['threshold']}, 方向: {'正向' if best_acc_row['inverse']==0 else '逆向'}")
            print(f"  Bad F1: {best_acc_row['bad_f1_given']:.4f}")
            
            print(f"\n数据统计:")
            print(f"  平均Bad F1: {results_df['bad_f1_given'].mean():.4f}")
            print(f"  平均准确率: {results_df['accuracy'].mean():.4f}")
            print(f"  处理成功: {len(results_df)}/{len(df)} 个文件")
            
        else:
            print("没有成功处理任何文件")
            
    except FileNotFoundError:
        print(f"找不到文件: {csv_file_path}")
    except Exception as e:
        print(f"读取文件时出错: {e}")