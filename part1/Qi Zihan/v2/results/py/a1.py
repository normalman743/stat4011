import pandas as pd
import numpy as np

def analyze_bad_groups_corrected(merged_file_path, confusion_results_path):
    """
    修正版本：通过逆向投票结果分析每个分数区间的bad/good分布
    """
    
    # 读取原始预测数据
    print("读取原始预测数据...")
    original_data = pd.read_csv(merged_file_path)
    print(f"总样本数: {len(original_data)}")
    
    # 读取混淆矩阵分析结果
    print("读取混淆矩阵分析结果...")
    confusion_results = pd.read_csv(confusion_results_path)
    
    # 只分析逆向投票结果（inverse=1）
    inverse_results = confusion_results[confusion_results['inverse'] == 1].copy()
    inverse_results = inverse_results.sort_values('threshold').reset_index(drop=True)  # 重置索引
    
    print("\n=== 逆向投票结果分析 ===")
    print("阈值\t预测Bad\tTP\tTP变化\t预测Bad变化")
    print("-" * 60)
    
    # 分析每个阈值变化
    groups = []
    
    # 在groups开头添加分数=0的分析
    if len(inverse_results) > 0:
        first_row = inverse_results.iloc[0]
        if first_row['threshold'] == 0:
            # 分数=0的样本在阈值0逆向投票中被预测为good
            score_0_samples = len(original_data[original_data['Predict'] == 0])
            score_0_bad = first_row['TP']  # 这些good预测中的bad数量
            score_0_good = score_0_samples - score_0_bad
            
            groups.insert(0, {
                'score_range': 'score=0',
                'total_samples': score_0_samples,
                'bad_count': score_0_bad,
                'good_count': score_0_good,
                'bad_ratio': score_0_bad / score_0_samples if score_0_samples > 0 else 0,
                'threshold_start': 0,
                'threshold_end': 0
            })
    
    prev_pred_bad = None
    prev_tp = None
    
    for idx in range(len(inverse_results)):  # 使用位置索引而不是原始索引
        row = inverse_results.iloc[idx]
        threshold = row['threshold']
        pred_bad = row['pred_bad']
        tp = row['TP']
        
        if prev_pred_bad is not None:
            tp_change = tp - prev_tp
            pred_bad_change = pred_bad - prev_pred_bad
            
            print(f"{threshold:2d}\t{pred_bad:4d}\t{tp:3d}\t{tp_change:+3d}\t{pred_bad_change:+4d}")
            
            # 分析被移除的样本
            if pred_bad_change < 0:  # 预测Bad减少了
                removed_samples = -pred_bad_change
                bad_in_removed = -tp_change if tp_change < 0 else 0  # TP减少意味着移除了bad
                good_in_removed = removed_samples - bad_in_removed
                
                if removed_samples > 0:
                    # 这些样本的分数范围是 (prev_threshold, current_threshold]
                    prev_threshold = inverse_results.iloc[idx-1]['threshold'] if idx > 0 else -1
                    
                    groups.append({
                        'score_range': f"({prev_threshold}, {threshold}]" if prev_threshold >= 0 else f"≤{threshold}",
                        'total_samples': removed_samples,
                        'bad_count': bad_in_removed,
                        'good_count': good_in_removed,
                        'bad_ratio': bad_in_removed / removed_samples,
                        'threshold_start': prev_threshold,
                        'threshold_end': threshold
                    })
        else:
            print(f"{threshold:2d}\t{pred_bad:4d}\t{tp:3d}\t --\t   --")
            
        prev_pred_bad = pred_bad
        prev_tp = tp
    
    # 添加分数最高的"绝对good"区域（基于阈值0正向投票结果）
    # 从confusion_results获取阈值0正向投票的good预测数
    threshold_0_forward = confusion_results[(confusion_results['threshold'] == 0) & 
                                          (confusion_results['inverse'] == 0)].iloc[0]
    absolute_good_count = threshold_0_forward['pred_good']
    
    groups.insert(0, {
        'score_range': '绝对Good区域',
        'total_samples': absolute_good_count,
        'bad_count': 0,
        'good_count': absolute_good_count,
        'bad_ratio': 0.0,
        'threshold_start': 999,  # 最高分数区域
        'threshold_end': 999
    })
    
    # 按分数从高到低排序显示
    groups_sorted = sorted(groups, key=lambda x: x['threshold_end'], reverse=True)
    
    print(f"\n=== 分数区间分析（按分数从高到低）===")
    print("分数区间\t\t总样本\tBad数\tGood数\tBad比例")
    print("-" * 65)
    
    total_analyzed_samples = 0
    total_bad = 0
    total_good = 0
    
    for group in groups_sorted:
        total_analyzed_samples += group['total_samples']
        total_bad += group['bad_count']
        total_good += group['good_count']
        
        print(f"{group['score_range']:15}\t{group['total_samples']:4d}\t{group['bad_count']:3d}\t{group['good_count']:4d}\t{group['bad_ratio']:6.2%}")
    
    print(f"\n总计分析: {total_analyzed_samples} 样本, Bad={total_bad}, Good={total_good}")
    
    # 统计每个具体分数的样本数
    print(f"\n=== 原始分数分布 ===")
    score_counts = original_data['Predict'].value_counts().sort_index()
    print("分数\t样本数")
    print("-" * 20)
    for score, count in score_counts.items():
        print(f"{score:2d}\t{count:4d}")
    
    # 识别确定的good区域
    print(f"\n=== 确定是Good的区域 ===")
    definitely_good_groups = [g for g in groups_sorted if g['bad_count'] == 0]
    total_definitely_good = sum(g['total_samples'] for g in definitely_good_groups)
    
    for group in definitely_good_groups:
        print(f"{group['score_range']}: {group['total_samples']}个样本, Bad比例=0%")
    
    print(f"总共 {total_definitely_good} 个样本确认为Good")
    
    # 识别高风险区域
    print(f"\n=== Bad比例最高的区域（>5%）===")
    high_risk_groups = [g for g in groups_sorted if g['bad_ratio'] > 0.05 and g['bad_count'] > 0]
    
    for group in high_risk_groups:
        print(f"{group['score_range']}: {group['bad_count']}/{group['total_samples']} = {group['bad_ratio']:.2%}")
    
    # 生成按分数范围的详细映射
    generate_score_mapping(groups_sorted, original_data, merged_file_path)
    
    # 保存结果
    results_df = pd.DataFrame(groups_sorted)
    output_file = merged_file_path.replace('.csv', '_corrected_group_analysis.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n分组结果已保存到: {output_file}")
    
    return groups_sorted

def generate_score_mapping(groups, original_data, merged_file_path):
    """
    为每个样本生成基于分数区间的标签
    """
    print(f"\n=== 生成样本标签映射 ===")
    
    # 为每个样本分配到对应的分数区间
    sample_labels = []
    
    for _, row in original_data.iterrows():
        sample_id = row['ID']
        score = row['Predict']
        
        # 找到这个分数属于哪个区间
        assigned_group = None
        
        # 特殊处理绝对good区域（最高的1665个分数）
        top_1665_threshold = original_data.nlargest(1665, 'Predict')['Predict'].min()
        if score >= top_1665_threshold:
            assigned_group = {
                'range': '绝对Good区域',
                'bad_ratio': 0.0,
                'label': 'definitely_good'
            }
        else:
            # 查找对应的分数区间
            for group in groups:
                if group['score_range'] == '绝对Good区域':
                    continue
                    
                start = group['threshold_start']
                end = group['threshold_end']
                
                if start == -1:  # 最低区间
                    if score <= end:
                        assigned_group = {
                            'range': group['score_range'],
                            'bad_ratio': group['bad_ratio'],
                            'label': 'definitely_good' if group['bad_ratio'] == 0 else 'mixed'
                        }
                        break
                else:
                    if start < score <= end:
                        assigned_group = {
                            'range': group['score_range'],
                            'bad_ratio': group['bad_ratio'],
                            'label': 'definitely_good' if group['bad_ratio'] == 0 else 'mixed'
                        }
                        break
        
        if assigned_group is None:
            assigned_group = {'range': 'unknown', 'bad_ratio': 0.5, 'label': 'unknown'}
        
        sample_labels.append({
            'ID': sample_id,
            'Predict': score,
            'score_range': assigned_group['range'],
            'bad_probability': assigned_group['bad_ratio'],
            'label': assigned_group['label']
        })
    
    # 保存标签数据
    labels_df = pd.DataFrame(sample_labels)
    output_file = merged_file_path.replace('.csv', '_sample_labels.csv')
    labels_df.to_csv(output_file, index=False)
    
    # 统计
    label_counts = labels_df['label'].value_counts()
    print("标签统计:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} 样本")
    
    print(f"样本标签已保存到: {output_file}")
    
    return labels_df

def find_optimal_cutoff(groups):
    """
    寻找最优的分割点以最大化F1分数
    """
    print(f"\n=== 寻找最优分割点 ===")
    
    # 按分数从低到高累积统计
    groups_by_score = sorted([g for g in groups if g['score_range'] != '绝对Good区域'], 
                           key=lambda x: x['threshold_end'])
    
    best_f1 = 0
    best_cutoff = None
    best_stats = None
    
    cumulative_bad = 0
    cumulative_total = 0
    
    print("分割点\t\t\tTP\tFP\tFN\tTN\t精确率\t召回率\tF1分数")
    print("-" * 80)
    
    for group in groups_by_score:
        cumulative_bad += group['bad_count']
        cumulative_total += group['total_samples']
        
        # 如果在这个分数点切分：≤此分数为bad，>此分数为good
        tp = cumulative_bad
        fp = cumulative_total - cumulative_bad
        fn = 727 - tp  # 假设总共727个bad
        tn = 7558 - cumulative_total - fn
        
        if tp > 0 and (tp + fp) > 0:
            precision = tp / (tp + fp)
            recall = tp / 727 if 727 > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"{group['score_range']:15}\t{tp:3d}\t{fp:4d}\t{fn:3d}\t{tn:4d}\t{precision:.3f}\t{recall:.3f}\t{f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_cutoff = group['score_range']
                best_stats = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 
                             'precision': precision, 'recall': recall, 'f1': f1}
    
    if best_stats:
        print(f"\n最优分割点: {best_cutoff}")
        print(f"最佳F1分数: {best_stats['f1']:.4f}")
        print(f"精确率: {best_stats['precision']:.4f}")
        print(f"召回率: {best_stats['recall']:.4f}")

# 主程序
if __name__ == "__main__":
    merged_file_path = "/Users/mannormal/4011/Qi Zihan/v2/results/py/merged_result1.csv"
    confusion_results_path = "/Users/mannormal/4011/Qi Zihan/v2/results/py/confusion_matrix_analysis.csv"
    
    try:
        groups = analyze_bad_groups_corrected(merged_file_path, confusion_results_path)
        find_optimal_cutoff(groups)
        
        print(f"\n=== 修正后的总结 ===")
        print("1. 正确分析了每个分数区间的bad/good分布")
        print("2. 识别了'确定是good'的区域")
        print("3. 找出了high-risk区域") 
        print("4. 生成了准确的样本标签")
        
    except Exception as e:
        print(f"分析出错: {e}")
        import traceback
        traceback.print_exc()