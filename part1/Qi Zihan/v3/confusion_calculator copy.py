import pandas as pd

def calculate_confusion_from_f1(bad_f1, predicted_bad_count, total_accounts=7558, true_bad=727, true_good=6831):
    """
    根据bad F1分数和预测分布计算混淆矩阵
    
    Args:
        bad_f1 (float): bad类别的F1分数
        predicted_bad_count (int): 预测为bad的账户数量
        total_accounts (int): 总账户数，默认7558
        true_bad (int): 真实bad数量，默认727
        true_good (int): 真实good数量，默认6831
    
    Returns:
        dict: {"TP": x, "FP": x, "FN": x, "TN": x}
    """
    
    # 如果F1为0，说明没有正确预测
    if bad_f1 == 0:
        confusion_matrix = {
            'TP': 0,
            'FP': predicted_bad_count,
            'FN': true_bad,
            'TN': true_good - (predicted_bad_count - 0)  # TN = true_good - FP
        }
        
        return confusion_matrix
    
    # 通过F1分数反推TP
    best_tp = 0
    best_f1_diff = float('inf')
    best_confusion = None
    
    # 遍历所有可能的TP值
    max_possible_tp = min(predicted_bad_count, true_bad)
    
    for tp in range(max_possible_tp + 1):
        # 计算其他值
        fp = predicted_bad_count - tp
        fn = true_bad - tp
        tn = true_good - fp
        
        # 验证合理性
        if fp < 0 or fn < 0 or tn < 0:
            continue
        
        # 验证总数是否正确
        if tp + fp + fn + tn != total_accounts:
            continue
        
        # 计算F1
        if predicted_bad_count == 0:
            precision = 0
        else:
            precision = tp / predicted_bad_count
        
        if true_bad == 0:
            recall = 0
        else:
            recall = tp / true_bad
        
        if precision + recall == 0:
            calculated_f1 = 0
        else:
            calculated_f1 = 2 * (precision * recall) / (precision + recall)
        
        # 比较F1差异
        f1_diff = abs(calculated_f1 - bad_f1)
        if f1_diff < best_f1_diff:
            best_f1_diff = f1_diff
            best_tp = tp
            best_confusion = {
                'TP': tp,
                'FP': fp,
                'FN': fn,
                'TN': tn
            }
    
    if best_confusion is None:
        print("❌ 无法找到合理的混淆矩阵")
        return None

    
    # 验证计算
    tp, fp, fn, tn = best_confusion['TP'], best_confusion['FP'], best_confusion['FN'], best_confusion['TN']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    calculated_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    
    return best_confusion
