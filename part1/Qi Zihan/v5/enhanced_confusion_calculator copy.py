import pandas as pd
import json

# 层级配置
LAYER_CONFIG = {
    "high_good": {
        "real_good": 6626,
        "real_bad": 154,
        "total": 6780,
        "description": "0.0-0.1范围"
    },
    "mid": {
        "real_good": 168, 
        "real_bad": 124,
        "total": 292,
        "description": "0.1-0.8范围合并"
    },
    "high_bad": {
        "real_good": 37,
        "real_bad": 449, 
        "total": 486,
        "description": "0.8-1.0范围"
    }
}

def calculate_confusion_from_f1(bad_f1, predicted_bad_count, layer, total_accounts=7558, true_bad=727, true_good=6831):
    """
    根据bad F1分数和预测分布计算混淆矩阵，增加layer参数
    
    Args:
        bad_f1 (float): bad类别的F1分数
        predicted_bad_count (int): 预测为bad的账户数量
        layer (str): 层级名称 ("high_good", "mid", "high_bad")
        total_accounts (int): 总账户数，默认7558
        true_bad (int): 真实bad数量，默认727
        true_good (int): 真实good数量，默认6831
    
    Returns:
        dict: {"TP": x, "FP": x, "FN": x, "TN": x, "layer_analysis": {...}}
    """
    
    # 如果F1为0，说明没有正确预测
    if bad_f1 == 0:
        confusion_matrix = {
            'TP': 0,
            'FP': predicted_bad_count,
            'FN': true_bad,
            'TN': true_good - predicted_bad_count,
            'layer_analysis': {
                'layer': layer,
                'layer_config': LAYER_CONFIG.get(layer, {}),
                'f1_score': bad_f1,
                'predicted_bad_count': predicted_bad_count
            }
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
                'TN': tn,
                'layer_analysis': {
                    'layer': layer,
                    'layer_config': LAYER_CONFIG.get(layer, {}),
                    'f1_score': bad_f1,
                    'predicted_bad_count': predicted_bad_count,
                    'calculated_f1': calculated_f1,
                    'f1_diff': f1_diff
                }
            }
    
    if best_confusion is None:
        print("❌ 无法找到合理的混淆矩阵")
        return None
    
    return best_confusion

def get_A_B_only(bad_f1, predicted_bad_count, layer):
    result = calculate_confusion_from_f1(bad_f1, predicted_bad_count, layer=layer)
    baseline = LAYER_CONFIG.get(layer, {})
    realgood = baseline.get('real_good', 0)
    realbad = baseline.get('real_bad', 0)
    print(f" TP: {result['TP']}")
    print(f" FP: {result['FP']}")
    print(f" FN: {result['FN']}")
    print(f" TN: {result['TN']}")

    print(f"realgood: {realgood}")
    print(f"realbad: {realbad}")
    print(f"B_TP : {realgood - result['TN']}")
    print(f"B_FP : {realbad - result['FN']}")
    if result:
        return {
            'A_TP': result['TN'],
            'A_FP': result['FN'],
            'B_TP': realgood - result['TN'],
            'B_FP': realbad - result['FN'],
        }
    return None

