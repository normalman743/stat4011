import pandas as pd
from time import sleep
def calculate_f1_from_real_flags(prediction_csv_path, real_flag_csv_path="/Users/mannormal/Desktop/课程/y4t1/stat 4011/融合二分模型_最终版 copy.csv"):
    """
    根据真实标签文件计算预测结果的F1分数
    
    Args:
        prediction_csv_path (str): 预测CSV文件路径 (格式: ID, Predict)
        real_flag_csv_path (str): 真实标签文件路径 (格式: ID, RealFlag)
    
    Returns:
        float: F1分数，如果失败返回None
    """
    #RealFlag
    try:
        # 1. 加载真实标签
        real_flags_df = pd.read_csv(real_flag_csv_path)
        real_flags = dict(zip(real_flags_df['ID'], real_flags_df['Predict']))
        
        # 2. 加载预测结果
        pred_df = pd.read_csv(prediction_csv_path)
        predictions = dict(zip(pred_df['ID'], pred_df['Predict']))
        
        # 3. 验证数据一致性
        if len(real_flags) != len(predictions):
            print(f"⚠️ 数据长度不匹配: 真实标签{len(real_flags)}, 预测结果{len(predictions)}")
        
        missing_accounts = set(real_flags.keys()) - set(predictions.keys())
        if missing_accounts:
            print(f"⚠️ 预测中缺少 {len(missing_accounts)} 个账户")
        
        # 4. 计算混淆矩阵
        tp = fp = fn = tn = 0
        
        for account_id in real_flags.keys():
            if account_id not in predictions:
                continue
            
            real_flag = real_flags[account_id]
            pred_flag = predictions[account_id]
            
            if real_flag == 1 and pred_flag == 1:
                tp += 1
            elif real_flag == 0 and pred_flag == 1:
                fp += 1
            elif real_flag == 1 and pred_flag == 0:
                fn += 1
            else:  # real_flag == 0 and pred_flag == 0
                tn += 1
        
        # 5. 计算F1分数
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        
        return f1_score
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        return None
    except Exception as e:
        print(f"❌ 计算F1失败: {e}")
        return None

def get_confusion_matrix(prediction_csv_path, real_flag_csv_path="/Users/mannormal/Desktop/课程/y4t1/stat 4011/融合二分模型_最终版 copy.csv"):
    """
    获取详细的混淆矩阵信息
    
    Args:
        prediction_csv_path (str): 预测CSV文件路径
        real_flag_csv_path (str): 真实标签文件路径
    
    Returns:
        dict: 包含TP, FP, FN, TN和各种指标的字典
    """
    
    try:
        # 1. 加载数据
        real_flags_df = pd.read_csv(real_flag_csv_path)
        real_flags = dict(zip(real_flags_df['ID'], real_flags_df['Predict']))
        
        pred_df = pd.read_csv(prediction_csv_path)
        predictions = dict(zip(pred_df['ID'], pred_df['Predict']))
        
        # 2. 计算混淆矩阵
        tp = fp = fn = tn = 0
        
        for account_id in real_flags.keys():
            if account_id not in predictions:
                continue
            
            real_flag = real_flags[account_id]
            pred_flag = predictions[account_id]
            
            if real_flag == 1 and pred_flag == 1:
                tp += 1
            elif real_flag == 0 and pred_flag == 1:
                fp += 1
            elif real_flag == 1 and pred_flag == 0:
                fn += 1
            else:
                tn += 1
        
        # 3. 计算各种指标
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1_score
        }
        
    except Exception as e:
        print(f"❌ 计算混淆矩阵失败: {e}")
        return None

def simulate_f1(prediction_csv_path):
    """
    模拟F1计算的简化接口
    直接使用默认的test_real_flag.csv文件
    
    Args:
        prediction_csv_path (str): 预测CSV文件路径
    
    Returns:
        float: F1分数
    """
    
    return calculate_f1_from_real_flags(prediction_csv_path)


def test():
    test_csv = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/submit.csv"
    f1 = calculate_f1_from_real_flags(test_csv)
    confusion = get_confusion_matrix(test_csv)
    print(f"计算的F1分数: {f1}")
    print(f"混淆矩阵: {confusion}")

if __name__ == "__main__":
    # 示例用法
    test2_csv = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/FINAL_BEST.csv"
    # ID Predict

    f12 = calculate_f1_from_real_flags(test2_csv)

    print(f"模拟F1分数: {f12}")


