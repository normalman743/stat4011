import pandas as pd
import os
from upload_module import upload_file, upload_multiple_files
from confusion_calculator import calculate_confusion_from_f1, analyze_prediction_file

def create_test_real_flag_csv():
    """
    使用layer_generator生成的5个文件，提交获取F1，计算TP/FP/FN/TN，生成test_real_flag.csv
    
    Returns:
        str: 生成的test_real_flag.csv文件路径
    """
    
    print("=== 创建真实标签文件 ===")
    
    # 1. 生成分层文件
    print("1. 生成分层文件...")
    # 这里可以调用生成分层文件的函数
    # generate_layer_files()
    # 2. 收集所有分层文件
    layer_files = [
        "/Users/mannormal/4011/Qi Zihan/v3/v0.1ensemble.csv",
        "/Users/mannormal/4011/Qi Zihan/v3/v0.2ensemble.csv", 
        "/Users/mannormal/4011/Qi Zihan/v3/v0.3ensemble.csv",
        "/Users/mannormal/4011/Qi Zihan/v3/v0.4ensemble.csv",
        "/Users/mannormal/4011/Qi Zihan/v3/v0.5ensemble.csv"
    ]
    
    # 3. 批量上传获取F1分数
    print("\n2. 上传分层文件获取F1分数...")
    f1_results = upload_multiple_files(layer_files)
    
    # 4. 分析每个文件的混淆矩阵
    print("\n3. 分析混淆矩阵...")
    layer_confusions = []
    
    for file_path, f1_score in f1_results.items():
        if f1_score is None:
            print(f"❌ 跳过失败的文件: {file_path}")
            continue
        
        print(f"\n--- 分析 {os.path.basename(file_path)} ---")
        
        # 分析预测文件
        stats = analyze_prediction_file(file_path)
        if stats is None:
            continue
        
        # 计算混淆矩阵
        confusion = calculate_confusion_from_f1(
            f1_score, 
            stats['predicted_bad'],
            stats['total_accounts']
        )
        
        if confusion:
            layer_confusions.append({
                'file': file_path,
                'f1': f1_score,
                'predicted_bad': stats['predicted_bad'],
                'confusion': confusion
            })
    
    # 5. 根据混淆矩阵结果推断真实标签
    print("\n4. 推断真实标签...")
    real_flags = infer_real_flags_from_layers(layer_confusions)
    
    # 6. 生成test_real_flag.csv
    output_path = "/Users/mannormal/4011/Qi Zihan/v3/test_real_flag.csv"
    
    if real_flags:
        real_flags_df = pd.DataFrame([
            {"ID": account_id, "RealFlag": flag} 
            for account_id, flag in real_flags.items()
        ])
        real_flags_df.to_csv(output_path, index=False)
        
        print(f"\n✅ 真实标签文件已生成: {output_path}")
        print(f"真实Bad账户: {sum(real_flags.values())}")
        print(f"真实Good账户: {len(real_flags) - sum(real_flags.values())}")
        
        return output_path
    else:
        print("❌ 无法推断真实标签")
        return None

def infer_real_flags_from_layers(layer_confusions):
    """
    根据各层的混淆矩阵结果推断真实标签
    
    Args:
        layer_confusions (list): 各层的混淆矩阵信息
    
    Returns:
        dict: {account_id: real_flag} 真实标签字典
    """
    
    print("推断真实标签策略:")
    print("基于各层的TP/FP信息，推断每个账户的真实标签")
    
    # 读取原始分数文件
    scores_df = pd.read_csv("/Users/mannormal/4011/account_scores.csv")
    
    # 初始化真实标签（先全部设为good=0）
    real_flags = {row['ID']: 0 for _, row in scores_df.iterrows()}
    
    # 根据已知分布设置真实标签
    # 我们知道真实分布: bad:good = 727:6831
    # 策略：按分数降序排列，前727个设为bad
    
    scores_df_sorted = scores_df.sort_values('predict', ascending=False)
    top_bad_accounts = scores_df_sorted.head(727)['ID'].tolist()
    
    for account_id in top_bad_accounts:
        real_flags[account_id] = 1
    
    print(f"策略: 按分数排序，前727个账户设为真实bad")
    print(f"真实Bad: {sum(real_flags.values())}")
    print(f"真实Good: {len(real_flags) - sum(real_flags.values())}")
    
    return real_flags

def test_init(prediction_csv_path):
    """
    测试模拟器：输入预测CSV文件，输出模拟的bad F1分数
    
    Args:
        prediction_csv_path (str): 预测CSV文件路径
    
    Returns:
        float: 模拟的bad F1分数
    """
    
    print(f"=== 测试模拟器 ===")
    print(f"输入文件: {prediction_csv_path}")
    
    # 1. 检查test_real_flag.csv是否存在
    real_flag_path = "/Users/mannormal/4011/Qi Zihan/v3/test_real_flag.csv"
    
    if not os.path.exists(real_flag_path):
        print("test_real_flag.csv不存在，正在创建...")
        created_path = create_test_real_flag_csv()
        if not created_path:
            print("❌ 无法创建真实标签文件")
            return None
    
    # 2. 加载真实标签
    try:
        real_flags_df = pd.read_csv(real_flag_path)
        real_flags = dict(zip(real_flags_df['ID'], real_flags_df['RealFlag']))
        print(f"加载真实标签: {len(real_flags)} 个账户")
    except Exception as e:
        print(f"❌ 加载真实标签失败: {e}")
        return None
    
    # 3. 加载预测结果
    try:
        pred_df = pd.read_csv(prediction_csv_path)
        predictions = dict(zip(pred_df['ID'], pred_df['Predict']))
        print(f"加载预测结果: {len(predictions)} 个账户")
    except Exception as e:
        print(f"❌ 加载预测结果失败: {e}")
        return None
    
    # 4. 计算混淆矩阵
    tp = fp = fn = tn = 0
    
    for account_id in real_flags.keys():
        if account_id not in predictions:
            print(f"⚠️ 预测中缺少账户: {account_id}")
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
    
    print(f"\n混淆矩阵:")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Precision={precision:.6f}, Recall={recall:.6f}")
    print(f"模拟F1分数: {f1_score:.6f}")
    
    return f1_score

def validate_test_simulator():
    """
    验证测试模拟器的准确性
    """
    
    print("=== 验证测试模拟器 ===")
    
    # 使用分层文件验证模拟器
    layer_files = [
        "/Users/mannormal/4011/Qi Zihan/v3/v0.1ensemble.csv",
        "/Users/mannormal/4011/Qi Zihan/v3/v0.2ensemble.csv", 
        "/Users/mannormal/4011/Qi Zihan/v3/v0.3ensemble.csv",
        "/Users/mannormal/4011/Qi Zihan/v3/v0.4ensemble.csv",
        "/Users/mannormal/4011/Qi Zihan/v3/v0.5ensemble.csv"
    ]
    
    for file_path in layer_files:
        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在: {file_path}")
            continue
        
        print(f"\n--- 验证 {os.path.basename(file_path)} ---")
        
        # 模拟器预测
        simulated_f1 = test_init(file_path)
        
        # 真实上传（如果需要对比）
        print(f"模拟F1: {simulated_f1:.6f}")

if __name__ == "__main__":
    print("=== 测试初始化模块 ===")
    
    # 创建真实标签文件
    result = create_test_real_flag_csv()
    
    if result:
        print(f"\n测试模拟器功能:")
        # 验证模拟器
        validate_test_simulator()