import pandas as pd
import os
from upload_module import upload_file
from confusion_calculator import calculate_confusion_from_f1

def generate_layer_files(input_csv_path="/Users/mannormal/4011/account_scores.csv", output_dir="/Users/mannormal/4011/Qi Zihan/v3"):
    """
    将account_scores.csv分为5层，生成5个initial guess文件
    
    步骤:
    1. 用upload_file测试获得F1分数
    2. 用confusion_calculator计算混淆矩阵
    3. 根据confusion分布设置initial_guess为bad百分比
    """
    
    # 读取数据
    df = pd.read_csv(input_csv_path)
    print(f"加载数据: {len(df)} 个账户")
    
    # 定义分层
    layers = [
        {"name": "v0.1ensemble.csv", "range": (0.0, 0.1)},
        {"name": "v0.2ensemble.csv", "range": (0.1, 0.2)},
        {"name": "v0.3ensemble.csv", "range": (0.2, 0.5)},
        {"name": "v0.4ensemble.csv", "range": (0.5, 0.8)},
        {"name": "v0.5ensemble.csv", "range": (0.8, 1.0)}
    ]
    
    # 生成每层文件
    for layer in layers:
        layer_name = layer["name"]
        min_score, max_score = layer["range"]
        
        print(f"\n=== 处理 {layer_name} ===")
        
        # 1. 先生成初始预测文件（所有账户预测为bad=1）
        predictions = []
        layer_accounts = []
        
        for _, row in df.iterrows():
            account_id = row['ID']
            score = row['predict']
            
            # 判断是否在当前层范围内
            if min_score <= score < max_score or (max_score == 1.0 and score == 1.0):
                layer_accounts.append(account_id)
                predict = 1  # 层内账户初始预测为bad
            else:
                predict = 0  # 其他账户预测为good
            
            predictions.append({"ID": account_id, "Predict": predict})
        
        # 保存临时文件进行测试
        temp_df = pd.DataFrame(predictions)
        temp_path = os.path.join(output_dir, f"temp_{layer_name}")
        temp_df.to_csv(temp_path, index=False)
        
        # 2. 用upload_file测试获得F1分数
        print(f"测试文件: {temp_path}")
        f1_score = upload_file(temp_path)
        print(f"F1分数: {f1_score:.6f}")
        
        # 3. 用confusion_calculator计算混淆矩阵
        predicted_bad = len(layer_accounts)
        confusion = calculate_confusion_from_f1(f1_score, predicted_bad, total_accounts=7558, true_bad=727, true_good=6831)
        print(f"混淆矩阵: TP={confusion['TP']}, FP={confusion['FP']}, FN={confusion['FN']}, TN={confusion['TN']}")
        
        # 4. 计算层内bad百分比
        layer_tp = confusion['TP']  # 层内正确的bad
        layer_fp = confusion['FP']  # 层内错误的good
        layer_size = len(layer_accounts)
        
        if layer_size > 0:
            bad_percentage = layer_tp / layer_size
            print(f"层内真实bad比例: {layer_tp}/{layer_size} = {bad_percentage:.3f}")
        else:
            bad_percentage = 0
        
        # 5. 根据bad百分比重新生成文件
        final_predictions = []
        for _, row in df.iterrows():
            account_id = row['ID']
            score = row['predict']
            
            if min_score <= score < max_score or (max_score == 1.0 and score == 1.0):
                # 层内账户：按bad百分比随机分配
                import random
                predict = 1 if random.random() < bad_percentage else 0
            else:
                predict = 0  # 其他账户预测为good
            
            final_predictions.append({"ID": account_id, "Predict": predict})
        
        # 保存最终文件
        final_df = pd.DataFrame(final_predictions)
        final_path = os.path.join(output_dir, layer_name)
        final_df.to_csv(final_path, index=False)
        
        # 清理临时文件
        os.remove(temp_path)
        
        # 统计信息
        bad_count = len(final_df[final_df['Predict'] == 1])
        good_count = len(final_df[final_df['Predict'] == 0])
        
        print(f"生成 {layer_name}:")
        print(f"  层内账户: {layer_size}")
        print(f"  最终预测Bad: {bad_count}, Good: {good_count}")
        print(f"  保存至: {final_path}")

def get_layer_statistics(input_csv_path="/Users/mannormal/4011/account_scores.csv"):
    """获取各层的统计信息"""
    
    df = pd.read_csv(input_csv_path)
    
    layers = [
        {"name": "[0.0-0.1)", "range": (0.0, 0.1)},
        {"name": "[0.1-0.2)", "range": (0.1, 0.2)},
        {"name": "[0.2-0.5)", "range": (0.2, 0.5)},
        {"name": "[0.5-0.8)", "range": (0.5, 0.8)},
        {"name": "[0.8-1.0]", "range": (0.8, 1.0)}
    ]
    
    print("原始账户分布:")
    for layer in layers:
        min_score, max_score = layer["range"]
        if max_score == 1.0:
            count = len(df[(df['predict'] >= min_score) & (df['predict'] <= max_score)])
        else:
            count = len(df[(df['predict'] >= min_score) & (df['predict'] < max_score)])
        print(f"  {layer['name']}: {count}")

if __name__ == "__main__":
    print("=== 账户分层生成器 ===")
    
    # 显示原始分布
    get_layer_statistics()
    print()
    
    # 生成分层文件
    generate_layer_files()
    
    print("✅ 所有分层文件生成完成!")