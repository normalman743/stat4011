import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def load_data():
    """加载训练和测试数据"""
    # 加载训练数据
    train_df = pd.read_csv('/Users/mannormal/4011/Qi Zihan/original_data/train_acc.csv')
    test_df = pd.read_csv('/Users/mannormal/4011/Qi Zihan/original_data/test_acc_predict.csv')
    
    print(f"训练数据形状: {train_df.shape}")
    print(f"测试数据形状: {test_df.shape}")
    print(f"训练数据列名: {train_df.columns.tolist()}")
    print(f"测试数据列名: {test_df.columns.tolist()}")
    
    return train_df, test_df

def analyze_training_labels(train_df):
    """分析训练数据的标签分布"""
    print("\n=== 训练数据标签分析 ===")
    
    # 假设标签列是'flag'或类似名称
    label_cols = [col for col in train_df.columns if 'flag' in col.lower() or 'label' in col.lower()]
    if not label_cols:
        # 如果没有明显的标签列，显示所有列的唯一值
        print("未找到明显的标签列，显示所有列的唯一值:")
        for col in train_df.columns:
            unique_vals = train_df[col].unique()
            if len(unique_vals) <= 10:  # 只显示唯一值较少的列
                print(f"{col}: {unique_vals}")
        return None
    
    label_col = label_cols[0]
    print(f"使用标签列: {label_col}")
    
    # 标签分布
    label_counts = train_df[label_col].value_counts()
    print(f"标签分布:\n{label_counts}")
    print(f"标签比例:\n{train_df[label_col].value_counts(normalize=True)}")
    
    return label_col

def create_baseline_predictions(test_df, prediction_value):
    """创建baseline预测（全部预测为0或1）"""
    predictions = np.full(len(test_df), prediction_value)
    return predictions

def evaluate_predictions(y_true, y_pred, model_name):
    """评估预测结果"""
    print(f"\n=== {model_name} 性能评估 ===")
    
    # 基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1_binary = f1_score(y_true, y_pred, average='binary', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (binary): {precision:.4f}")
    print(f"Recall (binary): {recall:.4f}")
    print(f"F1-Score (binary): {f1_binary:.4f}")
    print(f"F1-Score (weighted): {f1_weighted:.4f}")
    print(f"F1-Score (macro): {f1_macro:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"混淆矩阵:\n{cm}")
    
    # 详细分类报告
    print("详细分类报告:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # 预测分布
    unique, counts = np.unique(y_pred, return_counts=True)
    pred_dist = dict(zip(unique, counts))
    print(f"预测分布: {pred_dist}")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_binary': precision,
        'recall_binary': recall,
        'f1_binary': f1_binary,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'prediction_distribution': pred_dist
    }

def main():
    """主函数"""
    print("=== Binary Baseline Models 评估 ===")
    
    # 加载数据
    train_df, test_df = load_data()
    
    # 分析训练数据标签
    label_col = analyze_training_labels(train_df)
    
    if label_col is None:
        print("请手动指定标签列名")
        return
    
    # 获取真实标签（这里假设测试数据也有标签，用于评估）
    # 如果测试数据没有标签，您需要使用验证集或交叉验证
    if label_col in test_df.columns:
        y_true = test_df[label_col].values
        print(f"使用测试数据的标签进行评估")
    else:
        # 使用训练数据的一部分作为验证集
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            train_df.drop(columns=[label_col]), 
            train_df[label_col], 
            test_size=0.2, 
            random_state=42,
            stratify=train_df[label_col]
        )
        y_true = y_val.values
        test_size = len(y_val)
        print(f"使用训练数据的20%作为验证集进行评估，验证集大小: {test_size}")
    
    print(f"真实标签分布: {np.unique(y_true, return_counts=True)}")
    
    # 创建和评估baseline模型
    results = []
    
    # Model 1: 全部预测为0
    pred_all_zero = create_baseline_predictions(test_df, 0)
    if len(pred_all_zero) != len(y_true):
        pred_all_zero = pred_all_zero[:len(y_true)]
    result_zero = evaluate_predictions(y_true, pred_all_zero, "全部预测0模型")
    results.append(result_zero)
    
    # Model 2: 全部预测为1  
    pred_all_one = create_baseline_predictions(test_df, 1)
    if len(pred_all_one) != len(y_true):
        pred_all_one = pred_all_one[:len(y_true)]
    result_one = evaluate_predictions(y_true, pred_all_one, "全部预测1模型")
    results.append(result_one)
    
    # 对比结果
    print("\n=== 模型对比总结 ===")
    comparison_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Accuracy': f"{r['accuracy']:.4f}",
            'F1-Binary': f"{r['f1_binary']:.4f}",
            'F1-Weighted': f"{r['f1_weighted']:.4f}",
            'F1-Macro': f"{r['f1_macro']:.4f}",
            'Precision': f"{r['precision_binary']:.4f}",
            'Recall': f"{r['recall_binary']:.4f}"
        } for r in results
    ])
    
    print(comparison_df.to_string(index=False))
    
    # 保存结果
    comparison_df.to_csv('baseline_binary_models_results.csv', index=False)
    print(f"\n结果已保存到 baseline_binary_models_results.csv")

if __name__ == "__main__":
    main()