# 优化后的Enhanced Ensemble模型 - 20250902_214337
# 基于原始enhanced_natxis_classification.py的改进版本
# 最佳参数: {'n_models': 100, 'sample_size': 532, 'voting_threshold': 93, 'n_estimators': 100}

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

def optimized_enhanced_ensemble():
    """使用优化参数的Enhanced Ensemble模型"""
    
    # 加载数据
    try:
        features_df = pd.read_csv('../../feature_extraction/generated_features/all_features_with_categories.csv')
        print(f"使用增强特征集: {features_df.shape[1]-1}个特征")
    except:
        features_df = pd.read_csv('../../feature_extraction/generated_features/all_features.csv')
        print(f"使用基础特征集: {features_df.shape[1]-1}个特征")
    
    train_labels = pd.read_csv('../../original_data/train_acc.csv')
    data1 = features_df.merge(train_labels, on='account', how='inner')
    
    print(f"数据形状: {data1.shape}")
    print(f"标签分布: {data1['flag'].value_counts().to_dict()}")
    
    # 处理分类特征 - 重新创建和训练编码器
    feature_cols = [col for col in data1.columns if col not in ['account', 'flag']]
    label_encoders = {}
    
    print("\n=== 处理分类特征 ===")
    categorical_cols = []
    for col in feature_cols:
        if data1[col].dtype == 'object' or data1[col].dtype.name == 'category':
            categorical_cols.append(col)
            print(f"发现分类特征: {col}, 唯一值: {len(data1[col].unique())}")
    
    # 编码分类特征
    for col in categorical_cols:
        le = LabelEncoder()
        data1[col] = le.fit_transform(data1[col].astype(str))
        label_encoders[col] = le
        print(f"编码 {col}: {dict(list(zip(le.classes_, le.transform(le.classes_)))[:5])}...")  # 只显示前5个
    
    # 确保所有特征都是数值型并处理缺失值
    for col in feature_cols:
        if data1[col].dtype == 'object':
            print(f"强制转换 {col} 为数值型")
            data1[col] = pd.to_numeric(data1[col], errors='coerce')
        
        # 统一处理所有缺失值
        if data1[col].isnull().any():
            null_count = data1[col].isnull().sum()
            print(f"填充 {col} 的 {null_count} 个缺失值")
            data1[col] = data1[col].fillna(0)
        
        # 确保数据类型为float64
        data1[col] = data1[col].astype(np.float64)
    
    print(f"分类特征编码完成，共处理 {len(categorical_cols)} 个分类特征")
    
    # 最优参数
    BEST_PARAMS = {
        'n_models': 100, 
        'sample_size': 532, 
        'voting_threshold': 93, 
        'n_estimators': 100
    }
    
    print(f"\n=== 开始集成预测 ===")
    print(f"使用参数: {BEST_PARAMS}")
    
    # 集成预测
    np.random.seed(42)
    y_preds1 = []
    
    print(f"训练 {BEST_PARAMS['n_models']} 个模型...")
    
    for i in range(BEST_PARAMS['n_models']):
        if (i + 1) % 20 == 0:
            print(f"  完成 {i+1}/{BEST_PARAMS['n_models']} 个模型")
        
        # 平衡采样
        positive_samples = len(data1[data1.flag == 1])
        negative_samples = len(data1[data1.flag == 0])
        
        train11 = data1[data1.flag == 1].sample(
            min(BEST_PARAMS['sample_size'], positive_samples), 
            random_state=42 + i
        )
        train10 = data1[data1.flag == 0].sample(
            min(BEST_PARAMS['sample_size'], negative_samples), 
            random_state=42 + i + 1000
        )
        
        train1 = pd.concat([train10, train11], axis='rows')
        x_train1 = train1[feature_cols].values.astype(np.float64)
        y_train1 = train1.flag.values.astype(np.int64)
        
        # 处理NaN值和无穷值
        if np.any(pd.isnull(x_train1)) or not np.isfinite(x_train1).all():
            x_train1 = np.where(np.isfinite(x_train1), x_train1, 0)
        
        # RandomForest
        clf1 = RandomForestClassifier(
            n_estimators=BEST_PARAMS['n_estimators'],
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42 + i,
            n_jobs=-1
        )
        
        clf1.fit(x_train1, y_train1)
        
        # 预测全部数据
        x_pred = data1[feature_cols].values.astype(np.float64)
        if np.any(pd.isnull(x_pred)) or not np.isfinite(x_pred).all():
            x_pred = np.where(np.isfinite(x_pred), x_pred, 0)
        
        pred = clf1.predict(x_pred)
        y_preds1.append(pred)
    
    # 集成投票
    votes_df = pd.DataFrame(y_preds1)
    vote_counts = votes_df.sum(axis=0).values
    y_pred1 = np.where(vote_counts > BEST_PARAMS['voting_threshold'], 1, 0)
    
    print(f"\n=== 投票统计 ===")
    print(f"投票阈值: {BEST_PARAMS['voting_threshold']}")
    print(f"最大投票数: {vote_counts.max()}")
    print(f"最小投票数: {vote_counts.min()}")
    print(f"预测为正类的样本数: {(y_pred1 == 1).sum()}")
    
    # 评估
    y_true = data1['flag'].values
    accuracy = accuracy_score(y_true, y_pred1)
    f1_binary = f1_score(y_true, y_pred1, average='binary', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred1, average='weighted', zero_division=0)
    
    # 详细评估
    from sklearn.metrics import precision_score, recall_score, classification_report
    precision = precision_score(y_true, y_pred1, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred1, average='binary', zero_division=0)
    
    print(f"\n=== 优化后模型性能 ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Binary: {f1_binary:.4f}")
    print(f"F1-Weighted: {f1_weighted:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    print(f"\n=== 分类报告 ===")
    print(classification_report(y_true, y_pred1, target_names=['正常', '异常']))
    
    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred1)
    print(f"\n=== 混淆矩阵 ===")
    print(f"              预测")
    print(f"         正常    异常")
    print(f"实际 正常  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"     异常  {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    return y_pred1, accuracy, f1_binary

def test_reproducibility(n_runs=3):
    """测试重现性"""
    print("=== 重现性测试 ===")
    results = []
    
    for run in range(n_runs):
        print(f"\n--- 第 {run+1} 次运行 ---")
        predictions, acc, f1 = optimized_enhanced_ensemble()
        
        stats = {
            'accuracy': acc,
            'f1_binary': f1,
            'positive_predictions': (predictions == 1).sum(),
            'prediction_sum': predictions.sum()
        }
        results.append(stats)
        print(f"运行 {run+1}: Accuracy={acc:.4f}, F1={f1:.4f}")
    
    # 检查一致性
    print(f"\n=== 重现性结果 ===")
    first_result = results[0]
    all_same = True
    
    for i, result in enumerate(results[1:], 1):
        for key in first_result.keys():
            if abs(first_result[key] - result[key]) > 1e-10:
                print(f"运行 {i+1} 与运行 1 在 {key} 上不同:")
                print(f"  运行 1: {first_result[key]}")
                print(f"  运行 {i+1}: {result[key]}")
                all_same = False
    
    if all_same:
        print("✅ 所有运行结果完全一致，模型具有完美的重现性")
    else:
        print("❌ 运行结果存在微小差异")
    
    return results

if __name__ == "__main__":
    # 单次运行
    print("=== 单次运行测试 ===")
    predictions, acc, f1 = optimized_enhanced_ensemble()
    print(f"\n最终结果: Accuracy={acc:.4f}, F1-Binary={f1:.4f}")
    
    # 可选：测试重现性（取消注释下面的代码）
    print("\n" + "="*50)
    test_reproducibility(3)