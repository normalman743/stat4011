# 优化后的Enhanced Ensemble模型 - 20250902_221211
# 基于原始enhanced_natxis_classification.py的改进版本
# 最佳参数: {'n_models': 100, 'sample_size': 700, 'voting_threshold': 88, 'n_estimators': 100}

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def optimized_enhanced_ensemble():
    """使用优化参数的Enhanced Ensemble模型"""

    # 加载数据
    try:
        features_df = pd.read_csv('../../feature_extraction/generated_features/all_features_with_categories.csv')
    except:
        features_df = pd.read_csv('../../feature_extraction/generated_features/all_features.csv')

    train_labels = pd.read_csv('../../original_data/train_acc.csv')
    
    # 重要修复1: 复制原始代码的标签处理逻辑
    train_labels.loc[train_labels['flag'] == 0, 'flag'] = -1
    
    data1 = features_df.merge(train_labels, on='account', how='inner')

    # 处理分类特征
    label_encoders = {'traditional_category': {'backward_only': 0, 'both_directions': 1, 'forward_only': 2, 'isolated': 3}, 'volume_category': {'high_volume': 0, 'low_volume': 1, 'medium_volume': 2, 'no_transactions': 3}, 'profit_category': {'loss_or_zero': 0, 'very_high_profit': 1}, 'interaction_category': {'A_in_A_out_B_in': 0, 'A_in_A_out_B_in_B_out': 1, 'A_in_A_out_B_out': 2, 'A_in_B_in': 3, 'A_in_B_in_B_out': 4, 'A_in_B_out': 5, 'A_out': 6, 'A_out_B_in': 7, 'A_out_B_in_B_out': 8, 'A_out_B_out': 9, 'B_in': 10, 'B_in_B_out': 11, 'B_out': 12}, 'behavior_category': {'inactive': 0, 'low_activity': 1, 'medium_activity_bidirectional': 2, 'medium_activity_unidirectional': 3}}
    feature_cols = [col for col in data1.columns if col not in ['account', 'flag']]

    for col in feature_cols:
        if col in label_encoders:
            mapping = label_encoders[col]
            data1[col] = data1[col].astype(str).map(mapping).fillna(-1).astype(int)
        else:
            if data1[col].dtype == 'object':
                data1[col] = pd.to_numeric(data1[col], errors='coerce').fillna(0)
    
    # 确保所有特征都是数值型
    for col in feature_cols:
        if data1[col].dtype == 'object':
            data1[col] = pd.to_numeric(data1[col], errors='coerce')
        data1[col] = data1[col].fillna(0).astype(np.float64)

    # 重要修复2: 为了与原始逻辑一致，转换标签为0/1用于训练
    data_for_training = data1.copy()
    data_for_training.loc[data_for_training['flag'] == -1, 'flag'] = 0

    # 创建训练/验证分割
    train_data, val_data = train_test_split(
        data_for_training, test_size=0.2, random_state=42, stratify=data_for_training['flag']
    )
    
    print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")
    print(f"验证集标签分布: {val_data['flag'].value_counts().to_dict()}")

    # 尝试不同的参数组合
    param_sets = [
        {'n_models': 100, 'sample_size': 700, 'voting_threshold': 88, 'n_estimators': 100},
        {'n_models': 100, 'sample_size': 532, 'voting_threshold': 93, 'n_estimators': 100},  # 原始参数
        {'n_models': 100, 'sample_size': 400, 'voting_threshold': 90, 'n_estimators': 100},
        {'n_models': 80, 'sample_size': 532, 'voting_threshold': 75, 'n_estimators': 120},
        {'n_models': 150, 'sample_size': 600, 'voting_threshold': 120, 'n_estimators': 80},
    ]

    best_f1 = 0
    best_params = None
    best_predictions = None

    for i, BEST_PARAMS in enumerate(param_sets):
        print(f"\n=== 测试参数组合 {i+1}/{len(param_sets)} ===")
        print(f"参数: {BEST_PARAMS}")

        # 在训练集上进行集成训练，然后在验证集上预测
        np.random.seed(42)
        trained_models = []

        print(f"在训练集上训练 {BEST_PARAMS['n_models']} 个模型...")

        for j in range(BEST_PARAMS['n_models']):
            if (j + 1) % 20 == 0:
                print(f"  完成 {j+1}/{BEST_PARAMS['n_models']} 个模型")

            # 重要修复3: 使用与原始代码一致的平衡采样策略
            positive_samples = len(train_data[train_data.flag == 1])
            negative_samples = len(train_data[train_data.flag == 0])
            
            # 使用最小值进行平衡采样，确保类别平衡
            sample_size = min(BEST_PARAMS['sample_size'], positive_samples, negative_samples)
            
            train11 = train_data[train_data.flag == 1].sample(
                sample_size, 
                random_state=42 + j,
                replace=True if sample_size > positive_samples else False
            )
            train10 = train_data[train_data.flag == 0].sample(
                sample_size,
                random_state=42 + j + 1000,
                replace=True if sample_size > negative_samples else False
            )

            train1 = pd.concat([train10, train11], axis='rows')
            x_train1 = train1[feature_cols].values
            y_train1 = train1.flag.values

            # 处理NaN值
            x_train1 = np.nan_to_num(x_train1)

            # RandomForest
            clf1 = RandomForestClassifier(
                n_estimators=BEST_PARAMS.get('n_estimators', 100),
                max_depth=BEST_PARAMS.get('max_depth', None),
                min_samples_split=BEST_PARAMS.get('min_samples_split', 2),
                min_samples_leaf=BEST_PARAMS.get('min_samples_leaf', 1),
                random_state=42 + j,
                n_jobs=-1
            )

            clf1.fit(x_train1, y_train1)
            trained_models.append(clf1)

        # 使用训练好的模型在验证集上预测
        print(f"使用 {len(trained_models)} 个训练好的模型在验证集上预测...")
        y_preds1 = []

        x_pred = val_data[feature_cols].values
        x_pred = np.nan_to_num(x_pred)

        for j, clf in enumerate(trained_models):
            if (j + 1) % 20 == 0:
                print(f"  完成预测 {j+1}/{len(trained_models)} 个模型")
            pred = clf.predict(x_pred)
            y_preds1.append(pred)

        # 集成投票
        y_pred1 = np.where(
            pd.DataFrame(y_preds1).sum(axis='rows').values > BEST_PARAMS['voting_threshold'],
            1, 0
        )

        # 评估（在验证集上）
        accuracy = accuracy_score(val_data['flag'].values, y_pred1)
        f1_binary = f1_score(val_data['flag'].values, y_pred1, average='binary')
        f1_weighted = f1_score(val_data['flag'].values, y_pred1, average='weighted')

        print(f"验证集性能:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Binary: {f1_binary:.4f}")
        print(f"  F1-Weighted: {f1_weighted:.4f}")

        if f1_binary > best_f1:
            best_f1 = f1_binary
            best_params = BEST_PARAMS
            best_predictions = y_pred1

    print(f"\n=== 最佳结果 ===")
    print(f"最佳参数: {best_params}")
    print(f"最佳F1-Binary: {best_f1:.4f}")

    return best_predictions, accuracy_score(val_data['flag'].values, best_predictions), best_f1

if __name__ == "__main__":
    predictions, acc, f1 = optimized_enhanced_ensemble()
    print(f"\n最终结果: Accuracy={acc:.4f}, F1-Binary={f1:.4f}")
