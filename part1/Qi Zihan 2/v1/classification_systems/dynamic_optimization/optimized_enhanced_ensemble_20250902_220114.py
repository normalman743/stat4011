
# 优化后的Enhanced Ensemble模型 - 20250902_220114
# 基于原始enhanced_natxis_classification.py的改进版本
# 最佳参数: {'n_models': 100, 'sample_size': 532, 'voting_threshold': 93, 'n_estimators': 100}
# 标签编码映射 (classes -> integer): {'traditional_category': {'backward_only': np.int64(0), 'both_directions': np.int64(1), 'forward_only': np.int64(2), 'isolated': np.int64(3)}, 'volume_category': {'high_volume': np.int64(0), 'low_volume': np.int64(1), 'medium_volume': np.int64(2), 'no_transactions': np.int64(3)}, 'profit_category': {'loss_or_zero': np.int64(0), 'very_high_profit': np.int64(1)}, 'interaction_category': {'A_in_A_out_B_in': np.int64(0), 'A_in_A_out_B_in_B_out': np.int64(1), 'A_in_A_out_B_out': np.int64(2), 'A_in_B_in': np.int64(3), 'A_in_B_in_B_out': np.int64(4), 'A_in_B_out': np.int64(5), 'A_out': np.int64(6), 'A_out_B_in': np.int64(7), 'A_out_B_in_B_out': np.int64(8), 'A_out_B_out': np.int64(9), 'B_in': np.int64(10), 'B_in_B_out': np.int64(11), 'B_out': np.int64(12)}, 'behavior_category': {'inactive': np.int64(0), 'low_activity': np.int64(1), 'medium_activity_bidirectional': np.int64(2), 'medium_activity_unidirectional': np.int64(3)}}

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def optimized_enhanced_ensemble():
    """使用优化参数的Enhanced Ensemble模型"""

    # 加载数据
    try:
        features_df = pd.read_csv('../../feature_extraction/generated_features/all_features_with_categories.csv')
    except:
        features_df = pd.read_csv('../../feature_extraction/generated_features/all_features.csv')

    train_labels = pd.read_csv('../../original_data/train_acc.csv')
    data1 = features_df.merge(train_labels, on='account', how='inner')

    # 处理分类特征
    label_encoders = {'traditional_category': {'backward_only': np.int64(0), 'both_directions': np.int64(1), 'forward_only': np.int64(2), 'isolated': np.int64(3)}, 'volume_category': {'high_volume': np.int64(0), 'low_volume': np.int64(1), 'medium_volume': np.int64(2), 'no_transactions': np.int64(3)}, 'profit_category': {'loss_or_zero': np.int64(0), 'very_high_profit': np.int64(1)}, 'interaction_category': {'A_in_A_out_B_in': np.int64(0), 'A_in_A_out_B_in_B_out': np.int64(1), 'A_in_A_out_B_out': np.int64(2), 'A_in_B_in': np.int64(3), 'A_in_B_in_B_out': np.int64(4), 'A_in_B_out': np.int64(5), 'A_out': np.int64(6), 'A_out_B_in': np.int64(7), 'A_out_B_in_B_out': np.int64(8), 'A_out_B_out': np.int64(9), 'B_in': np.int64(10), 'B_in_B_out': np.int64(11), 'B_out': np.int64(12)}, 'behavior_category': {'inactive': np.int64(0), 'low_activity': np.int64(1), 'medium_activity_bidirectional': np.int64(2), 'medium_activity_unidirectional': np.int64(3)}}
    feature_cols = [col for col in data1.columns if col not in ['account', 'flag']]

    for col in feature_cols:
        if col in label_encoders:
            mapping = label_encoders[col]
            data1[col] = data1[col].astype(str).map(mapping).fillna(-1).astype(int)
        else:
            # 非分类特征，尝试转为数值型
            if data1[col].dtype == 'object':
                data1[col] = pd.to_numeric(data1[col], errors='coerce').fillna(0)

    # 最优参数
    BEST_PARAMS = {'n_models': 100, 'sample_size': 532, 'voting_threshold': 93, 'n_estimators': 100}

    # 集成预测
    np.random.seed(42)
    y_preds1 = []

    print(f"训练 {BEST_PARAMS['n_models']} 个模型...")

    for i in range(BEST_PARAMS['n_models']):
        if (i + 1) % 20 == 0:
            print(f"  完成 {i+1}/{BEST_PARAMS['n_models']} 个模型")

        # 平衡采样
        train11 = data1[data1.flag == 1].sample(
            min(BEST_PARAMS['sample_size'], len(data1[data1.flag == 1])),
            random_state=42 + i
        )
        train10 = data1[data1.flag == 0].sample(
            min(BEST_PARAMS['sample_size'], len(data1[data1.flag == 0])),
            random_state=42 + i + 1000
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
            random_state=42 + i,
            n_jobs=-1
        )

        clf1.fit(x_train1, y_train1)

        x_pred = data1[feature_cols].values
        x_pred = np.nan_to_num(x_pred)
        pred = clf1.predict(x_pred)
        y_preds1.append(pred)

    # 集成投票
    y_pred1 = np.where(
        pd.DataFrame(y_preds1).sum(axis='rows').values > BEST_PARAMS['voting_threshold'],
        1, 0
    )

    # 评估
    accuracy = accuracy_score(data1['flag'].values, y_pred1)
    f1_binary = f1_score(data1['flag'].values, y_pred1, average='binary')
    f1_weighted = f1_score(data1['flag'].values, y_pred1, average='weighted')

    print(f"\n=== 优化后模型性能 ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Binary: {f1_binary:.4f}")
    print(f"F1-Weighted: {f1_weighted:.4f}")

    return y_pred1, accuracy, f1_binary

if __name__ == "__main__":
    predictions, acc, f1 = optimized_enhanced_ensemble()
    print(f"\n最终结果: Accuracy={acc:.4f}, F1-Binary={f1:.4f}")
