import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def voting_rf_with_ratio_control():
    """
    5折交叉验证随机森林投票预测，控制输出比例接近9.3:1
    """
    print("=== 5折交叉验证随机森林投票预测 ===")
    
    # 读取清理后的特征数据
    features_file = "/Users/mannormal/4011/Qi Zihan/v2/feature_extraction/result/all_features_no_flag_features.csv"
    or_train_file = "/Users/mannormal/4011/Qi Zihan/original_data/train_acc.csv"
    or_test_file = "/Users/mannormal/4011/Qi Zihan/original_data/test_acc_predict.csv"
    fe = pd.read_csv(features_file)
    ta = pd.read_csv(or_train_file)
    te = pd.read_csv(or_test_file)

    print(f"数据维度: {fe.shape}")
    
    # 根据原始文件区分训练集和测试集
    train_accounts = set(ta['account'])
    test_accounts = set(te['account'])
    
    # 处理训练数据 - 需要合并flag标签
    df_train = fe[fe['account'].isin(train_accounts)].copy()
    df_train = df_train.merge(ta[['account', 'flag']], on='account', how='inner')
    
    # 处理测试数据 - 保持原始顺序
    df_test = fe[fe['account'].isin(test_accounts)].copy()
    
    # 确保测试数据按照te文件中的顺序排列
    df_test = te[['account']].merge(df_test, on='account', how='left')
    
    print(f"训练集大小: {len(df_train)}")
    print(f"测试集大小: {len(df_test)}")
    print(f"测试集账户顺序检查 - 前5个: {df_test['account'].head().tolist()}")
    print(f"原始te文件顺序 - 前5个: {te['account'].head().tolist()}")
    
    # 准备特征和标签
    feature_cols = [col for col in fe.columns if col != 'account']
    
    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train['flag']
    X_test = df_test[feature_cols].fillna(0)
    
    # 将标签转换为0/1 (good=0, bad=1)
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    print(f"特征数量: {len(feature_cols)}")
    print(f"训练集标签分布: {np.bincount(y_train_encoded)}")
    print(f"标签映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    print("\n=== 5折交叉验证 ===")
    
    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    trained_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train_encoded)):
        print(f"\n--- Fold {fold + 1} ---")
        
        # 分割数据
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train_encoded[train_idx], y_train_encoded[val_idx]
        
        # 训练随机森林
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42 + fold,
            n_jobs=-1
        )
        
        rf.fit(X_fold_train, y_fold_train)
        trained_models.append(rf)
        
        # 验证集预测
        y_pred = rf.predict(X_fold_val)
        
        # 混淆矩阵
        cm = confusion_matrix(y_fold_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"混淆矩阵:")
        print(f"  True Good (TN): {tn}")
        print(f"  False Bad (FP): {fp}")
        print(f"  False Good (FN): {fn}")
        print(f"  True Bad (TP): {tp}")
        
        # F1分数
        f1_good = f1_score(y_fold_val, y_pred, pos_label=0)
        f1_bad = f1_score(y_fold_val, y_pred, pos_label=1)
        f1 = f1_score(y_fold_val, y_pred, average='weighted')
        f1_macro = f1_score(y_fold_val, y_pred, average='macro')
        f1_binary = f1_score(y_fold_val, y_pred, average='binary')

        print(f"F1分数:")
        print(f"  Overall (weighted): {f1:.4f}")
        print(f"  Macro: {f1_macro:.4f}")
        print(f"  Binary (bad=1): {f1_binary:.4f}")
        print(f"  Good F1: {f1_good:.4f}")
        print(f"  Bad F1: {f1_bad:.4f}")
        
        fold_results.append({
            'fold': fold + 1,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'f1_good': f1_good, 'f1_bad': f1_bad
        })
    
    # 计算平均F1分数
    mean_f1_good = np.mean([r['f1_good'] for r in fold_results])
    mean_f1_bad = np.mean([r['f1_bad'] for r in fold_results])
    
    print(f"\n=== 交叉验证总结 ===")
    print(f"平均 Good F1: {mean_f1_good:.4f}")
    print(f"平均 Bad F1: {mean_f1_bad:.4f}")
    
    # 找到F1分数最高的模型
    best_f1_bad = max([r['f1_bad'] for r in fold_results])
    best_model_idx = [r['f1_bad'] for r in fold_results].index(best_f1_bad)
    best_model = trained_models[best_model_idx]
    
    print(f"\n=== 选择最佳模型进行预测 ===")
    print(f"最佳模型: Fold {best_model_idx + 1}")
    print(f"最佳Bad F1: {best_f1_bad:.4f}")
    
    # 用最佳模型预测测试集
    test_probabilities = best_model.predict_proba(X_test)
    
    # 调整阈值以达到接近9.3:1的比例
    target_bad_ratio = 1 / (9.3 + 1)  # ≈ 0.097
    print(f"目标bad比例: {target_bad_ratio:.3f}")
    
    # 寻找合适的阈值
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_threshold = 0.5
    best_diff = float('inf')
    
    for threshold in thresholds:
        predictions = (test_probabilities[:, 1] >= threshold).astype(int)
        actual_bad_ratio = predictions.sum() / len(predictions)
        diff = abs(actual_bad_ratio - target_bad_ratio)
        
        if diff < best_diff:
            best_diff = diff
            best_threshold = threshold
    
    # 最终预测
    final_predictions = (test_probabilities[:, 1] >= best_threshold).astype(int)
    final_bad_ratio = final_predictions.sum() / len(final_predictions)
    good_count = (final_predictions == 0).sum()
    bad_count = (final_predictions == 1).sum()
    
    print(f"最佳阈值: {best_threshold:.3f}")
    print(f"实际bad比例: {final_bad_ratio:.3f}")
    print(f"Good:Bad = {good_count}:{bad_count} ≈ {good_count/bad_count:.1f}:1")
    
    # 准备输出结果
    result_df = pd.DataFrame({
        'account': df_test['account'].values,
        'Predict': final_predictions
    })
    
    # 生成文件名 - 使用最佳模型的F1分数
    best_f1_str = f"{best_f1_bad:.3f}".replace('.', '')
    ratio_str = f"{good_count}_{bad_count}"
    filename = f"best_rf_badf1_{best_f1_str}_ratio_{ratio_str}.csv"
    output_path = f"/Users/mannormal/4011/Qi Zihan/v2/results/{filename}"
    
    # 保存结果
    result_df.to_csv(output_path, index=False)
    
    print(f"\n=== 结果保存 ===")
    print(f"预测结果保存到: {output_path}")
    print(f"文件名含义:")
    print(f"  best_rf: 使用F1分数最高的单个模型")
    print(f"  badf1_{best_f1_str}: 最佳bad F1 = {best_f1_bad:.3f}")
    print(f"  ratio_{ratio_str}: Good:Bad = {good_count}:{bad_count}")
    
    # 显示预测结果样例
    print(f"\n预测结果前10行:")
    print(result_df.head(10))
    
    return result_df, fold_results, best_f1_bad

if __name__ == "__main__":
    try:
        result_df, fold_results, best_f1_bad = voting_rf_with_ratio_control()
        print("最佳模型预测完成！")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()