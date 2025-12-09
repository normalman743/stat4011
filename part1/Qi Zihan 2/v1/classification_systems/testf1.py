import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import os
from datetime import datetime

def create_bad_model_predictions():
    """创建故意表现很差的模型来诊断F1评价指标"""
    
    print("="*60)
    print("🔍 故意训练差模型 - F1指标诊断实验")
    print("="*60)
    
    # 1. 数据加载
    print("\n📂 加载数据...")
    data_path = "/Users/mannormal/4011/Qi Zihan/feature_extraction/generated_features/all_features_with_time.csv"
    df = pd.read_csv(data_path)
    
    train_path = "/Users/mannormal/4011/Qi Zihan/original_data/train_acc.csv"
    train_df = pd.read_csv(train_path)
    
    test_path = "/Users/mannormal/4011/Qi Zihan/original_data/test_acc_predict.csv"
    test_df = pd.read_csv(test_path)
    
    # 处理训练数据
    train_accounts = set(train_df['account'])
    df_train = df[df['account'].isin(train_accounts)].copy()
    df_train = df_train.merge(train_df[['account', 'flag']], on='account', how='inner')
    df_train['label'] = df_train['flag']
    
    # 处理测试数据
    test_accounts = set(test_df['account'])
    df_test = df[df['account'].isin(test_accounts)].copy()
    
    print(f"训练数据: {df_train.shape}")
    print(f"测试数据: {df_test.shape}")
    print(f"标签分布: {np.bincount(df_train['label'])}")
    
    # 2. 特征选择 - 故意选择很少且可能不重要的特征
    print(f"\n🎯 故意选择少量特征来降低模型性能...")
    
    time_cols = ['first_transaction_time', 'last_transaction_time']
    all_features = [col for col in df.columns if col not in ['account'] + time_cols]
    
    # 随机选择4个特征，故意制造信息不足
    np.random.seed(42)
    bad_features = np.random.choice(all_features, size=min(4, len(all_features)), replace=False).tolist()
    
    print(f"选择的'差'特征: {bad_features}")
    
    # 3. 数据预处理 - 故意简化
    X_train = df_train[bad_features].copy()
    X_test = df_test[bad_features].copy()
    y_train = df_train['label'].values
    
    # 简单处理缺失值
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_train.mean())
    
    # 简单标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"特征处理后形状: 训练{X_train_scaled.shape}, 测试{X_test_scaled.shape}")
    
    # 4. 创建8个候选差模型
    print(f"\n🤖 创建8个候选差模型...")
    
    # 定义模型类
    class RandomModel:
        def __init__(self, bad_ratio=0.05, random_seed=456):
            self.bad_ratio = bad_ratio
            self.random_seed = random_seed
            
        def fit(self, X, y):
            pass
            
        def predict(self, X):
            n_samples = len(X)
            np.random.seed(self.random_seed)
            predictions = np.random.choice([0, 1], size=n_samples, p=[1-self.bad_ratio, self.bad_ratio])
            return predictions
    
    class AlwaysGoodModel:
        def fit(self, X, y): pass
        def predict(self, X): return np.zeros(len(X), dtype=int)
    
    class AlwaysBadModel:
        def fit(self, X, y): pass
        def predict(self, X): return np.ones(len(X), dtype=int)
    
    # 创建候选模型
    candidate_models = {}
    candidate_models['over_regularized'] = LogisticRegression(C=0.0001, max_iter=5, solver='liblinear', random_state=42)
    candidate_models['underfit'] = LogisticRegression(C=100, max_iter=3, solver='liblinear', random_state=123)
    candidate_models['random_conservative'] = RandomModel(bad_ratio=0.05, random_seed=456)
    candidate_models['random_aggressive'] = RandomModel(bad_ratio=0.25, random_seed=789)
    candidate_models['minimal_features'] = LogisticRegression(C=1.0, max_iter=20, solver='liblinear', random_state=999)
    candidate_models['extreme_overfit'] = LogisticRegression(C=10000, max_iter=2, solver='liblinear', random_state=555)
    candidate_models['always_good'] = AlwaysGoodModel()
    candidate_models['always_bad'] = AlwaysBadModel()
    
    # 5. 5折交叉验证评估所有候选模型
    print(f"\n📊 5折交叉验证评估候选模型...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidate_results = {}
    
    for model_name, model in candidate_models.items():
        print(f"\n评估候选模型: {model_name}")
        
        cv_weighted_f1 = []
        cv_macro_f1 = []
        cv_bad_f1 = []
        cv_accuracy = []
        cv_bad_recall = []
        cv_bad_precision = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
            X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # 训练模型
            model.fit(X_fold_train, y_fold_train)
            val_pred = model.predict(X_fold_val)
            
            # 计算各种F1
            weighted_f1 = f1_score(y_fold_val, val_pred, average='weighted')
            macro_f1 = f1_score(y_fold_val, val_pred, average='macro') if len(np.unique(val_pred)) > 1 else 0
            bad_f1 = f1_score(y_fold_val, val_pred, pos_label=1) if len(np.unique(val_pred)) > 1 and 1 in val_pred else 0
            accuracy = accuracy_score(y_fold_val, val_pred)
            
            # 坏客户指标
            bad_recall = recall_score(y_fold_val, val_pred, pos_label=1, zero_division=0)
            bad_precision = precision_score(y_fold_val, val_pred, pos_label=1, zero_division=0)
            
            cv_weighted_f1.append(weighted_f1)
            cv_macro_f1.append(macro_f1)
            cv_bad_f1.append(bad_f1)
            cv_accuracy.append(accuracy)
            cv_bad_recall.append(bad_recall)
            cv_bad_precision.append(bad_precision)
        
        candidate_results[model_name] = {
            'cv_weighted_f1': np.mean(cv_weighted_f1),
            'cv_macro_f1': np.mean(cv_macro_f1),
            'cv_bad_f1': np.mean(cv_bad_f1),
            'cv_accuracy': np.mean(cv_accuracy),
            'cv_bad_recall': np.mean(cv_bad_recall),
            'cv_bad_precision': np.mean(cv_bad_precision),
            'f1_scores': [np.mean(cv_weighted_f1), np.mean(cv_macro_f1), np.mean(cv_bad_f1)]
        }
        
        print(f"  加权F1: {np.mean(cv_weighted_f1):.4f}")
        print(f"  宏F1: {np.mean(cv_macro_f1):.4f}")  
        print(f"  坏客户F1: {np.mean(cv_bad_f1):.4f}")
    
    # 6. 选择三个F1差距最大的模型
    print(f"\n🎯 选择三个F1差距最大的模型...")
    
    # 计算每个模型的三个F1之间的方差（差距）
    model_f1_variance = {}
    for model_name, result in candidate_results.items():
        f1_scores = result['f1_scores']
        variance = np.var(f1_scores)  # 方差越大，差距越大
        range_span = max(f1_scores) - min(f1_scores)  # 范围跨度
        model_f1_variance[model_name] = {
            'variance': variance,
            'range': range_span,
            'scores': f1_scores,
            'weighted_f1': result['cv_weighted_f1'],
            'macro_f1': result['cv_macro_f1'],
            'bad_f1': result['cv_bad_f1']
        }
    
    # 按F1差距排序，选择前3个
    sorted_models = sorted(model_f1_variance.items(), 
                          key=lambda x: x[1]['variance'], reverse=True)
    
    top3_models = sorted_models[:3]
    
    print(f"选择的3个F1差距最大的模型:")
    for i, (model_name, info) in enumerate(top3_models, 1):
        print(f"{i}. {model_name}:")
        print(f"   加权F1: {info['weighted_f1']:.4f}")
        print(f"   宏F1: {info['macro_f1']:.4f}")
        print(f"   坏客户F1: {info['bad_f1']:.4f}")
        print(f"   F1方差: {info['variance']:.4f}")
        print(f"   F1跨度: {info['range']:.4f}")
    
    # 7. 用选定的3个模型生成测试集预测
    print(f"\n🔮 生成测试集预测...")
    
    selected_models = {}
    
    for model_name, _ in top3_models:
        model = candidate_models[model_name]
        
        # 用全部训练数据训练
        model.fit(X_train_scaled, y_train)
        
        # 预测测试集
        test_pred = model.predict(X_test_scaled)
        
        selected_models[model_name] = {
            'model': model,
            'cv_results': candidate_results[model_name],
            'test_predictions': test_pred,
            'test_bad_ratio': np.mean(test_pred)
        }
        
        print(f"{model_name}: 测试集预测Bad客户 {np.sum(test_pred)} ({np.mean(test_pred)*100:.1f}%)")
    
    # 8. 详细对比分析
    print(f"\n" + "="*80)
    print("📊 最终选定的3个模型F1对比")
    print("="*80)
    
    print(f"{'模型':<20} {'加权F1':<12} {'宏F1':<12} {'坏F1':<12} {'F1方差':<12} {'测试Bad%':<10}")
    print("-" * 90)
    
    for model_name, result in selected_models.items():
        cv_results = result['cv_results']
        variance = model_f1_variance[model_name]['variance']
        print(f"{model_name:<20} "
              f"{cv_results['cv_weighted_f1']:<12.4f} "
              f"{cv_results['cv_macro_f1']:<12.4f} "
              f"{cv_results['cv_bad_f1']:<12.4f} "
              f"{variance:<12.4f} "
              f"{result['test_bad_ratio']*100:<10.1f}%")
    
    # 9. 保存预测文件
    result_dir = "/Users/mannormal/4011/Qi Zihan/result_analysis/prediction_results"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = []
    
    for i, (model_name, result) in enumerate(selected_models.items(), 1):
        # 保存测试集预测
        pred_df = pd.DataFrame({
            'account': df_test['account'].values,
            'prediction': result['test_predictions']
        })
        
        cv_results = result['cv_results']
        filename = f"DIAGNOSTIC_MODEL_{i}_{model_name}_{timestamp}_wF1_{cv_results['cv_weighted_f1']:.4f}_mF1_{cv_results['cv_macro_f1']:.4f}_bF1_{cv_results['cv_bad_f1']:.4f}.csv"
        pred_df.to_csv(os.path.join(result_dir, filename), index=False)
        saved_files.append(filename)
        print(f"已保存模型{i}: {filename}")
    
    # 10. F1指标诊断结论
    print(f"\n" + "="*80)
    print("🔍 F1指标诊断实验结论")
    print("="*80)
    
    print(f"\n通过5折交叉验证，我们选择了3个F1差距最大的模型:")
    
    for i, (model_name, result) in enumerate(selected_models.items(), 1):
        cv_results = result['cv_results']
        variance_info = model_f1_variance[model_name]
        
        print(f"\n🏷️  模型{i}: {model_name}")
        print(f"   加权F1: {cv_results['cv_weighted_f1']:.4f}")
        print(f"   宏平均F1: {cv_results['cv_macro_f1']:.4f}")
        print(f"   坏客户F1: {cv_results['cv_bad_f1']:.4f}")
        print(f"   F1最大差距: {variance_info['range']:.4f}")
        print(f"   测试集预测Bad客户: {np.sum(result['test_predictions'])} ({result['test_bad_ratio']*100:.1f}%)")
    
    print(f"\n📋 诊断使用方法:")
    print(f"1. 用这3个预测文件测试你的评价系统")
    print(f"2. 看系统给出的分数最接近哪个F1指标")
    print(f"3. 如果系统分数:")
    print(f"   - 接近加权F1 → 系统用weighted average F1")
    print(f"   - 接近宏平均F1 → 系统用macro average F1")
    print(f"   - 接近坏客户F1 → 系统专注坏客户检测")
    
    return selected_models, saved_files

if __name__ == "__main__":
    results, files = create_bad_model_predictions()
    print(f"\n✅ F1诊断实验完成!")
    print(f"生成了3个F1差距最大的模型预测文件:")
    for i, file in enumerate(files, 1):
        print(f"  {i}. {file}")
    print(f"\n🧪 现在可以用这些文件测试你的评价系统，诊断使用的是哪种F1指标!")