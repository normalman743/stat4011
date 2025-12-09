import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

print("=== Category-Based Decision Tree Classification System ===")

# =====================================================
# 数据加载函数
# =====================================================
def load_strategy_categories():
    strategy_paths = {
        'traditional': '/Users/mannormal/4011/Qi Zihan/classification_strategies/traditional_4types/traditional_category_mapping.csv',
        'volume': '/Users/mannormal/4011/Qi Zihan/classification_strategies/volume_based/volume_category_mapping.csv',
        'profit': '/Users/mannormal/4011/Qi Zihan/classification_strategies/profit_based/profit_category_mapping.csv',
        'interaction': '/Users/mannormal/4011/Qi Zihan/classification_strategies/interaction_based/interaction_category_mapping.csv',
        'behavior': '/Users/mannormal/4011/Qi Zihan/classification_strategies/behavior_based/behavior_category_mapping.csv'
    }
    
    strategy_data = {}
    print("\n=== Loading Classification Strategies ===")
    for strategy_name, path in strategy_paths.items():
        if os.path.exists(path):
            strategy_data[strategy_name] = pd.read_csv(path)
            print(f"✅ {strategy_name}: {len(strategy_data[strategy_name])} accounts")
        else:
            print(f"❌ {strategy_name}: File not found")
    
    return strategy_data

def classify_account_type_original(row):
    has_forward = (row['normal_fprofit'] > 0 or row['abnormal_fprofit'] > 0 or 
                   row['normal_fsize'] > 0 or row['abnormal_fsize'] > 0)
    has_backward = (row['normal_bprofit'] > 0 or row['abnormal_bprofit'] > 0 or
                    row['normal_bsize'] > 0 or row['abnormal_bsize'] > 0)
    
    if has_forward and has_backward:
        return 'type1'
    elif has_forward and not has_backward:
        return 'type2'
    elif not has_forward and has_backward:
        return 'type3'
    else:
        return 'type4'

# =====================================================
# 类别决策树训练函数
# =====================================================
def train_category_decision_tree(data, category_name, category_data=None, n_folds=5):
    """
    按类别训练决策树，输出详细的交叉验证分数
    """
    print(f"\n{'='*60}")
    print(f"🌳 Training Decision Tree for {category_name.upper()}")
    print(f"{'='*60}")
    
    # 准备数据
    if category_data is not None:
        # 合并类别数据
        data_with_category = data.merge(category_data, on='account', how='left')
        category_col = f"{category_name}_category"
        data_with_category[category_col] = data_with_category[category_col].fillna('unknown')
        
        # 创建类别哑变量
        category_dummies = pd.get_dummies(data_with_category[category_col], prefix=category_name)
        feature_cols = [col for col in data.columns if col not in ['account', 'flag', 'account_type']]
        
        # 合并特征
        X = pd.concat([
            data_with_category[feature_cols],
            category_dummies
        ], axis=1)
        
        print(f"📊 Features: {len(feature_cols)} original + {len(category_dummies.columns)} category = {X.shape[1]} total")
    else:
        # 仅使用原始特征（用于account_type分类）
        feature_cols = [col for col in data.columns if col not in ['account', 'flag']]
        X = data[feature_cols]
        print(f"📊 Features: {X.shape[1]} original features")
    
    # 准备标签
    y = data['flag'].copy()
    y = np.where(y == -1, 0, 1)  # 转换为0/1
    
    print(f"📈 Data shape: {X.shape}")
    print(f"📊 Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # 检查类别平衡
    good_count = np.sum(y == 1)
    bad_count = np.sum(y == 0)
    if good_count == 0 or bad_count == 0:
        print("⚠️  Warning: Imbalanced data (missing one class)")
        return None, None, None
    
    # 设置决策树参数
    dt_params = {
        'criterion': 'gini',
        'max_depth': 15,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'random_state': 42,
        'class_weight': 'balanced'  # 处理不平衡数据
    }
    
    # 交叉验证
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    train_scores = []
    test_scores = []
    f1_train_scores = []
    f1_test_scores = []
    
    print(f"\n🔄 Cross-Validation ({n_folds} folds):")
    print("Fold | Train Acc | Test Acc | Train F1 | Test F1  | Overfitting")
    print("-" * 65)
    
    fold_models = []
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # 分割数据
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练决策树
        dt = DecisionTreeClassifier(**dt_params)
        dt.fit(X_train, y_train)
        
        # 预测
        y_train_pred = dt.predict(X_train)
        y_test_pred = dt.predict(X_test)
        
        # 计算分数
        train_acc = metrics.accuracy_score(y_train, y_train_pred)
        test_acc = metrics.accuracy_score(y_test, y_test_pred)
        train_f1 = metrics.f1_score(y_train, y_train_pred, zero_division=0)
        test_f1 = metrics.f1_score(y_test, y_test_pred, zero_division=0)
        
        # 过拟合检测
        overfitting = train_acc - test_acc
        overfit_status = "🔴 High" if overfitting > 0.1 else "🟡 Medium" if overfitting > 0.05 else "🟢 Low"
        
        print(f"{fold+1:4d} | {train_acc:8.4f} | {test_acc:8.4f} | {train_f1:8.4f} | {test_f1:8.4f} | {overfit_status}")
        
        # 保存结果
        fold_results.append({
            'fold': fold + 1,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'overfitting': overfitting
        })
        
        train_scores.append(train_acc)
        test_scores.append(test_acc)
        f1_train_scores.append(train_f1)
        f1_test_scores.append(test_f1)
        fold_models.append(dt)
    
    # 计算平均分数
    avg_train_acc = np.mean(train_scores)
    avg_test_acc = np.mean(test_scores)
    avg_train_f1 = np.mean(f1_train_scores)
    avg_test_f1 = np.mean(f1_test_scores)
    avg_overfitting = avg_train_acc - avg_test_acc
    
    print("-" * 65)
    print(f"Avg  | {avg_train_acc:8.4f} | {avg_test_acc:8.4f} | {avg_train_f1:8.4f} | {avg_test_f1:8.4f} | {avg_overfitting:+7.4f}")
    print(f"Std  | {np.std(train_scores):8.4f} | {np.std(test_scores):8.4f} | {np.std(f1_train_scores):8.4f} | {np.std(f1_test_scores):8.4f} |")
    
    # 过拟合分析
    print(f"\n📈 Overfitting Analysis:")
    if avg_overfitting > 0.1:
        print("🔴 HIGH overfitting detected! Consider:")
        print("   - Reducing max_depth")
        print("   - Increasing min_samples_split/min_samples_leaf")
        print("   - Using pruning")
    elif avg_overfitting > 0.05:
        print("🟡 MEDIUM overfitting. Monitor closely.")
    else:
        print("🟢 LOW overfitting. Model generalizes well.")
    
    # 训练最终模型（全部数据）
    final_dt = DecisionTreeClassifier(**dt_params)
    final_dt.fit(X, y)
    
    # 返回结果
    results_summary = {
        'category': category_name,
        'avg_train_acc': avg_train_acc,
        'avg_test_acc': avg_test_acc,
        'avg_train_f1': avg_train_f1,
        'avg_test_f1': avg_test_f1,
        'overfitting': avg_overfitting,
        'fold_details': fold_results,
        'feature_count': X.shape[1],
        'sample_count': X.shape[0]
    }
    
    return final_dt, results_summary, X.columns.tolist()

# =====================================================
# 主程序
# =====================================================
def main():
    print("=== Loading Data ===")
    
    # 加载特征数据
    features_path = '/Users/mannormal/4011/Qi Zihan/feature_extraction/generated_features/all_features.csv'
    if not os.path.exists(features_path):
        print(f"❌ Error: {features_path} not found!")
        return
        
    all_features_df = pd.read_csv(features_path)
    print(f"✅ Features loaded: {all_features_df.shape}")
    
    # 加载训练数据
    pwd = '/Users/mannormal/4011/Qi Zihan/original_data/'
    ta = pd.read_csv(pwd + 'train_acc.csv')
    ta.loc[ta['flag'] == 0, 'flag'] = -1  # 转换标签
    
    # 合并数据
    training_df = pd.merge(all_features_df, ta[['account', 'flag']], on='account', how='inner')
    training_df['account_type'] = training_df.apply(classify_account_type_original, axis=1)
    
    print(f"✅ Training data ready: {training_df.shape}")
    print(f"📊 Account type distribution: {dict(training_df['account_type'].value_counts())}")
    print(f"📊 Flag distribution: {dict(training_df['flag'].value_counts())}")
    
    # 加载策略分类数据
    strategy_data = load_strategy_categories()
    
    # 存储所有结果
    all_results = []
    trained_models = {}
    
    # 1. 按账户类型训练决策树
    print(f"\n{'='*80}")
    print("🎯 PHASE 1: Training by Account Types")
    print(f"{'='*80}")
    
    for account_type in ['type1', 'type2', 'type3', 'type4']:
        type_data = training_df[training_df['account_type'] == account_type].copy()
        
        if len(type_data) < 20:  # 数据太少跳过
            print(f"⚠️  Skipping {account_type}: insufficient data ({len(type_data)} samples)")
            continue
            
        # 去除account_type列（用于训练）
        type_data_clean = type_data.drop(['account_type'], axis=1)
        
        model, results, feature_names = train_category_decision_tree(
            type_data_clean, 
            f"account_{account_type}",
            n_folds=5
        )
        
        if model is not None:
            trained_models[f"account_{account_type}"] = {
                'model': model,
                'feature_names': feature_names,
                'results': results
            }
            all_results.append(results)
    
    # 2. 按策略类别训练决策树
    print(f"\n{'='*80}")
    print("🎯 PHASE 2: Training by Strategy Categories")
    print(f"{'='*80}")
    
    for strategy_name, strategy_categories in strategy_data.items():
        model, results, feature_names = train_category_decision_tree(
            training_df, 
            strategy_name, 
            strategy_categories,
            n_folds=5
        )
        
        if model is not None:
            trained_models[f"strategy_{strategy_name}"] = {
                'model': model,
                'feature_names': feature_names,
                'results': results
            }
            all_results.append(results)
    
    # 3. 全数据训练（基准）
    print(f"\n{'='*80}")
    print("🎯 PHASE 3: Training on Full Dataset (Baseline)")
    print(f"{'='*80}")
    
    full_data = training_df.drop(['account_type'], axis=1)
    model, results, feature_names = train_category_decision_tree(
        full_data, 
        "full_dataset",
        n_folds=5
    )
    
    if model is not None:
        trained_models["full_dataset"] = {
            'model': model,
            'feature_names': feature_names,
            'results': results
        }
        all_results.append(results)
    
    # =====================================================
    # 结果分析
    # =====================================================
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE RESULTS ANALYSIS")
    print(f"{'='*80}")
    
    # 创建结果对比表
    print("\nCategory Performance Comparison:")
    print("-" * 100)
    print(f"{'Category':<25} | {'Samples':>8} | {'Features':>8} | {'Train F1':>8} | {'Test F1':>8} | {'Overfit':>8} | {'Status':>10}")
    print("-" * 100)
    
    for result in all_results:
        overfit_level = "HIGH" if result['overfitting'] > 0.1 else "MED" if result['overfitting'] > 0.05 else "LOW"
        print(f"{result['category']:<25} | {result['sample_count']:>8} | {result['feature_count']:>8} | "
              f"{result['avg_train_f1']:>8.4f} | {result['avg_test_f1']:>8.4f} | "
              f"{result['overfitting']:>+8.4f} | {overfit_level:>10}")
    
    print("-" * 100)
    
    # 找出最佳模型
    best_test_f1 = max(all_results, key=lambda x: x['avg_test_f1'])
    best_generalization = min(all_results, key=lambda x: x['overfitting'])
    
    print(f"\n🏆 Best Performance:")
    print(f"   Highest Test F1: {best_test_f1['category']} (F1={best_test_f1['avg_test_f1']:.4f})")
    print(f"   Best Generalization: {best_generalization['category']} (Overfit={best_generalization['overfitting']:+.4f})")
    
    # 保存结果
    results_df = pd.DataFrame(all_results)
    output_path = '/Users/mannormal/4011/Qi Zihan/result_analysis/category_decision_tree_results.csv'
    
    print(f"\n💾 Results Summary:")
    print(f"   Total models trained: {len(trained_models)}")
    print(f"   Results will be saved to: {output_path}")
    
    return trained_models, all_results, results_df, output_path

# =====================================================
# 运行主程序
# =====================================================
if __name__ == "__main__":
    trained_models, all_results, results_df, output_path = main()
    
    print(f"\n{'='*80}")
    print("✅ Category-Based Decision Tree Training Complete!")
    print(f"{'='*80}")
    
    # 改进建议
    print(f"\n🔧 IMPROVEMENT SUGGESTIONS:")
    print("=" * 50)
    
    print("1. 📈 Feature Engineering:")
    print("   - Add polynomial features for non-linear relationships")
    print("   - Create interaction features between categories")
    print("   - Use feature selection to reduce overfitting")
    
    print("\n2. 🌳 Model Improvements:")
    print("   - Try ensemble of decision trees (Random Forest)")
    print("   - Use gradient boosting (XGBoost, LightGBM)")
    print("   - Implement pruning for better generalization")
    
    print("\n3. 📊 Data Handling:")
    print("   - SMOTE for handling imbalanced classes")
    print("   - Stratified sampling within categories")
    print("   - Outlier detection and removal")
    
    print("\n4. 🎯 Category-Specific Optimization:")
    print("   - Different hyperparameters per category")
    print("   - Category-specific feature selection")
    print("   - Weighted voting based on category confidence")
    
    print("\n5. 🔍 Advanced Techniques:")
    print("   - Meta-learning across categories")
    print("   - Transfer learning from high-performing categories")
    print("   - Multi-task learning with shared representations")
