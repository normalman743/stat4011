import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

print("=== ULTRA Multi-Strategy Ensemble System with Meta-ANN ===")

# =====================================================
# 数据加载函数（保持不变）
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
# 模型训练函数（保持不变，返回预测）
# =====================================================
def train_universal_ensemble(data, n_models=50):
    print(f"\n=== Training Universal Ensemble ({n_models} models) ===")
    data_copy = data.copy()
    data_copy.loc[data_copy['flag'] == -1, 'flag'] = 0
    feature_cols = [col for col in data_copy.columns 
                   if col not in ['account', 'flag', 'account_type'] and not col.endswith('_category')]
    
    good_accounts = len(data_copy[data_copy['flag'] == 1])
    bad_accounts = len(data_copy[data_copy['flag'] == 0])
    sample_size = min(good_accounts, bad_accounts)
    
    predictions = []
    cv_scores = []
    X_all = data_copy[feature_cols].values
    y_all = data_copy['flag'].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i in tqdm(range(n_models), desc="Universal Models"):
        good_sample = data_copy[data_copy['flag'] == 1].sample(n=sample_size, replace=True, random_state=i)
        bad_sample = data_copy[data_copy['flag'] == 0].sample(n=sample_size, replace=True, random_state=i+1000)
        train_data = pd.concat([good_sample, bad_sample], axis=0)
        
        X_train = train_data[feature_cols].values
        y_train = train_data['flag'].values
        
        clf = RandomForestClassifier(
            n_estimators=100, 
            max_depth=15, 
            min_samples_split=10,
            random_state=i
        )
        clf.fit(X_train, y_train)
        
        cv_score = np.mean([clf.score(X_all[train_idx], y_all[train_idx]) 
                           for train_idx, val_idx in skf.split(X_all, y_all)])
        cv_scores.append(cv_score)
        
        y_pred = clf.predict_proba(X_all)[:, 1]  # 概率预测更适合做 meta-feature
        predictions.append(y_pred)
    
    return np.array(predictions), cv_scores

def train_strategy_ensemble(data, strategy_name, strategy_categories, n_models=10):
    print(f"\n=== Training {strategy_name.upper()} Strategy Ensemble ({n_models} models) ===")
    data_with_strategy = data.merge(strategy_categories, on='account', how='left')
    strategy_col = f"{strategy_name}_category"
    data_with_strategy[strategy_col] = data_with_strategy[strategy_col].fillna('unknown')
    
    data_copy = data_with_strategy.copy()
    data_copy.loc[data_copy['flag'] == -1, 'flag'] = 0
    
    feature_cols = [col for col in data_copy.columns if col not in ['account', 'flag', 'account_type']]
    strategy_dummies = pd.get_dummies(data_copy[strategy_col], prefix=strategy_name)
    feature_data = pd.concat([
        data_copy[[col for col in feature_cols if not col.endswith('_category')]],
        strategy_dummies
    ], axis=1)
    
    good_accounts = len(data_copy[data_copy['flag'] == 1])
    bad_accounts = len(data_copy[data_copy['flag'] == 0])
    sample_size = min(good_accounts, bad_accounts)
    
    predictions = []
    cv_scores = []
    X_all = feature_data.values
    y_all = data_copy['flag'].values
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for i in tqdm(range(n_models), desc=f"{strategy_name} Models"):
        good_sample = data_copy[data_copy['flag'] == 1].sample(n=sample_size, replace=True, random_state=i*100)
        bad_sample = data_copy[data_copy['flag'] == 0].sample(n=sample_size, replace=True, random_state=i*100+50)
        sample_indices = list(good_sample.index) + list(bad_sample.index)
        
        X_train = feature_data.loc[sample_indices].values
        y_train = pd.concat([good_sample, bad_sample])['flag'].values
        
        clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=18,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=i*10
        )
        clf.fit(X_train, y_train)
        
        cv_score = np.mean([metrics.f1_score(y_all[val_idx], clf.predict(X_all[val_idx]), zero_division=0) 
                           for train_idx, val_idx in skf.split(X_all, y_all)])
        cv_scores.append(cv_score)
        
        y_pred = clf.predict_proba(X_all)[:, 1]  # 概率
        predictions.append(y_pred)
    
    return np.array(predictions), cv_scores

# =====================================================
# Meta-ANN (stacking 第二层)
# =====================================================
def ultra_ensemble_meta_ann(all_predictions, y_true):
    """
    用 ANN 作为 meta-classifier
    all_predictions: (n_models, n_samples)
    y_true: (n_samples,)
    """
    X_meta = all_predictions.T  # shape: (n_samples, n_models)
    
    ann = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42
    )
    ann.fit(X_meta, y_true)
    
    y_pred = ann.predict(X_meta)
    f1 = metrics.f1_score(y_true, y_pred, average='binary', zero_division=0)
    print(f"Meta-ANN Training F1: {f1:.4f}")
    
    return y_pred, ann

# =====================================================
# 主程序
# =====================================================
print("=== Loading Data ===")
features_path = '/Users/mannormal/4011/Qi Zihan/feature_extraction/generated_features/all_features.csv'
all_features_df = pd.read_csv(features_path)

pwd = '/Users/mannormal/4011/Qi Zihan/original_data/'
ta = pd.read_csv(pwd + 'train_acc.csv')
te = pd.read_csv(pwd + 'test_acc_predict.csv')
ta.loc[ta['flag'] == 0, 'flag'] = -1

strategy_data = load_strategy_categories()
training_df = pd.merge(all_features_df, ta[['account', 'flag']], on='account', how='inner')
training_df['account_type'] = training_df.apply(classify_account_type_original, axis=1)

print(f"\nTraining data: {training_df.shape}")

all_predictions = []
all_cv_scores = []

# Universal ensemble
universal_preds, universal_cv = train_universal_ensemble(training_df, n_models=50)
all_predictions.append(universal_preds)
all_cv_scores.append(universal_cv)

# 5 strategy ensembles
for strategy_name, strategy_categories in strategy_data.items():
    strategy_preds, strategy_cv = train_strategy_ensemble(
        training_df, strategy_name, strategy_categories, n_models=10
    )
    all_predictions.append(strategy_preds)
    all_cv_scores.append(strategy_cv)

# Combine all predictions
combined_predictions = np.vstack(all_predictions)
print(f"Total models: {combined_predictions.shape[0]}")

# Meta-ANN
y_true = np.where(training_df['flag'].values == -1, 0, 1)
final_predictions, meta_ann = ultra_ensemble_meta_ann(combined_predictions, y_true)

# Metrics
accuracy = metrics.accuracy_score(y_true, final_predictions)
f1_binary = metrics.f1_score(y_true, final_predictions, average='binary', zero_division=0)

print("\n=== Final Results with Meta-ANN ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Binary: {f1_binary:.4f}")
