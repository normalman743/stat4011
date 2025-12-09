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
import random

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        # MPS设备设置确定性（如果支持的话）
        torch.mps.manual_seed(seed)

set_seed(743)

print("=== ULTRA Multi-Strategy Ensemble System with Meta-ANN ===")

# =====================================================
# 数据加载函数（保持不变）
# =====================================================
def load_strategy_categories():
    strategy_paths = {
        'traditional': '/Users/mannormal/4011/Qi Zihan/v1/ƒclassification_strategies/traditional_4types/traditional_category_mapping.csv',
        'volume': '/Users/mannormal/4011/Qi Zihan/v1/classification_strategies/volume_based/volume_category_mapping.csv',
        'profit': '/Users/mannormal/4011/Qi Zihan/v1/classification_strategies/profit_based/profit_category_mapping.csv',
        'interaction': '/Users/mannormal/4011/Qi Zihan/v1/classification_strategies/interaction_based/interaction_category_mapping.csv',
        'behavior': '/Users/mannormal/4011/Qi Zihan/v1/classification_strategies/behavior_based/behavior_category_mapping.csv'
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
def classify_account_type_improved(row):
    # 计算前向后向交易强度
    forward_strength = (row['A_fprofit'] + row['B_fprofit']) / max(row['A_fsize'] + row['B_fsize'], 1)
    backward_strength = (row['A_bprofit'] + row['B_bprofit']) / max(row['A_bsize'] + row['B_bsize'], 1)
    
    # A/B类型偏好程度
    a_dominance = (row['A_fprofit'] + row['A_bprofit']) / max(row['A_fprofit'] + row['A_bprofit'] + row['B_fprofit'] + row['B_bprofit'], 1)
    
    # 网络中心性
    centrality_score = row['betweenness_centrality'] + row['eigenvector_centrality']
    
    # 活跃度
    activity_intensity = row['activity_intensity']
    
    if centrality_score > 0.001 and activity_intensity > 2:
        return 'type1'  # 核心枢纽节点
    elif a_dominance > 0.8 and forward_strength > backward_strength:
        return 'type2'  # A类主导的发送方
    elif a_dominance < 0.2 and backward_strength > forward_strength:
        return 'type3'  # B类主导的接收方  
    else:
        return 'type4'  # 混合交易类型

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
    
    print(f"   Balanced sampling: {sample_size} per class")
    print(f"   Features: {feature_data.shape[1]} (base: {len([col for col in feature_cols if not col.endswith('_category')])}, strategy: {len(strategy_dummies.columns)})")
    
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
            random_state=i*10,
            class_weight='balanced'
        )
        clf.fit(X_train, y_train)
        
        # 计算分类别F1分数
        cv_f1_overall = []
        cv_f1_good = []
        cv_f1_bad = []
        
        for train_idx, val_idx in skf.split(X_all, y_all):
            val_pred = clf.predict(X_all[val_idx])
            f1_overall = metrics.f1_score(y_all[val_idx], val_pred, zero_division=0)
            f1_good = metrics.f1_score(y_all[val_idx], val_pred, pos_label=1, zero_division=0)
            f1_bad = metrics.f1_score(y_all[val_idx], val_pred, pos_label=0, zero_division=0)
            
            cv_f1_overall.append(f1_overall)
            cv_f1_good.append(f1_good)
            cv_f1_bad.append(f1_bad)
        
        cv_score = np.mean(cv_f1_overall)
        cv_scores.append(cv_score)
        
        y_pred = clf.predict_proba(X_all)[:, 1]
        predictions.append(y_pred)
    
    predictions_array = np.array(predictions).T
    
    # 计算平均分类别F1
    avg_f1_overall = np.mean(cv_scores)
    print(f"   Average CV F1 (Overall): {avg_f1_overall:.4f}")
    
    return predictions_array, cv_scores

# =====================================================
# PyTorch Meta-ANN with Feature Scaling
# =====================================================
class MetaANN(nn.Module):
    def __init__(self, n_base, n_feat, hidden1=256, hidden2=128, hidden3=64, 
                 hidden4=64, hidden5=32, hidden6=32, dropout=0.4):
        super().__init__()
        # 可训练缩放参数
        self.a = nn.Parameter(torch.ones(n_feat))
        self.b = nn.Parameter(torch.zeros(n_feat))
        
        # 6层深度网络架构
        self.fc1 = nn.Linear(n_base + n_feat, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc4 = nn.Linear(hidden3, hidden4)
        self.bn4 = nn.BatchNorm1d(hidden4)
        self.dropout4 = nn.Dropout(dropout)
        
        self.fc5 = nn.Linear(hidden4, hidden5)
        self.bn5 = nn.BatchNorm1d(hidden5)
        self.dropout5 = nn.Dropout(dropout)
        
        self.fc6 = nn.Linear(hidden5, hidden6)
        self.bn6 = nn.BatchNorm1d(hidden6)
        self.dropout6 = nn.Dropout(dropout)
        
        self.out = nn.Linear(hidden6, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_base, x_feat):
        # 特征缩放
        x_feat_scaled = self.a * x_feat + self.b
        
        # 特征融合
        x = torch.cat([x_base, x_feat_scaled], dim=1)
        
        # 6层深度网络
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        x = torch.relu(self.bn5(self.fc5(x)))
        x = self.dropout5(x)
        
        x = torch.relu(self.bn6(self.fc6(x)))
        x = self.dropout6(x)
        
        x = self.sigmoid(self.out(x))
        return x

def train_pytorch_meta_ann(base_predictions, original_features, y_true, n_epochs=500, patience=30):
    """
    使用PyTorch训练Meta-ANN
    base_predictions: (n_samples, n_models) - 基础模型预测
    original_features: (n_samples, n_features) - 原始特征
    y_true: (n_samples,) - 真实标签
    """
    print(f"\n🤖 Training PyTorch Meta-ANN")
    print(f"Base predictions shape: {base_predictions.shape}")
    print(f"Original features shape: {original_features.shape}")
    
    # 标准化原始特征
    scaler = StandardScaler()
    original_features = scaler.fit_transform(original_features)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    print(f"Using device: {device}")
    
    # 转换为PyTorch张量
    X_base_tensor = torch.tensor(base_predictions, dtype=torch.float32).to(device)
    X_feat_tensor = torch.tensor(original_features, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_true.reshape(-1,1), dtype=torch.float32).to(device)
    
    # 创建模型
    model = MetaANN(
        n_base=base_predictions.shape[1], 
        n_feat=original_features.shape[1],
        hidden1=256,
        hidden2=128, 
        hidden3=64,
        hidden4=64,
        hidden5=32, 
        hidden6=32,
        dropout=0.4
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)

    # 交叉验证分割用于早停
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(base_predictions, y_true))[0]
    
    Xb_train, Xb_val = X_base_tensor[train_idx], X_base_tensor[val_idx]
    Xf_train, Xf_val = X_feat_tensor[train_idx], X_feat_tensor[val_idx]
    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
    
    best_val_f1 = 0
    patience_counter = 0
    train_f1_history = []
    val_f1_history = []
    
    print("\nEpoch | Train F1 | Val F1   | Good F1  | Bad F1   | LR       | Status")
    print("-" * 70)
    
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        y_pred_train = model(Xb_train, Xf_train)
        loss = criterion(y_pred_train, y_train)
        loss.backward()
        optimizer.step()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            y_pred_val = model(Xb_val, Xf_val)
            
            # 计算F1分数
            y_train_prob = model(Xb_train, Xf_train).cpu().numpy()
            y_val_prob = y_pred_val.cpu().numpy()
            
            train_pred = (y_train_prob > 0.5).astype(int).flatten()
            val_pred = (y_val_prob > 0.5).astype(int).flatten()
            
            # 整体F1和分类别F1
            train_f1 = metrics.f1_score(y_true[train_idx], train_pred, zero_division=0)
            val_f1 = metrics.f1_score(y_true[val_idx], val_pred, zero_division=0)
            
            # 分别计算Good类(1)和Bad类(0)的F1
            val_f1_good = metrics.f1_score(y_true[val_idx], val_pred, pos_label=1, zero_division=0)
            val_f1_bad = metrics.f1_score(y_true[val_idx], val_pred, pos_label=0, zero_division=0)
            
            train_f1_history.append(train_f1)
            val_f1_history.append(val_f1)
        
        # 学习率调度
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 早停检查
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            status = "✅ Best"
        else:
            patience_counter += 1
            status = f"⏳ {patience_counter}/{patience}"
        
        # 打印进度
        if epoch % 50 == 0 or patience_counter == 0:
            print(f"{epoch:5d} | {train_f1:8.4f} | {val_f1:8.4f} | {val_f1_good:8.4f} | {val_f1_bad:8.4f} | {current_lr:.2e} | {status}")
        
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch}")
            print(f"🏆 Best validation F1: {best_val_f1:.4f}")
            break
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    model.eval()
    
    # 在全部训练数据上预测
    with torch.no_grad():
        y_final_pred = model(X_base_tensor, X_feat_tensor).cpu().numpy()
        y_final_label = (y_final_pred > 0.5).astype(int).flatten()
        
        final_f1 = metrics.f1_score(y_true, y_final_label, zero_division=0)
        final_f1_good = metrics.f1_score(y_true, y_final_label, pos_label=1, zero_division=0)
        final_f1_bad = metrics.f1_score(y_true, y_final_label, pos_label=0, zero_division=0)
        final_acc = metrics.accuracy_score(y_true, y_final_label)
        
    print(f"\n📊 Meta-ANN Final Results:")
    print(f"   Accuracy: {final_acc:.4f}")
    print(f"   Overall F1: {final_f1:.4f}")
    print(f"   Good Class F1 (pos_label=1): {final_f1_good:.4f}")
    print(f"   Bad Class F1 (pos_label=0): {final_f1_bad:.4f}")
    print(f"   Best Val F1: {best_val_f1:.4f}")
    print(f"   Overfitting: {train_f1_history[-1] - best_val_f1:+.4f}")
    
    return y_final_pred, model, scaler, {
        'final_f1': final_f1,
        'final_f1_good': final_f1_good,
        'final_f1_bad': final_f1_bad,
        'final_acc': final_acc,
        'best_val_f1': best_val_f1,
        'train_f1_history': train_f1_history,
        'val_f1_history': val_f1_history
    }

# =====================================================
# Enhanced Random Forest Ensemble Training
# =====================================================
def train_enhanced_rf_ensemble(data, n_models=100):
    """训练增强的随机森林集成 - 优化版本"""
    print(f"\n🌳 Training Enhanced RF Ensemble ({n_models} models)")
    
    data_copy = data.copy()
    data_copy.loc[data_copy['flag'] == -1, 'flag'] = 0
    feature_cols = [col for col in data_copy.columns 
                   if col not in ['account', 'flag', 'account_type'] and not col.endswith('_category')]
    
    good_accounts = len(data_copy[data_copy['flag'] == 1])
    bad_accounts = len(data_copy[data_copy['flag'] == 0])
    sample_size = min(good_accounts, bad_accounts)
    
    print(f"   💡 Data Info:")
    print(f"      Good accounts: {good_accounts}")
    print(f"      Bad accounts: {bad_accounts}")
    print(f"      Balanced sampling: {sample_size} per class")
    print(f"      Features: {len(feature_cols)}")
    print(f"      Imbalance ratio: 1:{bad_accounts//good_accounts}")
    
    X_all = data_copy[feature_cols].values
    y_all = data_copy['flag'].values
    
    predictions = []
    cv_scores = []
    
    # 优化的随机森林配置 - 更深更复杂
    rf_configs = [
        {'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 8, 'min_samples_leaf': 3},
        {'n_estimators': 180, 'max_depth': 30, 'min_samples_split': 6, 'min_samples_leaf': 2},
        {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 10, 'min_samples_leaf': 4},
        {'n_estimators': 220, 'max_depth': 35, 'min_samples_split': 12, 'min_samples_leaf': 5},
    ]
    
    for i in tqdm(range(n_models), desc="RF Models"):
        # 更激进的采样策略 - 增加样本多样性
        bootstrap_ratio = 0.8 + 0.4 * np.random.random()  # 0.8-1.2倍采样
        actual_sample_size = int(sample_size * bootstrap_ratio)
        
        good_sample = data_copy[data_copy['flag'] == 1].sample(
            n=actual_sample_size, replace=True, random_state=i
        )
        bad_sample = data_copy[data_copy['flag'] == 0].sample(
            n=actual_sample_size, replace=True, random_state=i+3000
        )
        train_data = pd.concat([good_sample, bad_sample], axis=0)
        
        X_train = train_data[feature_cols].values
        y_train = train_data['flag'].values
        
        # 循环使用不同配置
        config = rf_configs[i % len(rf_configs)]
        
        clf = RandomForestClassifier(
            **config,
            random_state=i,
            class_weight='balanced_subsample',  # 更好的类平衡
            max_features='sqrt',  # 特征选择策略
            bootstrap=True,
            oob_score=True,
            n_jobs=1
        )
        clf.fit(X_train, y_train)
        
        # 使用Out-of-Bag评估 + 交叉验证
        oob_score = clf.oob_score_ if hasattr(clf, 'oob_score_') else 0
        
        # 5折交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        cv_f1_scores = []
        for train_idx, val_idx in skf.split(X_all, y_all):
            val_pred = clf.predict(X_all[val_idx])
            f1 = metrics.f1_score(y_all[val_idx], val_pred, zero_division=0)
            cv_f1_scores.append(f1)
        
        cv_score = np.mean(cv_f1_scores)
        cv_scores.append(cv_score)
        
        # 概率预测
        y_pred_proba = clf.predict_proba(X_all)[:, 1]
        predictions.append(y_pred_proba)
    
    predictions_array = np.array(predictions).T  # (n_samples, n_models)
    avg_cv_score = np.mean(cv_scores)
    
    print(f"   📊 Results:")
    print(f"      Average CV F1: {avg_cv_score:.4f}")
    print(f"      CV F1 std: {np.std(cv_scores):.4f}")
    print(f"      CV F1 range: [{np.min(cv_scores):.4f}, {np.max(cv_scores):.4f}]")
    
    return predictions_array, cv_scores, feature_cols

# =====================================================
# Meta-ANN (stacking 第二层) - 保持原有的sklearn版本作为对比
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
# 主程序 - Enhanced PyTorch Version
# =====================================================

def main():
    print("=== ULTRA Multi-Strategy Ensemble with PyTorch Meta-ANN ===")
    
    # 数据加载
    print("\n=== Loading Data ===")
    features_path = '/Users/mannormal/4011/Qi Zihan/v2/feature_extraction/result/feature.csv'
    all_features_df = pd.read_csv(features_path)

    pwd = '/Users/mannormal/4011/Qi Zihan/original_data/'
    ta = pd.read_csv(pwd + 'train_acc.csv')
    te = pd.read_csv(pwd + 'test_acc_predict.csv')
    
    # 添加调试信息
    print(f"训练数据列: {list(ta.columns)}")
    print(f"特征数据列: {list(all_features_df.columns)}")
    print(f"训练数据样本: {ta.head()}")
    
    # 检查flag列是否存在
    if 'flag' not in ta.columns:
        print("❌ 错误：train_acc.csv中没有flag列！")
        return
    
    ta.loc[ta['flag'] == 0, 'flag'] = -1

    strategy_data = load_strategy_categories()
    
    # 如果all_features_df中已有flag列，先删除避免冲突
    cols_to_drop = []
    if 'flag' in all_features_df.columns:
        cols_to_drop.append('flag')
    if 'data_type' in all_features_df.columns:  # 同时删除data_type列
        cols_to_drop.append('data_type')
    
    if cols_to_drop:
        print(f"⚠️  特征数据中的以下列将被删除: {cols_to_drop}")
        all_features_df = all_features_df.drop(cols_to_drop, axis=1)
    
    training_df = pd.merge(all_features_df, ta[['account', 'flag']], on='account', how='inner')
    
    # 再次检查合并结果
    print(f"合并后的列: {list(training_df.columns)}")
    print(f"合并后是否有flag列: {'flag' in training_df.columns}")
    
    if 'flag' not in training_df.columns:
        print("❌ 错误：合并后数据中没有flag列！")
        return

    training_df['account_type'] = training_df.apply(classify_account_type_improved, axis=1)

    print(f"Training data: {training_df.shape}")
    print(f"Account type distribution: {dict(training_df['account_type'].value_counts())}")
    print(f"Flag distribution: {dict(training_df['flag'].value_counts())}")
    
    # 准备原始特征
    feature_cols = [col for col in training_df.columns 
                   if col not in ['account', 'flag', 'account_type']]
    original_features = training_df[feature_cols].values
    y_true = np.where(training_df['flag'].values == -1, 0, 1)
    
    print(f"Original features shape: {original_features.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y_true, return_counts=True)))}")

    # =====================================================
    # Phase 1: 增强随机森林集成
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 PHASE 1: Enhanced Random Forest Ensemble")
    print(f"{'='*80}")
    
    rf_predictions, rf_cv_scores, rf_feature_names = train_enhanced_rf_ensemble(
        training_df, n_models=100
    )
    
    # =====================================================
    # Phase 2: 策略集成
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 PHASE 2: Strategy-Based Ensembles")
    print(f"{'='*80}")
    
    all_strategy_predictions = []
    strategy_results = {}
    
    for strategy_name, strategy_categories in strategy_data.items():
        print(f"\n--- {strategy_name.upper()} Strategy ---")
        strategy_preds, strategy_cv = train_strategy_ensemble(
            training_df, strategy_name, strategy_categories, n_models=20
        )
        all_strategy_predictions.append(strategy_preds)
        strategy_results[strategy_name] = {
            'predictions': strategy_preds,
            'cv_scores': strategy_cv,
            'avg_cv': np.mean(strategy_cv)
        }
        print(f"   Average CV F1: {np.mean(strategy_cv):.4f}")
    
    # 合并所有预测
    print(f"\n📊 Combining Predictions:")
    print(f"   RF predictions: {rf_predictions.shape}")
    for i, strategy_name in enumerate(strategy_data.keys()):
        print(f"   {strategy_name} predictions: {all_strategy_predictions[i].shape}")
    
    combined_base_predictions = np.hstack([rf_predictions] + all_strategy_predictions)
    
    print(f"   📊 Combined base predictions: {combined_base_predictions.shape}")
    print(f"   📊 Total models: {combined_base_predictions.shape[1]} (100 RF + {combined_base_predictions.shape[1]-100} Strategy)")
    
    # =====================================================
    # Phase 3: PyTorch Meta-ANN Training
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 PHASE 3: PyTorch Meta-ANN Training")
    print(f"{'='*80}")
    
    meta_predictions, meta_model, feature_scaler, meta_results = train_pytorch_meta_ann(
        base_predictions=combined_base_predictions,
        original_features=original_features,
        y_true=y_true,
        n_epochs=500,
        patience=30
    )
    
    # =====================================================
    # Phase 4: 详细结果分析与交叉验证
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 PHASE 4: Comprehensive Cross-Validation Analysis")
    print(f"{'='*80}")
    
    # 5折交叉验证评估
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    
    print("\nFold | Train F1 | Val F1   | Good F1  | Bad F1   | Train Acc| Val Acc  | Overfitting")
    print("-" * 80)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(combined_base_predictions, y_true)):
        # 分割数据
        X_base_train = combined_base_predictions[train_idx]
        X_base_val = combined_base_predictions[val_idx]
        X_feat_train = original_features[train_idx]
        X_feat_val = original_features[val_idx]
        y_train_fold = y_true[train_idx]
        y_val_fold = y_true[val_idx]
        
        # 特征缩放
        scaler_fold = StandardScaler()
        X_feat_train_scaled = scaler_fold.fit_transform(X_feat_train)
        X_feat_val_scaled = scaler_fold.transform(X_feat_val)
        
        # 训练Meta-ANN
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_fold = MetaANN(
            n_base=X_base_train.shape[1], 
            n_feat=X_feat_train_scaled.shape[1],
            hidden1=256,
            hidden2=128, 
            hidden3=64,
            hidden4=64,
            hidden5=32, 
            hidden6=32,
            dropout=0.4
        ).to(device)
        
        optimizer = optim.AdamW(model_fold.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCELoss()
        
        # 转换为张量
        X_base_train_t = torch.tensor(X_base_train, dtype=torch.float32).to(device)
        X_feat_train_t = torch.tensor(X_feat_train_scaled, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train_fold.reshape(-1,1), dtype=torch.float32).to(device)
        
        X_base_val_t = torch.tensor(X_base_val, dtype=torch.float32).to(device)
        X_feat_val_t = torch.tensor(X_feat_val_scaled, dtype=torch.float32).to(device)
        
        # 快速训练（少epochs用于CV）
        for epoch in range(100):
            model_fold.train()
            optimizer.zero_grad()
            y_pred = model_fold(X_base_train_t, X_feat_train_t)
            loss = criterion(y_pred, y_train_t)
            loss.backward()
            optimizer.step()
        
        # 评估
        model_fold.eval()
        with torch.no_grad():
            train_pred = model_fold(X_base_train_t, X_feat_train_t).cpu().numpy()
            val_pred = model_fold(X_base_val_t, X_feat_val_t).cpu().numpy()
            
            train_label = (train_pred > 0.5).astype(int).flatten()
            val_label = (val_pred > 0.5).astype(int).flatten()
            
            train_f1 = metrics.f1_score(y_train_fold, train_label, zero_division=0)
            val_f1 = metrics.f1_score(y_val_fold, val_label, zero_division=0)
            val_f1_good = metrics.f1_score(y_val_fold, val_label, pos_label=1, zero_division=0)
            val_f1_bad = metrics.f1_score(y_val_fold, val_label, pos_label=0, zero_division=0)
            train_acc = metrics.accuracy_score(y_train_fold, train_label)
            val_acc = metrics.accuracy_score(y_val_fold, val_label)
            
            overfitting = train_f1 - val_f1
            overfit_status = "🔴 High" if overfitting > 0.1 else "🟡 Med" if overfitting > 0.05 else "🟢 Low"
            
            print(f"{fold+1:4d} | {train_f1:8.4f} | {val_f1:8.4f} | {val_f1_good:8.4f} | {val_f1_bad:8.4f} | {train_acc:8.4f} | {val_acc:8.4f} | {overfit_status}")
            
            cv_results.append({
                'fold': fold + 1,
                'train_f1': train_f1,
                'val_f1': val_f1,
                'val_f1_good': val_f1_good,
                'val_f1_bad': val_f1_bad,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'overfitting': overfitting
            })
    
    # CV统计
    avg_train_f1 = np.mean([r['train_f1'] for r in cv_results])
    avg_val_f1 = np.mean([r['val_f1'] for r in cv_results])
    avg_val_f1_good = np.mean([r['val_f1_good'] for r in cv_results])
    avg_val_f1_bad = np.mean([r['val_f1_bad'] for r in cv_results])
    avg_train_acc = np.mean([r['train_acc'] for r in cv_results])
    avg_val_acc = np.mean([r['val_acc'] for r in cv_results])
    avg_overfitting = avg_train_f1 - avg_val_f1
    
    print("-" * 80)
    print(f"Avg  | {avg_train_f1:8.4f} | {avg_val_f1:8.4f} | {avg_val_f1_good:8.4f} | {avg_val_f1_bad:8.4f} | {avg_train_acc:8.4f} | {avg_val_acc:8.4f} | {avg_overfitting:+7.4f}")
    
    print(f"\n🤖 Meta-ANN Performance:")
    print(f"   Training F1 (Overall): {meta_results['final_f1']:.4f}")
    print(f"   Training F1 (Good Class): {meta_results['final_f1_good']:.4f}")
    print(f"   Training F1 (Bad Class): {meta_results['final_f1_bad']:.4f}")
    print(f"   Cross-Validation F1 (Overall): {avg_val_f1:.4f}")
    print(f"   Cross-Validation F1 (Good Class): {avg_val_f1_good:.4f}")
    print(f"   Cross-Validation F1 (Bad Class): {avg_val_f1_bad:.4f}")
    print(f"   Generalization Gap: {avg_overfitting:+.4f}")
    
    # =====================================================
    # 最终结果汇总
    # =====================================================
    print(f"\n{'='*80}")
    print("🏆 FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n📊 Base Models Performance:")
    print(f"   Enhanced RF Ensemble: {np.mean(rf_cv_scores):.4f} F1")
    for strategy_name, results in strategy_results.items():
        print(f"   {strategy_name.capitalize()} Strategy: {results['avg_cv']:.4f} F1")
    
    print(f"\n🤖 Meta-ANN Performance:")
    print(f"   Training F1: {meta_results['final_f1']:.4f}")
    print(f"   Cross-Validation F1: {avg_val_f1:.4f}")
    print(f"   Generalization Gap: {avg_overfitting:+.4f}")
    
    if avg_overfitting > 0.1:
        print("   🔴 HIGH overfitting - consider regularization")
    elif avg_overfitting > 0.05:
        print("   🟡 MEDIUM overfitting - monitor closely")
    else:
        print("   🟢 LOW overfitting - good generalization")
    
    print(f"\n🎯 Model Architecture:")
    print(f"   Base models: {combined_base_predictions.shape[1]} (100 RF + {combined_base_predictions.shape[1]-100} Strategy)")
    print(f"   Original features: {original_features.shape[1]}")
    print(f"   Meta-ANN: {combined_base_predictions.shape[1]+original_features.shape[1]} → 128 → 64 → 64 → 32 → 32 → 1")

    # =====================================================
    # Phase 5: 测试集预测 & 生成提交文件
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 PHASE 5: Test Set Prediction & Submission Generation")
    print(f"{'='*80}")
    
    # 加载测试数据
    test_df = pd.merge(all_features_df, te[['account']], on='account', how='inner')
    test_df['account_type'] = test_df.apply(classify_account_type_improved, axis=1)
    
    print(f"📊 Test Data Info:")
    print(f"   Test accounts: {test_df.shape[0]}")
    print(f"   Test account type distribution: {dict(test_df['account_type'].value_counts())}")
    
    # =====================================================
    # 为测试集生成基础模型预测
    # =====================================================
    print(f"\n🔮 Generating Test Predictions...")
    
    # 1. RF预测 (重新训练所有RF模型)
    print("🌳 RF Ensemble Test Predictions...")
    test_rf_predictions = []
    
    data_copy = training_df.copy()
    data_copy.loc[data_copy['flag'] == -1, 'flag'] = 0
    rf_feature_cols = [col for col in data_copy.columns 
                      if col not in ['account', 'flag', 'account_type'] and not col.endswith('_category')]
    
    X_train_rf = data_copy[rf_feature_cols].values
    y_train_rf = data_copy['flag'].values
    X_test_rf = test_df[rf_feature_cols].values
    
    good_accounts = len(data_copy[data_copy['flag'] == 1])
    bad_accounts = len(data_copy[data_copy['flag'] == 0])
    sample_size = min(good_accounts, bad_accounts)
    
    rf_configs = [
        {'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 8, 'min_samples_leaf': 3},
        {'n_estimators': 180, 'max_depth': 30, 'min_samples_split': 6, 'min_samples_leaf': 2},
        {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 10, 'min_samples_leaf': 4},
        {'n_estimators': 220, 'max_depth': 35, 'min_samples_split': 12, 'min_samples_leaf': 5},
    ]
    
    for i in tqdm(range(100), desc="RF Test Prediction"):
        bootstrap_ratio = 0.8 + 0.4 * np.random.random()
        actual_sample_size = int(sample_size * bootstrap_ratio)
        
        good_sample = data_copy[data_copy['flag'] == 1].sample(
            n=actual_sample_size, replace=True, random_state=i
        )
        bad_sample = data_copy[data_copy['flag'] == 0].sample(
            n=actual_sample_size, replace=True, random_state=i+3000
        )
        train_data = pd.concat([good_sample, bad_sample], axis=0)
        
        X_train_rf_model = train_data[rf_feature_cols].values
        y_train_rf_model = train_data['flag'].values
        
        config = rf_configs[i % len(rf_configs)]
        clf = RandomForestClassifier(
            **config,
            random_state=i,
            class_weight='balanced_subsample',
            max_features='sqrt',
            bootstrap=True,
            n_jobs=1
        )
        clf.fit(X_train_rf_model, y_train_rf_model)
        
        test_pred = clf.predict_proba(X_test_rf)[:, 1]
        test_rf_predictions.append(test_pred)
    
    test_rf_predictions = np.array(test_rf_predictions).T
    print(f"   RF test predictions shape: {test_rf_predictions.shape}")
    
    # 2. 策略预测
    print("🎯 Strategy Ensemble Test Predictions...")
    test_strategy_predictions = []
    
    for strategy_name, strategy_categories in strategy_data.items():
        print(f"   Processing {strategy_name} strategy...")
        
        # 训练数据处理
        train_with_strategy = training_df.merge(strategy_categories, on='account', how='left')
        strategy_col = f"{strategy_name}_category"
        train_with_strategy[strategy_col] = train_with_strategy[strategy_col].fillna('unknown')
        train_copy = train_with_strategy.copy()
        train_copy.loc[train_copy['flag'] == -1, 'flag'] = 0
        
        # 测试数据处理
        test_with_strategy = test_df.merge(strategy_categories, on='account', how='left')
        test_with_strategy[strategy_col] = test_with_strategy[strategy_col].fillna('unknown')
        
        # 特征处理
        feature_cols_strategy = [col for col in train_copy.columns if col not in ['account', 'flag', 'account_type']]
        train_strategy_dummies = pd.get_dummies(train_copy[strategy_col], prefix=strategy_name)
        test_strategy_dummies = pd.get_dummies(test_with_strategy[strategy_col], prefix=strategy_name)
        
        # 确保训练和测试集有相同的列
        all_strategy_cols = set(train_strategy_dummies.columns) | set(test_strategy_dummies.columns)
        for col in all_strategy_cols:
            if col not in train_strategy_dummies.columns:
                train_strategy_dummies[col] = 0
            if col not in test_strategy_dummies.columns:
                test_strategy_dummies[col] = 0
        
        train_strategy_dummies = train_strategy_dummies[sorted(all_strategy_cols)]
        test_strategy_dummies = test_strategy_dummies[sorted(all_strategy_cols)]
        
        train_feature_data = pd.concat([
            train_copy[[col for col in feature_cols_strategy if not col.endswith('_category')]],
            train_strategy_dummies
        ], axis=1)
        
        test_feature_data = pd.concat([
            test_with_strategy[[col for col in train_copy.columns if col not in ['account', 'flag', 'account_type'] and not col.endswith('_category')]],
            test_strategy_dummies
        ], axis=1)
        
        # 训练模型并预测
        strategy_test_preds = []
        for i in range(20):
            good_sample = train_copy[train_copy['flag'] == 1].sample(n=sample_size, replace=True, random_state=i*100)
            bad_sample = train_copy[train_copy['flag'] == 0].sample(n=sample_size, replace=True, random_state=i*100+50)
            sample_indices = list(good_sample.index) + list(bad_sample.index)
            
            X_train_strategy = train_feature_data.loc[sample_indices].values
            y_train_strategy = pd.concat([good_sample, bad_sample])['flag'].values
            
            clf = RandomForestClassifier(
                n_estimators=120,
                max_depth=18,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=i*10,
                class_weight='balanced'
            )
            clf.fit(X_train_strategy, y_train_strategy)
            
            test_pred = clf.predict_proba(test_feature_data.values)[:, 1]
            strategy_test_preds.append(test_pred)
        
        strategy_test_preds = np.array(strategy_test_preds).T
        test_strategy_predictions.append(strategy_test_preds)
        print(f"   {strategy_name} test predictions shape: {strategy_test_preds.shape}")
    
    # 3. 合并测试集预测
    test_combined_base_predictions = np.hstack([test_rf_predictions] + test_strategy_predictions)
    print(f"📊 Combined test predictions shape: {test_combined_base_predictions.shape}")
    
    # 4. Meta-ANN测试预测
    print("🤖 Meta-ANN Test Prediction...")
    test_original_features = test_df[feature_cols].values
    test_original_features_scaled = feature_scaler.transform(test_original_features)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    X_test_base_tensor = torch.tensor(test_combined_base_predictions, dtype=torch.float32).to(device)
    X_test_feat_tensor = torch.tensor(test_original_features_scaled, dtype=torch.float32).to(device)
    
    meta_model.eval()
    with torch.no_grad():
        test_meta_predictions = meta_model(X_test_base_tensor, X_test_feat_tensor).cpu().numpy()
        test_final_labels = (test_meta_predictions > 0.5).astype(int).flatten()
    
    # =====================================================
    # 生成提交文件
    # =====================================================
    print(f"\n💾 Generating Submission File...")
    
    # 创建提交DataFrame
    submission_df = pd.DataFrame({
        'account': test_df['account'].values,
        'flag': test_final_labels
    })
    
    # 统计预测结果
    pred_counts = submission_df['flag'].value_counts()
    print(f"📊 Test Prediction Summary:")
    print(f"   Total test accounts: {len(submission_df)}")
    print(f"   Predicted Good (1): {pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/len(submission_df)*100:.1f}%)")
    print(f"   Predicted Bad (0): {pred_counts.get(0, 0)} ({pred_counts.get(0, 0)/len(submission_df)*100:.1f}%)")
    
    # 生成文件名
    mean_cv_f1 = avg_val_f1
    filename = f"ultra_meta_ann_mean_cv_f1_score_{mean_cv_f1:.4f}.csv"
    filepath = f"/Users/mannormal/4011/Qi Zihan/v2/results/{filename}"
    
    # 保存文件
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    submission_df.to_csv(filepath, index=False)
    
    print(f"✅ Submission file saved: {filename}")
    print(f"📁 Full path: {filepath}")
    
    return {
        'meta_model': meta_model,
        'feature_scaler': feature_scaler,
        'rf_predictions': rf_predictions,
        'strategy_predictions': all_strategy_predictions,
        'cv_results': cv_results,
        'meta_results': meta_results,
        'final_f1': avg_val_f1,
        'submission_df': submission_df,
        'submission_filepath': filepath
    }

if __name__ == "__main__":
    results = main()
    
    print(f"\n{'='*80}")
    print("✅ Enhanced PyTorch Meta-ANN Training Complete!")
    print(f"🎯 Cross-Validation F1: {results['final_f1']:.4f}")
    print(f"📁 Submission file: {results['submission_filepath']}")
    print(f"{'='*80}")
    
    # 显示最终的文件名
    filename_only = os.path.basename(results['submission_filepath'])
    print(f"\n🎉 SUBMISSION READY!")
    print(f"📄 File name: {filename_only}")
    print(f"🎯 Mean CV F1 Score: {results['final_f1']:.4f}")
    print(f"📊 Test predictions: {len(results['submission_df'])} accounts")
