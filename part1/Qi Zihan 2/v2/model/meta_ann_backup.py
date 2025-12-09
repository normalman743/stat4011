import pandas as pd
import numpy as np
import os
import argparse
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
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings('ignore')

# Constants for consistent label handling
GOOD_LABEL = 0  # Good accounts
BAD_LABEL = 1   # Bad accounts (fraud)

@dataclass
class ModelConfig:
    """Configuration class for model hyperparameters"""
    n_rf_models: int = 200
    meta_ann_hidden: int = 128
    cv_folds: int = 5
    meta_ann_epochs: int = 500
    meta_ann_patience: int = 30
    meta_ann_dropout: float = 0.3
    meta_ann_lr: float = 1e-3
    meta_ann_weight_decay: float = 1e-4
    
class PathConfig:
    """Configuration for file paths"""
    def __init__(self, base_dir: str = '/Users/mannormal/4011/Qi Zihan'):
        self.base_dir = Path(base_dir)
        self.features_path = self.base_dir / 'v2/feature_extraction/result/features_cleaned_no_leakage1.csv'
        self.train_path = self.base_dir / 'original_data/train_acc.csv'
        self.test_path = self.base_dir / 'original_data/test_acc_predict.csv'
        self.results_dir = self.base_dir / 'v2/results'
        self.models_dir = self.base_dir / 'v2/models'
        self.strategy_base_dir = self.base_dir / 'v1/classification_strategies'
        
        # Create directories if they don't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        # MPS设备设置确定性（如果支持的话）
        torch.mps.manual_seed(seed)

seed_num = 13
set_seed(seed_num)

# Initialize configurations
CONFIG = ModelConfig()
PATHS = PathConfig()

print("=== ULTRA Multi-Strategy Ensemble System with Meta-ANN ===")

# =====================================================
# 数据加载函数（保持不变）
# =====================================================
def load_strategy_categories():
    strategy_paths = {
        'traditional': '/Users/mannormal/4011/Qi Zihan/v1/classification_strategies/traditional_4types/traditional_category_mapping.csv',
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
    
    # 网络活跃度 - 使用现有特征替代已删除的中心性特征
    network_activity = row['out_degree'] + row['in_degree'] + row['neighbor_count_1hop']
    
    # 活跃度
    activity_intensity = row['activity_intensity']
    
    # 优化后的阈值 - 基于数据分析结果
    if network_activity > 0.528 and activity_intensity > 0.00189:  # 75%分位数
        return 'type1'  # 核心枢纽节点
    elif a_dominance > 0.479 and forward_strength > backward_strength:  # 80%分位数
        return 'type2'  # A类主导的发送方
    elif a_dominance < 0.476 and backward_strength > forward_strength:  # 20%分位数
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
                           for train_idx, _ in skf.split(X_all, y_all)])
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
        # cv_f1_good = []  # Not used
        # cv_f1_bad = []   # Not used
        
        for _, val_idx in skf.split(X_all, y_all):
            val_pred = clf.predict(X_all[val_idx])
            f1_overall = metrics.f1_score(y_all[val_idx], val_pred, zero_division=0)
            # f1_good = metrics.f1_score(y_all[val_idx], val_pred, pos_label=1, zero_division=0)  # Not used
            # f1_bad = metrics.f1_score(y_all[val_idx], val_pred, pos_label=0, zero_division=0)   # Not used
            
            cv_f1_overall.append(f1_overall)
            # cv_f1_good.append(f1_good)  # Not used
            # cv_f1_bad.append(f1_bad)    # Not used
        
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
# PyTorch Meta-ANN with ResNet Connections - 修改点1
# =====================================================
class MetaANN(nn.Module):
    def __init__(self, n_base, n_feat, hidden=128, dropout=0.3):
        super().__init__()
        self.a = nn.Parameter(torch.ones(n_feat))
        self.b = nn.Parameter(torch.zeros(n_feat))
        
        self.input_dim = n_base + n_feat
        self.input_proj = nn.Linear(self.input_dim, hidden)
        
        # ResNet blocks
        self.res_block1 = self._make_res_block(hidden, dropout)
        self.res_block2 = self._make_res_block(hidden, dropout)
        self.res_block3 = self._make_res_block(hidden, dropout)
        
        self.out = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()
    
    def _make_res_block(self, dim, dropout):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
    
    def forward(self, x_base, x_feat):
        # 特征缩放
        x_feat_scaled = self.a * x_feat + self.b
        
        # 特征融合
        x = torch.cat([x_base, x_feat_scaled], dim=1)
        x = torch.relu(self.input_proj(x))
        
        # 残差连接
        residual = x
        x = residual + self.res_block1(x)
        x = torch.relu(x)
        
        residual = x
        x = residual + self.res_block2(x)
        x = torch.relu(x)
        
        residual = x
        x = residual + self.res_block3(x)
        x = torch.relu(x)
        
        return self.sigmoid(self.out(x))

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
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 转换为PyTorch张量
    X_base_tensor = torch.tensor(base_predictions, dtype=torch.float32).to(device)
    X_feat_tensor = torch.tensor(original_features, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_true.reshape(-1,1), dtype=torch.float32).to(device)
    
    # 创建模型
    model = MetaANN(
        n_base=base_predictions.shape[1], 
        n_feat=original_features.shape[1],
        hidden=128,
        dropout=0.3
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
        # oob_score = clf.oob_score_ if hasattr(clf, 'oob_score_') else 0  # Not used
        
        # 5折交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=i)
        cv_f1_scores = []
        for _, val_idx in skf.split(X_all, y_all):
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
# Strategy-Specific RF Ensemble Training (200 models distributed)
# =====================================================
def train_strategy_specific_rf_ensemble(data, strategy_data, test_data=None):
    """训练按分类策略分配的专用RF集成 - 200个模型分布式训练，同时生成测试预测"""
    print(f"\n🎯 Training Strategy-Specific RF Ensemble (200 models distributed)")
    if test_data is not None:
        print(f"   📊 Also generating test predictions during training")
    
    # RF分配策略
    rf_allocation = {
        'account_type': {'type4': 25, 'type3': 15, 'type1': 10, 'type2': 10},  # 60个
        'traditional': {'isolated': 15, 'backward_only': 10, 'both_directions': 8, 'forward_only': 7},  # 40个
        'interaction': {'B_in_B_out': 12, 'A_in_B_in_B_out': 8, 'A_in_A_out_B_in_B_out': 6, 
                       'A_out_B_in_B_out': 5, 'B_in': 2, 'small_categories': 2},  # 35个
        'behavior': {'inactive': 12, 'low_activity': 10, 'medium_activity_unidirectional': 5, 
                    'medium_activity_bidirectional': 3},  # 30个
        'volume': {'no_transactions': 8, 'medium_volume': 6, 'low_volume': 4, 'high_volume': 2},  # 20个
        'profit': {'loss_or_zero': 8, 'very_high_profit': 7}  # 15个
    }
    
    data_copy = data.copy()
    data_copy.loc[data_copy['flag'] == -1, 'flag'] = 0
    feature_cols = [col for col in data_copy.columns 
                   if col not in ['account', 'flag', 'account_type'] and not col.endswith('_category')]
    
    X_all = data_copy[feature_cols].values
    y_all = data_copy['flag'].values
    
    all_predictions = []
    all_test_predictions = []  # Store test predictions
    all_cv_scores = []
    model_count = 0
    
    # Prepare test data if provided
    if test_data is not None:
        test_data_copy = test_data.copy()
        X_test = test_data_copy[feature_cols].values
    
    print(f"   📊 Distribution Overview:")
    total_models = sum(sum(allocation.values()) for allocation in rf_allocation.values())
    print(f"      Total models planned: {total_models}")
    
    # 遍历每个分类策略
    for strategy_name, allocation in rf_allocation.items():
        print(f"\n   🔸 Training {strategy_name.upper()} strategy models...")
        
        if strategy_name == 'account_type':
            # 使用内置账户分类
            data_copy['current_strategy'] = data_copy.apply(classify_account_type_improved, axis=1)
        else:
            # 使用外部策略文件
            if strategy_name in strategy_data and not strategy_data[strategy_name].empty:
                strategy_df = strategy_data[strategy_name]
                strategy_mapping = dict(zip(strategy_df.iloc[:, 0], strategy_df.iloc[:, 1]))
                data_copy['current_strategy'] = data_copy['account'].map(strategy_mapping)
            else:
                print(f"      ⚠️  {strategy_name} strategy data not available, skipping...")
                continue
        
        # 为每个类别训练分配的RF模型
        for category, n_models in allocation.items():
            if category == 'small_categories':
                # 处理interaction策略的小类别
                if strategy_name == 'interaction':
                    small_cats = ['A_in_B_in', 'B_out', 'A_out_B_out', 'A_in_B_out', 
                                 'A_in_A_out_B_out', 'A_in_A_out_B_in', 'A_out_B_in', 'A_out']
                    category_data = data_copy[data_copy['current_strategy'].isin(small_cats)]
                else:
                    continue
            else:
                category_data = data_copy[data_copy['current_strategy'] == category]
            
            if len(category_data) < 10:  # 如果类别样本太少，跳过
                print(f"      ⚠️  {category}: Only {len(category_data)} samples, skipping...")
                continue
            
            print(f"      🎯 {category}: {len(category_data)} samples, {n_models} models")
            
            # 为当前类别训练n_models个RF
            if test_data is not None:
                category_predictions, category_test_predictions, category_cv_scores = train_category_specific_models(
                    category_data, data_copy, feature_cols, n_models, model_count, X_test
                )
                all_test_predictions.extend(category_test_predictions)
            else:
                category_predictions, category_cv_scores = train_category_specific_models(
                    category_data, data_copy, feature_cols, n_models, model_count
                )
            
            all_predictions.extend(category_predictions)
            all_cv_scores.extend(category_cv_scores)
            model_count += n_models
    
    # 转换为数组格式
    predictions_array = np.array(all_predictions).T  # (n_samples, n_models)
    
    print(f"\n   📊 Strategy-Specific Training Results:")
    print(f"      Total models trained: {len(all_predictions)}")
    print(f"      Average CV F1: {np.mean(all_cv_scores):.4f}")
    print(f"      CV F1 std: {np.std(all_cv_scores):.4f}")
    print(f"      CV F1 range: [{np.min(all_cv_scores):.4f}, {np.max(all_cv_scores):.4f}]")
    
    # 处理测试预测
    test_predictions_array = None
    if test_data is not None and len(all_test_predictions) > 0:
        test_predictions_array = np.array(all_test_predictions).T  # (n_test_samples, n_models)
        print(f"      Test predictions generated: {test_predictions_array.shape}")
    
    # 添加详细的F1评估
    if len(all_predictions) > 0:
        print(f"\n   🔍 Detailed F1 Evaluation on Training Data:")
        
        # 使用全部数据进行综合评估
        y_true = data_copy['flag'].values
        
        # 集成预测 (简单平均)
        ensemble_pred_proba = np.mean(predictions_array, axis=1)
        ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
        
        # 计算各种F1分数
        from sklearn.metrics import f1_score
        
        f1_bad = f1_score(y_true, ensemble_pred, pos_label=1)   # bad类F1 (标签1为bad)
        f1_good = f1_score(y_true, ensemble_pred, pos_label=0)  # good类F1 (标签0为good)
        f1_macro = f1_score(y_true, ensemble_pred, average='macro')
        f1_weighted = f1_score(y_true, ensemble_pred, average='weighted')
        
        print(f"      Bad F1 (pos_label=1): {f1_bad:.4f}")
        print(f"      Good F1 (pos_label=0): {f1_good:.4f}")
        print(f"      Macro F1: {f1_macro:.4f}")
        print(f"      Weighted F1: {f1_weighted:.4f}")
        
        # 过滤掉效果特别差的模型（F1 < 0.5）
        good_cv_scores = [score for score in all_cv_scores if score >= 0.5]
        if len(good_cv_scores) < len(all_cv_scores):
            print(f"      ⚠️  {len(all_cv_scores) - len(good_cv_scores)} models with F1 < 0.5 (possibly from small categories)")
            print(f"      Good models (F1≥0.5): {len(good_cv_scores)}, Avg F1: {np.mean(good_cv_scores):.4f}")
    else:
        print(f"      ⚠️  No models trained successfully")
    
    if test_data is not None:
        return predictions_array, test_predictions_array, all_cv_scores, feature_cols
    else:
        return predictions_array, all_cv_scores, feature_cols

def train_category_specific_models(category_data, full_data, feature_cols, n_models, base_seed, X_test=None):
    """为特定类别训练专用RF模型，同时生成测试预测"""
    
    # 检查类别数据是否有good/bad样本
    good_accounts = len(category_data[category_data['flag'] == 1])
    bad_accounts = len(category_data[category_data['flag'] == 0])
    
    if good_accounts == 0 or bad_accounts == 0:
        # 如果只有一个类别，使用全局数据进行平衡采样
        print(f"        ⚠️  Category has only one class, using global sampling")
        if X_test is not None:
            return [], [], []
        else:
            return [], []
    
    sample_size = min(good_accounts, bad_accounts, 500)  # 限制最大样本数
    
    # 用类别数据训练，但对全数据集预测
    X_category = category_data[feature_cols].values
    y_category = category_data['flag'].values
    X_all = full_data[feature_cols].values
    
    predictions = []
    test_predictions = []
    cv_scores = []
    
    # RF配置（针对类别优化）
    rf_configs = [
        {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2},
        {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 8, 'min_samples_leaf': 3},
        {'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 6, 'min_samples_leaf': 1},
    ]
    
    for i in range(n_models):
        current_seed = base_seed + i
        
        # 平衡采样
        good_sample = category_data[category_data['flag'] == 1].sample(
            n=sample_size, replace=True, random_state=current_seed
        )
        bad_sample = category_data[category_data['flag'] == 0].sample(
            n=sample_size, replace=True, random_state=current_seed + 5000
        )
        train_data = pd.concat([good_sample, bad_sample], axis=0)
        
        X_train = train_data[feature_cols].values
        y_train = train_data['flag'].values
        
        # 选择配置
        config = rf_configs[i % len(rf_configs)]
        
        clf = RandomForestClassifier(
            **config,
            random_state=current_seed,
            class_weight='balanced',
            max_features='sqrt',
            bootstrap=True,
            n_jobs=1
        )
        clf.fit(X_train, y_train)
        
        # 交叉验证评估（在类别数据上）
        if len(np.unique(y_category)) > 1:  # 确保有两个类别
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=current_seed)
            cv_f1_scores = []
            for _, val_idx in skf.split(X_category, y_category):
                if len(val_idx) > 0:
                    val_pred = clf.predict(X_category[val_idx])
                    f1 = metrics.f1_score(y_category[val_idx], val_pred, zero_division=0)
                    cv_f1_scores.append(f1)
            
            cv_score = np.mean(cv_f1_scores) if cv_f1_scores else 0
        else:
            cv_score = 0
        
        cv_scores.append(cv_score)
        
        # 概率预测（对全数据集）
        if hasattr(clf, 'predict_proba'):
            proba_pred = clf.predict_proba(X_all)
            y_pred_proba = proba_pred[:, 1] if proba_pred.shape[1] > 1 else proba_pred[:, 0]
        else:
            y_pred_proba = clf.predict(X_all).astype(float)
        
        predictions.append(y_pred_proba)
        
        # Generate test predictions if X_test is provided
        if X_test is not None:
            if hasattr(clf, 'predict_proba'):
                test_proba_pred = clf.predict_proba(X_test)
                test_pred_proba = test_proba_pred[:, 1] if test_proba_pred.shape[1] > 1 else test_proba_pred[:, 0]
            else:
                test_pred_proba = clf.predict(X_test).astype(float)
            test_predictions.append(test_pred_proba)
    
    if X_test is not None:
        return predictions, test_predictions, cv_scores
    else:
        return predictions, cv_scores

# =====================================================
# Note: generate_test_rf_predictions function removed
# Test predictions are now generated during training phase
# =====================================================

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
# 训练单个fold的Meta-ANN - 修改点2：新增函数
# =====================================================
def train_single_fold_meta_ann(X_base_train, X_feat_train, y_train, X_base_val, X_feat_val, y_val, 
                               fold_id, f1_type='bad', n_epochs=500, patience=30):
    """训练单个fold的Meta-ANN，使用早停"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 特征缩放
    scaler_fold = StandardScaler()
    X_feat_train_scaled = scaler_fold.fit_transform(X_feat_train)
    X_feat_val_scaled = scaler_fold.transform(X_feat_val)
    
    # 创建模型
    model_fold = MetaANN(
        n_base=X_base_train.shape[1], 
        n_feat=X_feat_train_scaled.shape[1],
        hidden=128,
        dropout=0.3
    ).to(device)
    
    optimizer = optim.AdamW(model_fold.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    
    # 转换为张量
    X_base_train_t = torch.tensor(X_base_train, dtype=torch.float32).to(device)
    X_feat_train_t = torch.tensor(X_feat_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32).to(device)
    
    X_base_val_t = torch.tensor(X_base_val, dtype=torch.float32).to(device)
    X_feat_val_t = torch.tensor(X_feat_val_scaled, dtype=torch.float32).to(device)
    
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    # 完整训练带早停
    for epoch in range(n_epochs):
        model_fold.train()
        optimizer.zero_grad()
        y_pred = model_fold(X_base_train_t, X_feat_train_t)
        loss = criterion(y_pred, y_train_t)
        loss.backward()
        optimizer.step()
        
        # 验证
        model_fold.eval()
        with torch.no_grad():
            val_pred_prob = model_fold(X_base_val_t, X_feat_val_t).cpu().numpy()
            val_pred_label = (val_pred_prob > 0.5).astype(int).flatten()
            
            # Calculate F1 based on f1_type  
            if f1_type == 'bad':
                val_f1 = metrics.f1_score(y_val, val_pred_label, pos_label=1, zero_division=0)  # bad=1
            elif f1_type == 'macro':
                val_f1 = metrics.f1_score(y_val, val_pred_label, average='macro', zero_division=0)
            elif f1_type == 'weighted':
                val_f1 = metrics.f1_score(y_val, val_pred_label, average='weighted', zero_division=0)
            else:  # default to 'bad'
                val_f1 = metrics.f1_score(y_val, val_pred_label, pos_label=1, zero_division=0)  # bad=1
        
        scheduler.step(val_f1)
        
        # 早停检查
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model_fold.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch}")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model_fold.load_state_dict(best_model_state)
    
    # 最终评估
    model_fold.eval()
    with torch.no_grad():
        train_pred = model_fold(X_base_train_t, X_feat_train_t).cpu().numpy()
        val_pred = model_fold(X_base_val_t, X_feat_val_t).cpu().numpy()
        
        train_label = (train_pred > 0.5).astype(int).flatten()
        val_label = (val_pred > 0.5).astype(int).flatten()
        
        # Calculate all f1 scores
        if f1_type == 'bad':
            train_f1 = metrics.f1_score(y_train, train_label, pos_label=1, zero_division=0)  # bad=1
            val_f1 = metrics.f1_score(y_val, val_label, pos_label=1, zero_division=0)       # bad=1
        elif f1_type == 'macro':
            train_f1 = metrics.f1_score(y_train, train_label, average='macro', zero_division=0)
            val_f1 = metrics.f1_score(y_val, val_label, average='macro', zero_division=0)
        elif f1_type == 'weighted':
            train_f1 = metrics.f1_score(y_train, train_label, average='weighted', zero_division=0)
            val_f1 = metrics.f1_score(y_val, val_label, average='weighted', zero_division=0)
        else:
            train_f1 = metrics.f1_score(y_train, train_label, pos_label=1, zero_division=0)  # bad=1
            val_f1 = metrics.f1_score(y_val, val_label, pos_label=1, zero_division=0)       # bad=1
            
        # Always calculate all types for output filename (统一标签定义：good=0, bad=1)
        val_f1_good = metrics.f1_score(y_val, val_label, pos_label=0, zero_division=0)  # good=0
        val_f1_bad = metrics.f1_score(y_val, val_label, pos_label=1, zero_division=0)   # bad=1
        val_f1_macro = metrics.f1_score(y_val, val_label, average='macro', zero_division=0)
        val_f1_weighted = metrics.f1_score(y_val, val_label, average='weighted', zero_division=0)
        
        train_acc = metrics.accuracy_score(y_train, train_label)
        val_acc = metrics.accuracy_score(y_val, val_label)
    
    return {
        'model': model_fold,
        'scaler': scaler_fold,
        'train_f1': train_f1,
        'val_f1': val_f1,
        'val_f1_good': val_f1_good,
        'val_f1_bad': val_f1_bad,
        'val_f1_macro': val_f1_macro,
        'val_f1_weighted': val_f1_weighted,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'overfitting': train_f1 - val_f1,
        'fold_id': fold_id
    }

# =====================================================
# 主程序 - Enhanced PyTorch Version
# =====================================================

def main(f1_type='bad'):
    print(f"=== ULTRA Multi-Strategy Ensemble with PyTorch Meta-ANN (F1 Type: {f1_type}) ===")
    print(f"f1_type :{f1_type} (可选：'bad', 'macro', 'weighted')")
    # 数据加载
    print("\n=== Loading Data ===")
    features_path = '/Users/mannormal/4011/Qi Zihan/v2/feature_extraction/result/features_cleaned_no_leakage1.csv'
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
    
    # 🎯 使用策略特定训练 (200个分布式模型) - 同时准备测试数据以生成测试预测
    print(f"🎯 Training Mode: Strategy-Specific (200 distributed models)")
    
    # 预先准备测试数据以便在训练时同步生成测试预测
    test_df = pd.merge(all_features_df, te[['account']], on='account', how='inner')
    test_df['account_type'] = test_df.apply(classify_account_type_improved, axis=1)
    
    rf_predictions, test_rf_predictions, rf_cv_scores, _ = train_strategy_specific_rf_ensemble(
        training_df, strategy_data, test_data=test_df
    )
    
    # =====================================================
    # Phase 2: 策略集成 (已包含在Phase 1中)
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 PHASE 2: Skipped (strategy-specific training completed in Phase 1)")
    print(f"{'='*80}")
    
    strategy_results = {}
    print(f"   Strategy-specific models already trained: {rf_predictions.shape[1] if len(rf_predictions.shape) > 1 else 0}")
    print(f"   Average CV F1: {np.mean(rf_cv_scores):.4f}")
    
    # 合并所有预测 (策略特定训练已包含所有模型)
    print(f"\n📊 Combining Predictions:")
    print(f"   Strategy-specific predictions: {rf_predictions.shape}")
    
    combined_base_predictions = rf_predictions
    print(f"   📊 Total models: {combined_base_predictions.shape[1]} (distributed across 6 strategies)")
    
    # =====================================================
    # Phase 3: PyTorch Meta-ANN Training (for reference only)
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 PHASE 3: PyTorch Meta-ANN Training (Reference)")
    print(f"{'='*80}")
    
    # Train reference model (results not used for final predictions)
    _, _, _, _ = train_pytorch_meta_ann(
        base_predictions=combined_base_predictions,
        original_features=original_features,
        y_true=y_true,
        n_epochs=500,
        patience=30
    )
    
    # =====================================================
    # Phase 4: CV训练并选择最佳模型 - 修改点3：主要修改
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 PHASE 4: Cross-Validation with Early Stopping & Best Model Selection")
    print(f"{'='*80}")
    
    # 5折交叉验证评估
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    fold_models = []
    
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
        
        # 训练Meta-ANN with early stopping
        fold_result = train_single_fold_meta_ann(
            X_base_train, X_feat_train, y_train_fold, 
            X_base_val, X_feat_val, y_val_fold, 
            fold_id=fold, f1_type=f1_type, n_epochs=500, patience=30
        )
        
        fold_models.append(fold_result)
        cv_results.append(fold_result)
        
        overfitting = fold_result['overfitting']
        overfit_status = "🔴 High" if overfitting > 0.1 else "🟡 Med" if overfitting > 0.05 else "🟢 Low"
        
        print(f"{fold+1:4d} | {fold_result['train_f1']:8.4f} | {fold_result['val_f1']:8.4f} | {fold_result['val_f1_good']:8.4f} | {fold_result['val_f1_bad']:8.4f} | {fold_result['train_acc']:8.4f} | {fold_result['val_acc']:8.4f} | {overfit_status}")
    
    # 选择最佳fold模型 - 修改点4：选择逻辑
    best_fold_idx = np.argmax([result['val_f1'] for result in cv_results])
    best_fold_model = fold_models[best_fold_idx]
    
    print(f"\n🏆 Best Fold: {best_fold_idx + 1} (Val F1: {best_fold_model['val_f1']:.4f})")
    
    # CV统计
    avg_train_f1 = np.mean([r['train_f1'] for r in cv_results])
    avg_val_f1 = np.mean([r['val_f1'] for r in cv_results])
    avg_val_f1_good = np.mean([r['val_f1_good'] for r in cv_results])
    avg_val_f1_bad = np.mean([r['val_f1_bad'] for r in cv_results])
    # Remove unused variables
    # avg_val_f1_macro = np.mean([r['val_f1_macro'] for r in cv_results])  
    # avg_val_f1_weighted = np.mean([r['val_f1_weighted'] for r in cv_results])
    avg_train_acc = np.mean([r['train_acc'] for r in cv_results])
    avg_val_acc = np.mean([r['val_acc'] for r in cv_results])
    avg_overfitting = avg_train_f1 - avg_val_f1
    
    print("-" * 80)
    print(f"Avg  | {avg_train_f1:8.4f} | {avg_val_f1:8.4f} | {avg_val_f1_good:8.4f} | {avg_val_f1_bad:8.4f} | {avg_train_acc:8.4f} | {avg_val_acc:8.4f} | {avg_overfitting:+7.4f}")
    
    print(f"\n🤖 Meta-ANN Performance (Using Best CV Fold):")
    print(f"   Best Fold Val F1: {best_fold_model['val_f1']:.4f}")
    print(f"   Average CV F1: {avg_val_f1:.4f}")
    print(f"   CV F1 std: {np.std([r['val_f1'] for r in cv_results]):.4f}")
    print(f"   Generalization Gap: {avg_overfitting:+.4f}")
    
    # =====================================================
    # 最终结果汇总
    # =====================================================
    print(f"\n{'='*80}")
    print("🏆 FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n📊 Base Models Performance:")
    print(f"   Enhanced RF Ensemble: {np.mean(rf_cv_scores):.4f} F1")
    # Strategy results are integrated into the RF ensemble above
    if strategy_results:  # Only print if there are separate strategy results
        for strategy_name, results in strategy_results.items():
            print(f"   {strategy_name.capitalize()} Strategy: {results['avg_cv']:.4f} F1")
    
    print(f"\n🤖 Meta-ANN Performance:")
    print(f"   Best CV Fold F1: {best_fold_model['val_f1']:.4f}")
    print(f"   Average CV F1: {avg_val_f1:.4f}")
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
    print(f"   Meta-ANN: ResNet-style with {combined_base_predictions.shape[1]+original_features.shape[1]} → 128 (3 res blocks) → 1")

    # =====================================================
    # Phase 5: 测试集预测 & 生成多个提交文件
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 PHASE 5: Test Set Prediction & Multiple Submission Generation")
    print(f"{'='*80}")
    
    print(f"📊 Test Data Info:")
    print(f"   Test accounts: {test_df.shape[0]}")
    print(f"   Test account type distribution: {dict(test_df['account_type'].value_counts())}")
    
    # =====================================================
    # 测试集RF预测已在训练阶段生成 (无数据泄漏)
    # =====================================================
    print(f"\n✅ Test Set RF Predictions (generated during training, no data leakage)")
    
    # 准备测试集特征
    feature_cols = [col for col in test_df.columns if col not in ['account', 'flag', 'account_type'] and not col.endswith('_category')]
    test_original_features = test_df[feature_cols].values
    
    # 使用训练时生成的测试预测
    test_combined_predictions = test_rf_predictions
    
    print(f"✅ Test RF predictions shape: {test_combined_predictions.shape}")
    print(f"   📊 Predictions generated during training phase - no redundant model training!")
    
    # =====================================================
    # 生成多个预测文件 (使用不同fold模型)
    # =====================================================
    print(f"\n🔮 Generating Multiple Test Predictions...")
    
    all_submissions = []
    
    # 使用前3个最好的fold模型生成预测
    sorted_folds = sorted(enumerate(cv_results), key=lambda x: x[1]['val_f1'], reverse=True)
    top_3_folds = sorted_folds[:3]
    
    print(f"   🏆 Using top 3 fold models:")
    for rank, (fold_idx, fold_result) in enumerate(top_3_folds, 1):
        print(f"   {rank}. Fold {fold_idx+1}: Val F1 = {fold_result['val_f1']:.4f}")
    
    for rank, (fold_idx, fold_result) in enumerate(top_3_folds, 1):
        print(f"\n   🔮 Generating predictions with Fold {fold_idx+1} model (rank {rank})...")
        
        try:
            # 使用该fold的模型进行预测
            fold_model_data = fold_models[fold_idx]
            model = fold_model_data['model']
            scaler = fold_model_data['scaler']
            
            # 准备测试数据 (匹配Meta-ANN的双输入架构)
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            
            # 1. 基础模型预测 (使用正确生成的测试集RF预测)
            # 修复数据泄漏：使用为测试集专门生成的RF预测
            test_base_features = test_combined_predictions
            
            # 2. 原始特征 
            test_original_features_scaled = scaler.transform(test_original_features)
            
            # 3. Meta-ANN预测 (双输入)
            model.eval()
            with torch.no_grad():
                X_base_tensor = torch.FloatTensor(test_base_features).to(device)
                X_feat_tensor = torch.FloatTensor(test_original_features_scaled).to(device)
                
                # Meta-ANN需要两个输入
                test_pred_proba = model(X_base_tensor, X_feat_tensor).cpu().numpy().flatten()
                test_pred_labels = (test_pred_proba > 0.5).astype(int)
            
            # 创建提交文件
            submission_df = pd.DataFrame({
                'account': test_df['account'].values,
                'flag': test_pred_labels
            })
            
            # 统计结果
            pred_counts = submission_df['flag'].value_counts()
            print(f"      Good (1): {pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/len(submission_df)*100:.1f}%)")
            print(f"      Bad (0): {pred_counts.get(0, 0)} ({pred_counts.get(0, 0)/len(submission_df)*100:.1f}%)")
            
            # 生成文件名
            fold_f1 = fold_result['val_f1']
            fold_f1_good = fold_result['val_f1_good'] 
            fold_f1_bad = fold_result['val_f1_bad']
            fold_f1_macro = fold_result['val_f1_macro']
            fold_f1_weighted = fold_result['val_f1_weighted']
            
            # 保持与之前一致的文件名格式，添加rank和fold信息
            random_seed = random.randint(100, 999)  # 生成随机数
            filename = f"ultra_resnet_meta_ann_rank{rank}_fold{fold_idx+1}_{f1_type}_f1_{fold_f1:.4f}_good_{fold_f1_good:.4f}_bad_{fold_f1_bad:.4f}_macro_{fold_f1_macro:.4f}_weighted_{fold_f1_weighted:.4f}_seed_{random_seed}.csv"
            filepath = f"/Users/mannormal/4011/Qi Zihan/v2/results/{filename}"
            
            # 保存文件
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            submission_df.to_csv(filepath, index=False)
            
            all_submissions.append({
                'rank': rank,
                'fold': fold_idx + 1,
                'val_f1': fold_f1,
                'filename': filename,
                'filepath': filepath,
                'submission_df': submission_df
            })
            
            print(f"      ✅ Saved: {filename}")
            
        except Exception as e:
            print(f"      ❌ Error generating prediction for Fold {fold_idx+1}: {str(e)}")
            print(f"      ⚠️  Skipping this fold and continuing...")
            continue
    
    if not all_submissions:
        raise RuntimeError("❌ Failed to generate any submission files!")
    
    print(f"\n🎯 Generated {len(all_submissions)} submission files:")
    for sub in all_submissions:
        print(f"   Rank {sub['rank']}: {sub['filename']} (Val F1: {sub['val_f1']:.4f})")
    
    # 设置最佳提交作为主要结果
    best_submission = all_submissions[0]
    submission_df = best_submission['submission_df']
    filepath = best_submission['filepath']
    
    # =====================================================
    # 返回结果汇总
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 TRAINING COMPLETED - RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"📊 Final Results:")
    print(f"   Best Validation F1: {best_submission['val_f1']:.4f}")
    print(f"   Total Fold Models: {len(cv_results)}")
    print(f"   Generated Submissions: {len(all_submissions)}")
    print(f"   Primary Submission: {best_submission['filename']}")
    
    return {
        'cv_results': cv_results,
        'best_submission': best_submission,
        'all_submissions': all_submissions,
        'rf_cv_scores': rf_cv_scores,
        'rf_predictions': rf_predictions
    }
    
if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Ultra ResNet Meta-ANN Training')
    parser.add_argument('--f1_type', type=str, default='bad', 
                        choices=['bad', 'macro', 'weighted'],
                        help='F1 score type for model selection (default: bad)')
    
    args = parser.parse_args()
    
    results = main(f1_type="weighted")
    
    print(f"\n{'='*80}")
    print("✅ Strategy-Specific Meta-ANN Training Complete!")
    print(f"🎯 RF Models Trained: {len(results['rf_cv_scores'])}")
    print(f"🎯 Average RF CV F1: {np.mean(results['rf_cv_scores']):.4f}")
    print(f"🎯 Best Submission: {results['best_submission']['filename']}")
    print(f"📊 Best Val F1: {results['best_submission']['val_f1']:.4f}")
    print(f"📊 Generated {len(results['all_submissions'])} submission files")
    print(f"{'='*80}")
    
    # 显示所有生成的文件
    print(f"\n🎉 SUBMISSIONS READY!")
    for i, sub in enumerate(results['all_submissions'], 1):
        print(f"📄 Rank {i}: {sub['filename']} (Val F1: {sub['val_f1']:.4f})")
    print(f"🌱 Seed: {seed_num}")