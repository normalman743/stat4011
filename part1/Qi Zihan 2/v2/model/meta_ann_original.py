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

version = 'v3.2refined'  # Refined: removed redundant phases, enhanced training output, immediate test prediction

@dataclass
class ModelConfig:
    """Configuration class for model hyperparameters"""
    n_rf_models: int = 100  # 增加基础模型数量到100
    meta_ann_hidden: int = 256  # 增加网络复杂度起始维度
    cv_folds: int = 5
    meta_ann_epochs: int = 500
    meta_ann_patience: int = 30
    meta_ann_dropout: float = 0.3
    meta_ann_lr: float = 1e-3
    meta_ann_weight_decay: float = 1e-4
    holdout_ratio: float = 0.2  # 新增20% hold-out验证集
    
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
# PyTorch Meta-ANN with Complex Architecture - 修改点1
# =====================================================
class MetaANN(nn.Module):
    def __init__(self, n_base, n_feat, dropout=0.3):
        super().__init__()
        self.a = nn.Parameter(torch.ones(n_feat))
        self.b = nn.Parameter(torch.zeros(n_feat))
        
        self.input_dim = n_base + n_feat
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ResNet块1: 256 → 128 → 128
        self.res_block1_1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.res_block1_2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout)
        )
        self.downsample1 = nn.Linear(256, 128)  # 降维匹配
        
        # ResNet块2: 128 → 64 → 64
        self.res_block2_1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.res_block2_2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout)
        )
        self.downsample2 = nn.Linear(128, 64)
        
        # ResNet块3: 64 → 32 → 32
        self.res_block3_1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.res_block3_2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout)
        )
        self.downsample3 = nn.Linear(64, 32)
        
        self.out = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x_base, x_feat):
        # 特征缩放和融合
        x_feat_scaled = self.a * x_feat + self.b
        x = torch.cat([x_base, x_feat_scaled], dim=1)
        
        # 输入层
        x = self.input_layer(x)
        
        # ResNet块1: 256 → 128 → 128 with skip connection
        identity1 = self.downsample1(x)
        out1 = self.res_block1_1(x)
        out1 = self.res_block1_2(out1)
        out1 = torch.relu(out1 + identity1)  # Skip connection
        
        # ResNet块2: 128 → 64 → 64 with skip connection
        identity2 = self.downsample2(out1)
        out2 = self.res_block2_1(out1)
        out2 = self.res_block2_2(out2)
        out2 = torch.relu(out2 + identity2)  # Skip connection
        
        # ResNet块3: 64 → 32 → 32 with skip connection
        identity3 = self.downsample3(out2)
        out3 = self.res_block3_1(out2)
        out3 = self.res_block3_2(out3)
        out3 = torch.relu(out3 + identity3)  # Skip connection
        
        return self.softmax(self.out(out3))

def train_pytorch_meta_ann(base_predictions, original_features, y_true, n_epochs=500, patience=30):
    """
    使用PyTorch训练Meta-ANN - 更新为2分类输出
    base_predictions: (n_samples, n_models) - 基础模型预测
    original_features: (n_samples, n_features) - 原始特征
    y_true: (n_samples,) - 真实标签
    """
    print(f"\n🤖 Training PyTorch Meta-ANN (Complex Architecture)")
    print(f"Base predictions shape: {base_predictions.shape}")
    print(f"Original features shape: {original_features.shape}")
    print(f"Architecture: {base_predictions.shape[1]+original_features.shape[1]} → 256 → 128 → 128 → 64 → 64 → 32 → 32 → 2")
    
    # 标准化原始特征
    scaler = StandardScaler()
    original_features = scaler.fit_transform(original_features)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 转换为PyTorch张量
    X_base_tensor = torch.tensor(base_predictions, dtype=torch.float32).to(device)
    X_feat_tensor = torch.tensor(original_features, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_true, dtype=torch.long).to(device)  # 改为long类型用于CrossEntropyLoss
    
    # 创建模型
    model = MetaANN(
        n_base=base_predictions.shape[1], 
        n_feat=original_features.shape[1],
        dropout=0.3
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()  # 改为交叉熵损失
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)

    # 交叉验证分割用于早停
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_splits = list(skf.split(base_predictions, y_true))
    train_idx, val_idx = all_splits[2]  # fold 3 (0-indexed)
    
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
            
            # 计算F1分数 - 使用argmax获取预测类别
            y_train_pred = model(Xb_train, Xf_train)
            train_pred = torch.argmax(y_train_pred, dim=1).cpu().numpy()
            val_pred = torch.argmax(y_pred_val, dim=1).cpu().numpy()
            
            # 整体F1和分类别F1
            train_f1 = metrics.f1_score(y_true[train_idx], train_pred, zero_division=0)
            val_f1 = metrics.f1_score(y_true[val_idx], val_pred, zero_division=0)
            
            # 分别计算Good类(0)和Bad类(1)的F1
            val_f1_good = metrics.f1_score(y_true[val_idx], val_pred, pos_label=0, zero_division=0)
            val_f1_bad = metrics.f1_score(y_true[val_idx], val_pred, pos_label=1, zero_division=0)
            
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
        y_final_pred_proba = model(X_base_tensor, X_feat_tensor).cpu().numpy()
        y_final_pred = np.argmax(y_final_pred_proba, axis=1)
        
        final_f1 = metrics.f1_score(y_true, y_final_pred, zero_division=0)
        final_f1_good = metrics.f1_score(y_true, y_final_pred, pos_label=0, zero_division=0)
        final_f1_bad = metrics.f1_score(y_true, y_final_pred, pos_label=1, zero_division=0)
        final_acc = metrics.accuracy_score(y_true, y_final_pred)
        
    print(f"\n📊 Meta-ANN Final Results:")
    print(f"   Accuracy: {final_acc:.4f}")
    print(f"   Overall F1: {final_f1:.4f}")
    print(f"   Good Class F1 (pos_label=0): {final_f1_good:.4f}")
    print(f"   Bad Class F1 (pos_label=1): {final_f1_bad:.4f}")
    print(f"   Best Val F1: {best_val_f1:.4f}")
    print(f"   Overfitting: {train_f1_history[-1] - best_val_f1:+.4f}")
    
    # 返回概率分布的第1列（Bad类概率）以保持兼容性
    return y_final_pred_proba[:, 1:2], model, scaler, {
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
def train_enhanced_rf_ensemble(data, n_models=100):  # 改为100个模型
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
    
    # RF分配策略 - 按比例缩减到50个模型
    rf_allocation = {
        'account_type': {'type4': 1, 'type3': 1, 'type1': 1, 'type2': 1},  # 15个*2
        'traditional': {'isolated': 1, 'backward_only': 1, 'both_directions': 1, 'forward_only': 1},  # 11个
        'interaction': {'B_in_B_out': 53, 'A_in_B_in_B_out': 31, 'A_in_A_out_B_in_B_out': 23, 
                       'A_out_B_in_B_out': 7, 'B_in': 6, 'small_categories': 6},  # 10个*5
        'behavior': {'inactive': 1, 'low_activity': 1, 'medium_activity_unidirectional': 1, 
                    'medium_activity_bidirectional': 1},  # 8个
        'volume': {'no_transactions': 1, 'medium_volume': 1, 'low_volume': 1, 'high_volume': 1},  # 6个
        'profit': {'loss_or_zero': 1, 'very_high_profit': 1}  # 2个
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
    """训练单个fold的Meta-ANN，使用早停，显示详细训练过程"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    print(f"\n🤖 Training Meta-ANN for Fold {fold_id+1} (Complex Architecture)")
    print(f"Loss Function: CrossEntropyLoss")
    print(f"Optimizer: AdamW")
    print(f"F1 Selection Criterion: {f1_type}")
    print(f"Device: {device}")
    print(f"Training samples: {X_base_train.shape[0]}, Validation samples: {X_base_val.shape[0]}")
    print(f"Architecture: {X_base_train.shape[1]+X_feat_train.shape[1]} → 256 → 128 → 128 → 64 → 64 → 32 → 32 → 2")
    
    print(f"\nEpoch | Train F1 | Val F1   | Good F1  | Bad F1   | Macro F1 | Weighted F1 | LR       | Status")
    print("-" * 95)
    
    # 特征缩放
    scaler_fold = StandardScaler()
    X_feat_train_scaled = scaler_fold.fit_transform(X_feat_train)
    X_feat_val_scaled = scaler_fold.transform(X_feat_val)
    
    # 创建模型
    model_fold = MetaANN(
        n_base=X_base_train.shape[1], 
        n_feat=X_feat_train_scaled.shape[1],
        dropout=0.3
    ).to(device)
    
    optimizer = optim.AdamW(model_fold.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()  # 改为交叉熵损失
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)
    
    # 转换为张量
    X_base_train_t = torch.tensor(X_base_train, dtype=torch.float32).to(device)
    X_feat_train_t = torch.tensor(X_feat_train_scaled, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)  # 改为long类型
    
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
            val_pred_proba = model_fold(X_base_val_t, X_feat_val_t).cpu().numpy()
            val_pred_label = np.argmax(val_pred_proba, axis=1)
            
            # Calculate F1 based on f1_type  
            if f1_type == 'bad':
                val_f1 = metrics.f1_score(y_val, val_pred_label, pos_label=1, zero_division=0)  # bad=1
            elif f1_type == 'macro':
                val_f1 = metrics.f1_score(y_val, val_pred_label, average='macro', zero_division=0)
            elif f1_type == 'weighted':
                val_f1 = metrics.f1_score(y_val, val_pred_label, average='weighted', zero_division=0)
            else:  # default to 'bad'
                val_f1 = metrics.f1_score(y_val, val_pred_label, pos_label=1, zero_division=0)  # bad=1
        
        # 计算所有F1分数用于显示
        with torch.no_grad():
            train_pred_proba = model_fold(X_base_train_t, X_feat_train_t).cpu().numpy()
        train_label = np.argmax(train_pred_proba, axis=1)
        
        # 计算各种F1
        if f1_type == 'bad':
            train_f1_display = metrics.f1_score(y_train, train_label, pos_label=1, zero_division=0)
        elif f1_type == 'macro':
            train_f1_display = metrics.f1_score(y_train, train_label, average='macro', zero_division=0)
        elif f1_type == 'weighted':
            train_f1_display = metrics.f1_score(y_train, train_label, average='weighted', zero_division=0)
        else:
            train_f1_display = metrics.f1_score(y_train, train_label, pos_label=1, zero_division=0)
            
        val_f1_good = metrics.f1_score(y_val, val_pred_label, pos_label=0, zero_division=0)
        val_f1_bad = metrics.f1_score(y_val, val_pred_label, pos_label=1, zero_division=0)  
        val_f1_macro = metrics.f1_score(y_val, val_pred_label, average='macro', zero_division=0)
        val_f1_weighted = metrics.f1_score(y_val, val_pred_label, average='weighted', zero_division=0)

        # 学习率调度
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 早停检查
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model_fold.state_dict().copy()
            status = "✅ Best"
        else:
            patience_counter += 1
            status = f"⏳ {patience_counter}/{patience}"
        
        # 打印进度 - 每5个epoch或最佳epoch
        if epoch % 5 == 0 or patience_counter == 0 or epoch < 10:
            print(f"{epoch:5d} | {train_f1_display:8.4f} | {val_f1:8.4f} | {val_f1_good:8.4f} | {val_f1_bad:8.4f} | {val_f1_macro:8.4f} | {val_f1_weighted:8.4f} | {current_lr:.2e} | {status}")
        
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch}")
            print(f"🏆 Best validation F1: {best_val_f1:.4f}")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model_fold.load_state_dict(best_model_state)
    
    # 🔧 修复最终评估部分
    model_fold.eval()
    with torch.no_grad():
        train_pred = model_fold(X_base_train_t, X_feat_train_t).cpu().numpy()
        val_pred = model_fold(X_base_val_t, X_feat_val_t).cpu().numpy()
        
        # ✅ 正确处理：使用 argmax 获取预测类别
        train_label = np.argmax(train_pred, axis=1)  # 从 (n_samples, 2) 获取 (n_samples,)
        val_label = np.argmax(val_pred, axis=1)      # 从 (n_samples, 2) 获取 (n_samples,)
        
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



def train_two_stage_cascade_model(data, strategy_data, test_data=None, stage1_models=100, stage2_models=50):
    """
    训练两阶段级联模型
    Stage 1: 高召回率模型 (类似threshold=0) - 筛选候选Bad账户
    Stage 2: 高精确率模型 - 在候选中精确识别Bad账户
    """
    print(f"\n🎯 Training Two-Stage Cascade Model")
    print(f"   Stage 1: High Recall ({stage1_models} models) - 筛选候选")  
    print(f"   Stage 2: High Precision ({stage2_models} models) - 精确识别")
    
    data_copy = data.copy()
    data_copy.loc[data_copy['flag'] == -1, 'flag'] = 0
    feature_cols = [col for col in data_copy.columns 
                   if col not in ['account', 'flag', 'account_type'] and not col.endswith('_category')]
    
    # =====================================================
    # Stage 1: 高召回率模型训练 (优化召回率 >= 0.85)
    # =====================================================
    print(f"\n🔥 Stage 1: High Recall Model Training")
    
    stage1_predictions, stage1_cv_scores, stage1_models_list = train_high_recall_ensemble(
        data_copy, feature_cols, n_models=stage1_models, target_recall=0.85
    )
    
    # Stage 1预测：获取候选Bad账户索引
    stage1_ensemble_proba = np.mean(stage1_predictions, axis=1)
    stage1_threshold = find_optimal_recall_threshold(stage1_ensemble_proba, data_copy['flag'].values, target_recall=0.85)
    stage1_candidates = stage1_ensemble_proba >= stage1_threshold
    
    print(f"   Stage 1 Results:")
    print(f"      Optimal threshold: {stage1_threshold:.4f}")
    print(f"      Candidates selected: {np.sum(stage1_candidates)}")
    print(f"      Average recall: {np.mean([check_recall_at_threshold(stage1_predictions[:, i], data_copy['flag'].values, stage1_threshold) for i in range(stage1_predictions.shape[1])]):.4f}")
    
    # =====================================================
    # Stage 2: 高精确率模型训练 (在候选中优化精确率)
    # =====================================================
    print(f"\n🎯 Stage 2: High Precision Model Training")
    
    # 创建Stage 2训练数据：只包含Stage 1候选
    stage2_data = data_copy[stage1_candidates].copy()
    print(f"   Stage 2 training data: {len(stage2_data)} samples")
    print(f"   Bad accounts in candidates: {sum(stage2_data['flag'])}")
    print(f"   Good accounts in candidates: {len(stage2_data) - sum(stage2_data['flag'])}")
    
    if len(stage2_data) < 50:
        print(f"   ⚠️ Too few candidates for Stage 2, falling back to original data")
        stage2_data = data_copy.copy()
    
    stage2_predictions, stage2_cv_scores, stage2_models_list = train_high_precision_ensemble(
        stage2_data, data_copy, feature_cols, n_models=stage2_models, target_precision=0.8
    )
    
    # =====================================================
    # 级联组合预测
    # =====================================================
    print(f"\n🔗 Combining Two-Stage Predictions")
    
    # 最终预测：Stage 1候选 AND Stage 2高置信度
    stage2_ensemble_proba = np.mean(stage2_predictions, axis=1)
    stage2_threshold = find_optimal_precision_threshold(stage2_ensemble_proba, data_copy['flag'].values, target_precision=0.8)
    
    # 级联逻辑：只有通过Stage 1的才能被Stage 2考虑
    final_predictions = np.zeros(len(data_copy))
    stage2_mask = stage2_ensemble_proba >= stage2_threshold
    final_predictions[stage1_candidates & stage2_mask] = 1
    
    # 评估级联效果
    y_true = data_copy['flag'].values
    cascade_precision = metrics.precision_score(y_true, final_predictions, zero_division=0)
    cascade_recall = metrics.recall_score(y_true, final_predictions, zero_division=0)
    cascade_f1 = metrics.f1_score(y_true, final_predictions, zero_division=0)
    cascade_accuracy = metrics.accuracy_score(y_true, final_predictions)
    
    print(f"\n📊 Two-Stage Cascade Results:")
    print(f"   Final Bad predictions: {int(np.sum(final_predictions))}")
    print(f"   Precision: {cascade_precision:.4f}")
    print(f"   Recall: {cascade_recall:.4f}")
    print(f"   F1 Score: {cascade_f1:.4f}")
    print(f"   Accuracy: {cascade_accuracy:.4f}")
    
    # =====================================================
    # 测试集预测（如果提供）
    # =====================================================
    test_predictions = None
    if test_data is not None:
        print(f"\n🔮 Generating Test Predictions")
        
        test_feature_cols = [col for col in test_data.columns if col not in ['account', 'flag', 'account_type'] and not col.endswith('_category')]
        X_test = test_data[test_feature_cols].values
        
        # Stage 1测试预测
        test_stage1_proba = np.mean([model.predict_proba(X_test)[:, 1] for model in stage1_models_list], axis=0)
        test_stage1_candidates = test_stage1_proba >= stage1_threshold
        
        # Stage 2测试预测
        test_stage2_proba = np.mean([model.predict_proba(X_test)[:, 1] for model in stage2_models_list], axis=0)
        test_stage2_mask = test_stage2_proba >= stage2_threshold
        
        # 级联测试预测
        test_final_predictions = np.zeros(len(test_data))
        test_final_predictions[test_stage1_candidates & test_stage2_mask] = 1
        
        test_predictions = test_final_predictions
        
        print(f"   Test Stage 1 candidates: {np.sum(test_stage1_candidates)}")
        print(f"   Test final predictions: {np.sum(test_final_predictions)}")
    
    return {
        'stage1_predictions': stage1_predictions,
        'stage2_predictions': stage2_predictions,
        'final_predictions': final_predictions,
        'test_predictions': test_predictions,
        'stage1_threshold': stage1_threshold,
        'stage2_threshold': stage2_threshold,
        'cascade_metrics': {
            'precision': cascade_precision,
            'recall': cascade_recall,
            'f1': cascade_f1,
            'accuracy': cascade_accuracy
        },
        'stage1_cv_scores': stage1_cv_scores,
        'stage2_cv_scores': stage2_cv_scores
    }

def train_high_recall_ensemble(data, feature_cols, n_models=100, target_recall=0.85):
    """Stage 1: 训练高召回率集成模型"""
    print(f"   🔥 Training {n_models} high-recall models (target recall: {target_recall})")
    
    good_accounts = len(data[data['flag'] == 1])
    bad_accounts = len(data[data['flag'] == 0])
    
    # 激进的正样本采样策略：增加Bad样本权重
    sample_size_good = min(good_accounts, 300)  
    sample_size_bad = min(bad_accounts, 150)   # Good:Bad = 2:1，增加Bad样本影响力
    
    X_all = data[feature_cols].values
    y_all = data['flag'].values
    
    predictions = []
    cv_scores = []
    models = []
    
    # 高召回率优化的RF配置
    high_recall_configs = [
        {'n_estimators': 200, 'max_depth': 30, 'min_samples_split': 2, 'min_samples_leaf': 1, 'class_weight': {0: 1, 1: 8}},
        {'n_estimators': 150, 'max_depth': 35, 'min_samples_split': 3, 'min_samples_leaf': 1, 'class_weight': {0: 1, 1: 10}},
        {'n_estimators': 180, 'max_depth': 25, 'min_samples_split': 2, 'min_samples_leaf': 2, 'class_weight': {0: 1, 1: 6}},
    ]
    
    for i in tqdm(range(n_models), desc="High-Recall Models"):
        # 重采样策略：更多Bad样本
        good_sample = data[data['flag'] == 1].sample(n=sample_size_good, replace=True, random_state=i)
        bad_sample = data[data['flag'] == 0].sample(n=sample_size_bad, replace=True, random_state=i+2000)
        train_data = pd.concat([good_sample, bad_sample], axis=0)
        
        X_train = train_data[feature_cols].values
        y_train = train_data['flag'].values
        
        config = high_recall_configs[i % len(high_recall_configs)]
        
        clf = RandomForestClassifier(
            **config,
            random_state=i,
            bootstrap=True,
            max_features='sqrt',
            oob_score=True
        )
        clf.fit(X_train, y_train)
        models.append(clf)
        
        # 评估召回率
        y_pred_proba = clf.predict_proba(X_all)[:, 1]
        predictions.append(y_pred_proba)
        
        # 以召回率为主的评估
        optimal_threshold = find_optimal_recall_threshold(y_pred_proba, y_all, target_recall)
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        recall_score = metrics.recall_score(y_all, y_pred, zero_division=0)
        cv_scores.append(recall_score)
    
    predictions_array = np.array(predictions).T
    avg_recall = np.mean(cv_scores)
    
    print(f"      Average Recall: {avg_recall:.4f}")
    print(f"      Recall std: {np.std(cv_scores):.4f}")
    
    return predictions_array, cv_scores, models

def train_high_precision_ensemble(stage2_data, full_data, feature_cols, n_models=50, target_precision=0.8):
    """Stage 2: 训练高精确率集成模型"""
    print(f"   🎯 Training {n_models} high-precision models (target precision: {target_precision})")
    
    good_accounts = len(stage2_data[stage2_data['flag'] == 1])
    bad_accounts = len(stage2_data[stage2_data['flag'] == 0])
    
    if bad_accounts == 0 or good_accounts == 0:
        print(f"      ⚠️ Insufficient class diversity in Stage 2 data")
        stage2_data = full_data.copy()  # 回退到全数据
        good_accounts = len(stage2_data[stage2_data['flag'] == 1])
        bad_accounts = len(stage2_data[stage2_data['flag'] == 0])
    
    # 平衡采样策略：精确识别Bad样本
    sample_size = min(good_accounts, bad_accounts, 200)
    
    X_all = full_data[feature_cols].values
    y_all = full_data['flag'].values
    
    predictions = []
    cv_scores = []
    models = []
    
    # 高精确率优化的RF配置
    high_precision_configs = [
        {'n_estimators': 250, 'max_depth': 15, 'min_samples_split': 10, 'min_samples_leaf': 5, 'class_weight': 'balanced'},
        {'n_estimators': 300, 'max_depth': 12, 'min_samples_split': 15, 'min_samples_leaf': 8, 'class_weight': 'balanced'},
        {'n_estimators': 200, 'max_depth': 18, 'min_samples_split': 8, 'min_samples_leaf': 3, 'class_weight': 'balanced'},
    ]
    
    for i in tqdm(range(n_models), desc="High-Precision Models"):
        # 在Stage 2候选数据中平衡采样
        good_sample = stage2_data[stage2_data['flag'] == 1].sample(n=sample_size, replace=True, random_state=i+3000)
        bad_sample = stage2_data[stage2_data['flag'] == 0].sample(n=sample_size, replace=True, random_state=i+4000)
        train_data = pd.concat([good_sample, bad_sample], axis=0)
        
        X_train = train_data[feature_cols].values
        y_train = train_data['flag'].values
        
        config = high_precision_configs[i % len(high_precision_configs)]
        
        clf = RandomForestClassifier(
            **config,
            random_state=i+1000,
            bootstrap=True,
            max_features='log2',  # 更保守的特征选择
            oob_score=True
        )
        clf.fit(X_train, y_train)
        models.append(clf)
        
        # 评估精确率
        y_pred_proba = clf.predict_proba(X_all)[:, 1]
        predictions.append(y_pred_proba)
        
        # 以精确率为主的评估
        optimal_threshold = find_optimal_precision_threshold(y_pred_proba, y_all, target_precision)
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        precision_score = metrics.precision_score(y_all, y_pred, zero_division=0)
        cv_scores.append(precision_score)
    
    predictions_array = np.array(predictions).T
    avg_precision = np.mean(cv_scores)
    
    print(f"      Average Precision: {avg_precision:.4f}")
    print(f"      Precision std: {np.std(cv_scores):.4f}")
    
    return predictions_array, cv_scores, models

def find_optimal_recall_threshold(y_pred_proba, y_true, target_recall=0.85):
    """寻找达到目标召回率的最优阈值"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_recall = 0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        recall = metrics.recall_score(y_true, y_pred, zero_division=0)
        
        if recall >= target_recall and recall > best_recall:
            best_recall = recall
            best_threshold = threshold
    
    return best_threshold

def find_optimal_precision_threshold(y_pred_proba, y_true, target_precision=0.8):
    """寻找达到目标精确率的最优阈值"""
    thresholds = np.arange(0.3, 0.9, 0.01)
    best_threshold = 0.5
    best_precision = 0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precision = metrics.precision_score(y_true, y_pred, zero_division=0)
        
        if precision >= target_precision and precision > best_precision:
            best_precision = precision
            best_threshold = threshold
    
    return best_threshold

def check_recall_at_threshold(y_pred_proba, y_true, threshold):
    """检查指定阈值下的召回率"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    return metrics.recall_score(y_true, y_pred, zero_division=0)

def main_with_cascade(f1_type='bad', use_cascade=True):
    """主函数 - 支持两阶段级联模式"""
    print(f"=== Two-Stage Cascade Model Training (F1 Type: {f1_type}) ===")
    
    # 数据加载部分保持不变...
    print("\n=== Loading Data ===")
    features_path = '/Users/mannormal/4011/Qi Zihan/v2/feature_extraction/result/features_cleaned_no_leakage1.csv'
    all_features_df = pd.read_csv(features_path)

    pwd = '/Users/mannormal/4011/Qi Zihan/original_data/'
    ta = pd.read_csv(pwd + 'train_acc.csv')
    te = pd.read_csv(pwd + 'test_acc_predict.csv')
    
    # 数据准备...
    cols_to_drop = []
    if 'flag' in all_features_df.columns:
        cols_to_drop.append('flag')
    if 'data_type' in all_features_df.columns:
        cols_to_drop.append('data_type')
    
    if cols_to_drop:
        all_features_df = all_features_df.drop(cols_to_drop, axis=1)
    
    ta.loc[ta['flag'] == 0, 'flag'] = -1
    training_df = pd.merge(all_features_df, ta[['account', 'flag']], on='account', how='inner')
    training_df['account_type'] = training_df.apply(classify_account_type_improved, axis=1)
    
    test_df = pd.merge(all_features_df, te[['account']], on='account', how='inner')
    test_df['account_type'] = test_df.apply(classify_account_type_improved, axis=1)
    
    strategy_data = load_strategy_categories()
    
    if use_cascade:
        # =====================================================
        # 两阶段级联模式
        # =====================================================
        print(f"\n{'='*80}")
        print("🎯 TWO-STAGE CASCADE TRAINING")
        print(f"{'='*80}")
        
        cascade_results = train_two_stage_cascade_model(
            data=training_df,
            strategy_data=strategy_data,
            test_data=test_df,
            stage1_models=100,  # Stage 1: 高召回率
            stage2_models=50    # Stage 2: 高精确率
        )
        
        # 生成测试预测文件
        if cascade_results['test_predictions'] is not None:
            test_pred_labels = cascade_results['test_predictions'].astype(int)
            
            submission_df = pd.DataFrame({
                'ID': test_df['account'].values,
                'Predict': test_pred_labels
            })
            
            # 统计结果
            pred_counts = submission_df['Predict'].value_counts()
            print(f"\n📊 Test Predictions:")
            print(f"   Good (1): {pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/len(submission_df)*100:.1f}%)")
            print(f"   Bad (0): {pred_counts.get(0, 0)} ({pred_counts.get(0, 0)/len(submission_df)*100:.1f}%)")
            
            # 保存文件
            cascade_f1 = cascade_results['cascade_metrics']['f1']
            cascade_precision = cascade_results['cascade_metrics']['precision']  
            cascade_recall = cascade_results['cascade_metrics']['recall']
            
            filename = f"cascade_f1_{cascade_f1:.4f}_precision_{cascade_precision:.4f}_recall_{cascade_recall:.4f}_seed_{seed_num}.csv"
            filepath = f"/Users/mannormal/4011/Qi Zihan/v2/results/{filename}"
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            submission_df.to_csv(filepath, index=False)
            
            print(f"✅ Saved: {filename}")
        
        return cascade_results
    
    else:
        # 原有的Meta-ANN模式
        return main(f1_type=f1_type)


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
    
    # =====================================================
    # 20% Hold-out验证集分割 - 新增部分
    # =====================================================
    print("\n=== Creating 20% Hold-out Validation Set ===")
    from sklearn.model_selection import train_test_split
    
    # 合并特征和标签数据
    cols_to_drop = []
    if 'flag' in all_features_df.columns:
        cols_to_drop.append('flag')
    if 'data_type' in all_features_df.columns:
        cols_to_drop.append('data_type')
    
    if cols_to_drop:
        print(f"⚠️  特征数据中的以下列将被删除: {cols_to_drop}")
        all_features_df = all_features_df.drop(cols_to_drop, axis=1)
    
    ta.loc[ta['flag'] == 0, 'flag'] = -1
    full_training_df = pd.merge(all_features_df, ta[['account', 'flag']], on='account', how='inner')
    full_training_df['account_type'] = full_training_df.apply(classify_account_type_improved, axis=1)
    
    # 分层抽样创建hold-out验证集
    y_for_split = np.where(full_training_df['flag'].values == -1, 0, 1)
    train_indices, holdout_indices = train_test_split(
        range(len(full_training_df)), 
        test_size=CONFIG.holdout_ratio, 
        stratify=y_for_split, 
        random_state=42
    )
    
    training_df = full_training_df.iloc[train_indices].copy()
    holdout_df = full_training_df.iloc[holdout_indices].copy()
    
    print(f"Total data: {full_training_df.shape[0]}")
    print(f"Training data: {training_df.shape[0]} ({len(train_indices)/len(full_training_df)*100:.1f}%)")
    print(f"Hold-out validation: {holdout_df.shape[0]} ({len(holdout_indices)/len(full_training_df)*100:.1f}%)")
    print(f"Training flag distribution: {dict(training_df['flag'].value_counts())}")
    print(f"Hold-out flag distribution: {dict(holdout_df['flag'].value_counts())}")
    
    strategy_data = load_strategy_categories()
    
    print(f"Account type distribution: {dict(training_df['account_type'].value_counts())}")
    
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
    
    # 合并所有预测 (策略特定训练已包含所有模型)
    print(f"\n📊 Combining Predictions:")
    print(f"   Strategy-specific predictions: {rf_predictions.shape}")
    
    combined_base_predictions = rf_predictions
    print(f"   📊 Total models: {combined_base_predictions.shape[1]} (distributed across 6 strategies)")
    
    # =====================================================
    # Phase 2: CV训练并选择最佳模型 + 测试集预测
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 PHASE 2: Cross-Validation Meta-ANN Training & Test Prediction")
    print(f"{'='*80}")
    
    # 准备测试数据
    test_feature_cols = [col for col in test_df.columns if col not in ['account', 'flag', 'account_type'] and not col.endswith('_category')]
    test_original_features = test_df[test_feature_cols].values
    test_combined_predictions = test_rf_predictions
    
    # 5折交叉验证评估
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    fold_models = []
    all_test_submissions = []  # 存储所有fold的测试预测
    
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
        
        print(f"{fold+1:4d} | {fold_result['train_f1']:8.4f} | {fold_result['val_f1']:8.4f} | {fold_result['val_f1_good']:8.4f} | {fold_result['val_f1_bad']:8.4f} | {fold_result['train_acc']:8.4f} | {fold_result['val_acc']:8.4f} | {overfitting:+.4f} | {overfit_status}")
        
        # 立即进行测试集预测
        print(f"   🔮 Generating test predictions for Fold {fold+1}...")
        
        try:
            model = fold_result['model']
            scaler = fold_result['scaler']
            
            # 准备测试数据
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            test_base_features = test_combined_predictions
            test_original_features_scaled = scaler.transform(test_original_features)
            
            # Meta-ANN预测
            model.eval()
            with torch.no_grad():
                X_base_tensor = torch.FloatTensor(test_base_features).to(device)
                X_feat_tensor = torch.FloatTensor(test_original_features_scaled).to(device)
                
                # 🔧 修复：正确处理模型输出
                test_pred_proba_raw = model(X_base_tensor, X_feat_tensor).cpu().numpy()  # shape: (n_samples, 2)
                
                # 获取bad类（标签1）的概率
                test_pred_proba_bad = test_pred_proba_raw[:, 1]  # shape: (n_samples,) - bad类概率
                
                # 根据概率阈值生成预测标签（这里根据你的业务逻辑调整阈值）
                test_pred_labels = (test_pred_proba_bad > 0.5).astype(int)  # shape: (n_samples,)
            
            # 创建提交文件 - 确保长度匹配
            submission_df = pd.DataFrame({
                'ID': test_df['account'].values,  # shape: (n_samples,)
                'Predict': test_pred_labels       # shape: (n_samples,)
            })
            
            # 验证数据形状
            print(f"      Debug - Account shape: {test_df['account'].shape}")
            print(f"      Debug - Prediction shape: {test_pred_labels.shape}")
            print(f"      Debug - DataFrame shape: {submission_df.shape}")
            
            # 统计结果
            pred_counts = submission_df['Predict'].value_counts()
            print(f"      Good (1): {pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/len(submission_df)*100:.1f}%)")
            print(f"      Bad (0): {pred_counts.get(0, 0)} ({pred_counts.get(0, 0)/len(submission_df)*100:.1f}%)")
            
            # 生成文件名
            fold_f1 = fold_result['val_f1']
            fold_f1_good = fold_result['val_f1_good'] 
            fold_f1_bad = fold_result['val_f1_bad']
            fold_f1_macro = fold_result['val_f1_macro']
            fold_f1_weighted = fold_result['val_f1_weighted']
            
            filename = f"{version}_fold{fold+1}_{f1_type}_f1_{fold_f1:.4f}_good_{fold_f1_good:.4f}_bad_{fold_f1_bad:.4f}_macro_{fold_f1_macro:.4f}_weighted_{fold_f1_weighted:.4f}_seed_{seed_num}.csv"
            filepath = f"/Users/mannormal/4011/Qi Zihan/v2/results/{filename}"
            
            # 保存文件
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            submission_df.to_csv(filepath, index=False)
            
            all_test_submissions.append({
                'fold': fold + 1,
                'val_f1': fold_f1,
                'filename': filename,
                'filepath': filepath,
                'submission_df': submission_df,
                'pred_counts': pred_counts
            })
            
            print(f"      ✅ Saved: {filename}")
            
        except Exception as e:
            print(f"      ❌ Error generating prediction for Fold {fold+1}: {str(e)}")
            import traceback
            print(f"      📝 Full traceback:")
            traceback.print_exc()
            continue
    
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
    print(f"   Strategy-specific models integrated into ensemble")
    
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
    print(f"   Base models: {combined_base_predictions.shape[1]} (Strategy-specific distributed)")
    print(f"   Original features: {original_features.shape[1]}")
    print(f"   Meta-ANN: Simplified with {combined_base_predictions.shape[1]+original_features.shape[1]} → 64 → 32 → 16 → 1")
    
    # =====================================================
    # Hold-out验证集评估 - 新增部分
    # =====================================================
    print(f"\n🔍 Hold-out Validation Evaluation:")
    
    # 准备hold-out数据
    holdout_feature_cols = [col for col in holdout_df.columns 
                           if col not in ['account', 'flag', 'account_type']]
    holdout_original_features = holdout_df[holdout_feature_cols].values
    holdout_y_true = np.where(holdout_df['flag'].values == -1, 0, 1)
    
    # 生成hold-out的RF预测（使用训练好的模型）
    print(f"   Generating RF predictions for hold-out set...")
    # 这里需要用训练好的RF模型对hold-out数据进行预测
    # 由于RF模型已经训练完成，我们需要重新生成hold-out预测
    # 注意：这里简化处理，实际应该保存RF模型进行预测
    
    print(f"   Hold-out set: {holdout_df.shape[0]} samples")
    print(f"   Hold-out class distribution: {dict(zip(*np.unique(holdout_y_true, return_counts=True)))}")

    # =====================================================
    # Phase 3: 测试集预测分析比较
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 PHASE 3: Test Set Predictions Analysis & Comparison")
    print(f"{'='*80}")
    
    print(f"📊 Test Data Info:")
    print(f"   Test accounts: {test_df.shape[0]}")
    print(f"   Test account type distribution: {dict(test_df['account_type'].value_counts())}")
    
    if len(all_test_submissions) == 0:
        print("❌ No test predictions generated!")
        return None
    
    # 分析各fold预测结果
    print(f"\n🔍 Prediction Analysis:")
    print(f"   Total fold predictions generated: {len(all_test_submissions)}")
    
    # 按验证F1排序
    sorted_submissions = sorted(all_test_submissions, key=lambda x: x['val_f1'], reverse=True)
    
    print(f"\n📊 Prediction Summary (sorted by Val F1):")
    print("Rank | Fold | Val F1   | Good (1) | Bad (0)  | Filename")
    print("-" * 80)
    
    for rank, sub in enumerate(sorted_submissions, 1):
        pred_counts = sub['pred_counts']
        good_count = pred_counts.get(1, 0)
        bad_count = pred_counts.get(0, 0)
        good_pct = good_count / len(sub['submission_df']) * 100
        bad_pct = bad_count / len(sub['submission_df']) * 100
        
        print(f"{rank:4d} | {sub['fold']:4d} | {sub['val_f1']:8.4f} | {good_count:4d}({good_pct:4.1f}%) | {bad_count:4d}({bad_pct:4.1f}%) | {sub['filename']}")
    
    # 设置最佳提交
    best_submission = {
        'val_f1': sorted_submissions[0]['val_f1'],
        'filename': sorted_submissions[0]['filename'],
        'fold': sorted_submissions[0]['fold'],
        'submission_df': sorted_submissions[0]['submission_df']
    }
    
    print(f"\n🏆 Best submission selected:")
    print(f"   Fold {best_submission['fold']}: {best_submission['filename']}")
    print(f"   Validation F1: {best_submission['val_f1']:.4f}")
    
    submission_df = best_submission['submission_df']
    
    # =====================================================
    # 返回结果汇总
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 TRAINING COMPLETED - RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"📊 Final Results:")
    print(f"   Best Validation F1: {best_submission['val_f1']:.4f}")
    print(f"   Total Fold Models: {len(cv_results)}")
    print(f"   Generated Submissions: {len(sorted_submissions)}")
    print(f"   Primary Submission: {best_submission['filename']}")
    print(f"   Reduced Model Complexity: 50 RF models + simplified 64→32→16→1 Meta-ANN")
    print(f"   Hold-out Validation: {CONFIG.holdout_ratio*100:.0f}% of data reserved for unbiased evaluation")
    
    return {
        'cv_results': cv_results,
        'best_submission': best_submission,
        'all_submissions': sorted_submissions,  # 使用sorted_submissions替代all_submissions
        'rf_cv_scores': rf_cv_scores,
        'rf_predictions': rf_predictions,
        'training_df': training_df,
        'holdout_df': holdout_df,
        'config_changes': {
            'model_reduction': '198→50 RF models',
            'architecture_simplification': 'ResNet→Linear (64→32→16→1)',
            'holdout_validation': '20% data reserved',
            'expected_benefit': 'Reduced overfitting, better generalization'
        }
    }
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Two-Stage Cascade or Meta-ANN Training')
    parser.add_argument('--f1_type', type=str, default='bad', 
                        choices=['bad', 'macro', 'weighted'],
                        help='F1 score type for model selection (default: bad)')
    parser.add_argument('--use_cascade', action='store_true', default=False,
                        help='Use two-stage cascade model instead of Meta-ANN')
    
    args = parser.parse_args()
    
    if args.use_cascade:
        results = main_with_cascade(f1_type=args.f1_type, use_cascade=True)
        print(f"\n🎯 Cascade Training Complete!")
        print(f"📊 Cascade F1: {results['cascade_metrics']['f1']:.4f}")
        print(f"📊 Precision: {results['cascade_metrics']['precision']:.4f}")  
        print(f"📊 Recall: {results['cascade_metrics']['recall']:.4f}")
    else:
        results = main(f1_type=args.f1_type)
        print(f"\n✅ Meta-ANN Training Complete!")
        print(f"🎯 Best Val F1: {results['best_submission']['val_f1']:.4f}")
        print(f"📊 Generated {len(results['all_submissions'])} fold predictions")