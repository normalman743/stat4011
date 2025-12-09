import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

print("=== ULTRA Multi-Strategy Ensemble System with Optimized Meta-ANN ===")

# =====================================================
# 优化的Meta-ANN类定义
# =====================================================
class RefinedMetaANN(nn.Module):
    """优化的Meta-ANN，融合base predictions和original features"""
    
    def __init__(self, n_base_models, n_original_features, dropout_rates=[0.3, 0.2, 0.2, 0.1, 0.1]):
        super(RefinedMetaANN, self).__init__()
        
        # 输入维度 = base model predictions + original features
        input_dim = n_base_models + n_original_features
        
        # 可训练的特征缩放参数
        self.feature_scaler = nn.Parameter(torch.ones(n_original_features))
        self.feature_bias = nn.Parameter(torch.zeros(n_original_features))
        
        # 更深的网络架构
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rates[2]),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rates[3]),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rates[4]),
            
            nn.Linear(16, 2)  # 输出2个类别的logits
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x_base, x_feat):
        # 特征缩放
        x_feat_scaled = self.feature_scaler * x_feat + self.feature_bias
        
        # 融合base predictions和scaled features
        x = torch.cat([x_base, x_feat_scaled], dim=1)
        
        # 通过网络
        return self.network(x)

class LightLabelSmoothingCE(nn.Module):
    """轻量级Label Smoothing交叉熵损失"""
    def __init__(self, smoothing=0.03, class_weights=None):
        super(LightLabelSmoothingCE, self).__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights
        
    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=1)
        
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (pred.size(1) - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -torch.sum(smooth_target * log_prob, dim=1)
        
        if self.class_weights is not None:
            weights = self.class_weights[target]
            loss = loss * weights
        
        return loss.mean()

class SimpleEarlyStopping:
    """简单的早停机制，专注于Bad客户F1"""
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}

def train_refined_meta_ann(base_predictions, original_features, y_true, n_epochs=500, patience=20):
    """
    训练优化的Meta-ANN，重点关注Bad客户F1分数，包含阈值优化
    
    Args:
        base_predictions: (n_samples, n_models) - 基础模型预测概率
        original_features: (n_samples, n_features) - 原始特征
        y_true: (n_samples,) - 真实标签 (0: Good, 1: Bad)
    
    Returns:
        训练好的模型、预测结果和详细指标
    """
    print(f"\n🤖 训练优化Meta-ANN (重点: Bad客户F1)")
    print(f"Base predictions shape: {base_predictions.shape}")
    print(f"Original features shape: {original_features.shape}")
    print(f"Label distribution: {dict(zip(*np.unique(y_true, return_counts=True)))}")
    
    # 特征标准化
    scaler = StandardScaler()
    original_features_scaled = scaler.fit_transform(original_features)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 转换为PyTorch张量
    X_base_tensor = torch.tensor(base_predictions, dtype=torch.float32).to(device)
    X_feat_tensor = torch.tensor(original_features_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_true, dtype=torch.long).to(device)  # 注意这里用LongTensor
    
    # 计算类别权重
    class_counts = np.bincount(y_true)
    class_weights = torch.FloatTensor(len(y_true) / (len(class_counts) * class_counts)).to(device)
    print(f"Class weights: {class_weights}")
    
    # 创建优化的模型
    model = RefinedMetaANN(
        n_base_models=base_predictions.shape[1], 
        n_original_features=original_features_scaled.shape[1],
        dropout_rates=[0.3, 0.2, 0.2, 0.1, 0.1]
    ).to(device)
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = LightLabelSmoothingCE(smoothing=0.03, class_weights=class_weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=8, factor=0.5, min_lr=1e-6)
    early_stopping = SimpleEarlyStopping(patience=patience, min_delta=0.001)
    
    # 分割训练和验证集
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(base_predictions, y_true))[0]
    
    Xb_train, Xb_val = X_base_tensor[train_idx], X_base_tensor[val_idx]
    Xf_train, Xf_val = X_feat_tensor[train_idx], X_feat_tensor[val_idx]
    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
    
    # 创建数据加载器
    train_dataset = TensorDataset(Xb_train, Xf_train, y_train)
    val_dataset = TensorDataset(Xb_val, Xf_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 训练历史记录
    train_history = []
    best_bad_f1 = 0
    
    print("\nEpoch | Train Loss | Bad F1 | Macro F1 | Accuracy | Good F1 | LR       | Status")
    print("-" * 80)
    
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_xb, batch_xf, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_xb, batch_xf)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_xb, batch_xf, batch_y in val_loader:
                outputs = model(batch_xb, batch_xf)
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        # 计算详细指标
        val_f1_macro = f1_score(val_targets, val_predictions, average='macro', zero_division=0)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        
        # 计算各类别F1
        val_f1_per_class = f1_score(val_targets, val_predictions, average=None, zero_division=0)
        good_f1 = val_f1_per_class[0]
        bad_f1 = val_f1_per_class[1] if len(val_f1_per_class) > 1 else 0  # 主要指标
        
        # 学习率调度
        scheduler.step(bad_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'bad_f1': bad_f1,
            'macro_f1': val_f1_macro,
            'accuracy': val_accuracy,
            'good_f1': good_f1,
            'lr': current_lr
        }
        train_history.append(epoch_data)
        
        # 早停检查
        if bad_f1 > best_bad_f1:
            best_bad_f1 = bad_f1
            status = "✅ Best"
        else:
            status = f"⏳ {early_stopping.counter+1}/{early_stopping.patience}"
        
        # 打印进度
        if epoch % 20 == 0 or epoch >= n_epochs - 5:
            print(f"{epoch+1:5d} | {avg_train_loss:10.4f} | {bad_f1:6.4f} | {val_f1_macro:8.4f} | "
                  f"{val_accuracy:8.4f} | {good_f1:7.4f} | {current_lr:.2e} | {status}")
        
        if early_stopping(bad_f1, model):
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            print(f"🏆 Best Bad F1: {best_bad_f1:.4f}")
            break
    
    # =====================================================
    # 最终评估 - 添加阈值优化
    # =====================================================
    print(f"\n{'='*60}")
    print("🎯 阈值优化与最终评估")
    print(f"{'='*60}")
    
    model.eval()
    with torch.no_grad():
        # 获取概率预测
        final_outputs = model(X_base_tensor, X_feat_tensor)
        final_probs = F.softmax(final_outputs, dim=1)
        final_probs_np = final_probs.cpu().numpy()
        bad_probabilities = final_probs_np[:, 1]
    
    # 多种阈值优化策略
    optimization_metrics = ['f1_bad', 'f1_macro', 'precision_recall_balance', 'youden']
    optimal_results = {}
    
    for opt_metric in optimization_metrics:
        print(f"\n--- 优化指标: {opt_metric} ---")
        optimal_threshold, optimal_score, threshold_details = find_optimal_threshold(
            y_true, bad_probabilities, metric=opt_metric
        )
        
        # 使用最优阈值预测
        optimal_predictions = (bad_probabilities >= optimal_threshold).astype(int)
        
        # 计算详细指标
        final_accuracy = accuracy_score(y_true, optimal_predictions)
        final_f1_macro = f1_score(y_true, optimal_predictions, average='macro', zero_division=0)
        final_f1_weighted = f1_score(y_true, optimal_predictions, average='weighted', zero_division=0)
        
        final_f1_per_class = f1_score(y_true, optimal_predictions, average=None, zero_division=0)
        final_good_f1 = final_f1_per_class[0]
        final_bad_f1 = final_f1_per_class[1] if len(final_f1_per_class) > 1 else 0
        
        precision_per_class = precision_score(y_true, optimal_predictions, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, optimal_predictions, average=None, zero_division=0)
        
        final_good_precision = precision_per_class[0]
        final_good_recall = recall_per_class[0]
        final_bad_precision = precision_per_class[1] if len(precision_per_class) > 1 else 0
        final_bad_recall = recall_per_class[1] if len(recall_per_class) > 1 else 0
        
        optimal_results[opt_metric] = {
            'threshold': optimal_threshold,
            'predictions': optimal_predictions,
            'final_bad_f1': final_bad_f1,
            'final_good_f1': final_good_f1,
            'final_macro_f1': final_f1_macro,
            'final_weighted_f1': final_f1_weighted,
            'final_accuracy': final_accuracy,
            'final_bad_precision': final_bad_precision,
            'final_bad_recall': final_bad_recall,
            'final_good_precision': final_good_precision,
            'final_good_recall': final_good_recall,
            'threshold_details': threshold_details
        }
    
    # 选择最佳的阈值优化策略 (基于Bad F1)
    best_strategy = max(optimal_results.keys(), 
                       key=lambda k: optimal_results[k]['final_bad_f1'])
    best_result = optimal_results[best_strategy]
    
    print(f"\n🏆 推荐的最佳阈值策略: {best_strategy}")
    print(f"   最优阈值: {best_result['threshold']:.3f}")
    print(f"   Bad客户F1: {best_result['final_bad_f1']:.4f}")
    print(f"   Good客户F1: {best_result['final_good_f1']:.4f}")
    print(f"   宏平均F1: {best_result['final_macro_f1']:.4f}")
    print(f"   整体准确率: {best_result['final_accuracy']:.4f}")
    print(f"   Bad客户检出率: {best_result['final_bad_recall']:.4f}")
    print(f"   Bad客户预测准确率: {best_result['final_bad_precision']:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, best_result['predictions'])
    print(f"\n   📋 混淆矩阵 (最优阈值 {best_result['threshold']:.3f}):")
    print(f"              预测Good  预测Bad")
    print(f"   实际Good    {cm[0,0]:6d}   {cm[0,1]:6d}")
    if cm.shape[0] > 1:
        print(f"   实际Bad     {cm[1,0]:6d}   {cm[1,1]:6d}")
    
    return bad_probabilities, model, scaler, {
        'optimal_results': optimal_results,
        'best_strategy': best_strategy,
        'best_threshold': best_result['threshold'],
        'final_bad_f1': best_result['final_bad_f1'],
        'final_good_f1': best_result['final_good_f1'],
        'final_macro_f1': best_result['final_macro_f1'],
        'final_weighted_f1': best_result['final_weighted_f1'],
        'final_accuracy': best_result['final_accuracy'],
        'final_bad_precision': best_result['final_bad_precision'],
        'final_bad_recall': best_result['final_bad_recall'],
        'best_val_bad_f1': best_bad_f1,
        'train_history': train_history,
        'confusion_matrix': cm,
        'best_predictions': best_result['predictions']
    }

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
# 主程序 - Enhanced PyTorch Version
# =====================================================
def main():
    print("=== ULTRA Multi-Strategy Ensemble with Optimized Meta-ANN ===")
    
    # 数据加载
    print("\n=== Loading Data ===")
    features_path = '/Users/mannormal/4011/Qi Zihan/feature_extraction/generated_features/all_features_super_optimized.csv'
    all_features_df = pd.read_csv(features_path)

    pwd = '/Users/mannormal/4011/Qi Zihan/original_data/'
    ta = pd.read_csv(pwd + 'train_acc.csv')
    te = pd.read_csv(pwd + 'test_acc_predict.csv')
    ta.loc[ta['flag'] == 0, 'flag'] = -1

    strategy_data = load_strategy_categories()
    training_df = pd.merge(all_features_df, ta[['account', 'flag']], on='account', how='inner')
    training_df['account_type'] = training_df.apply(classify_account_type_original, axis=1)

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
    # Phase 3: 优化的Meta-ANN训练 (替换原有的PyTorch Meta-ANN部分)
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 PHASE 3: 优化Meta-ANN训练 (重点: Bad客户F1)")
    print(f"{'='*80}")
    
    meta_predictions, meta_model, feature_scaler, meta_results = train_refined_meta_ann(
        base_predictions=combined_base_predictions,
        original_features=original_features,
        y_true=y_true,
        n_epochs=500,
        patience=20
    )
    
    # =====================================================
    # Phase 4: 增强的交叉验证分析
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 PHASE 4: 增强交叉验证分析 (专注Bad客户F1)")
    print(f"{'='*80}")
    
    # 10折交叉验证评估
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = []
    
    print("\nFold | Bad F1   | Good F1  | Macro F1 | Accuracy | Bad Prec | Bad Recall | Status")
    print("-" * 80)
    
    for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(combined_base_predictions, y_true), desc="CV Folds")):
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
        
        # 计算类别权重
        class_counts = np.bincount(y_train_fold)
        class_weights = torch.FloatTensor(len(y_train_fold) / (len(class_counts) * class_counts))
        
        # 训练Meta-ANN
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model_fold = RefinedMetaANN(
            n_base_models=X_base_train.shape[1], 
            n_original_features=X_feat_train_scaled.shape[1]
        ).to(device)
        
        optimizer = optim.AdamW(model_fold.parameters(), lr=1e-3, weight_decay=5e-4)
        criterion = LightLabelSmoothingCE(smoothing=0.03, class_weights=class_weights.to(device))
        
        # 转换为张量
        X_base_train_t = torch.tensor(X_base_train, dtype=torch.float32).to(device)
        X_feat_train_t = torch.tensor(X_feat_train_scaled, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train_fold, dtype=torch.long).to(device)
        
        X_base_val_t = torch.tensor(X_base_val, dtype=torch.float32).to(device)
        X_feat_val_t = torch.tensor(X_feat_val_scaled, dtype=torch.float32).to(device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_base_train_t, X_feat_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # 快速训练（用于CV）
        for epoch in range(100):
            model_fold.train()
            for batch_xb, batch_xf, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model_fold(batch_xb, batch_xf)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # 评估
        model_fold.eval()
        with torch.no_grad():
            val_outputs = model_fold(X_base_val_t, X_feat_val_t)
            _, val_preds = torch.max(val_outputs, 1)
            val_label = val_preds.cpu().numpy()
            
            # 计算详细指标
            val_accuracy = accuracy_score(y_val_fold, val_label)
            val_f1_macro = f1_score(y_val_fold, val_label, average='macro', zero_division=0)
            
            val_f1_per_class = f1_score(y_val_fold, val_label, average=None, zero_division=0)
            good_f1 = val_f1_per_class[0]
            bad_f1 = val_f1_per_class[1] if len(val_f1_per_class) > 1 else 0
            
            precision_per_class = precision_score(y_val_fold, val_label, average=None, zero_division=0)
            recall_per_class = recall_score(y_val_fold, val_label, average=None, zero_division=0)
            
            bad_precision = precision_per_class[1] if len(precision_per_class) > 1 else 0
            bad_recall = recall_per_class[1] if len(recall_per_class) > 1 else 0
            
            status = "Strong" if bad_f1 > 0.6 else "Good" if bad_f1 > 0.4 else "Weak"
            
            print(f"{fold+1:4d} | {bad_f1:8.4f} | {good_f1:8.4f} | {val_f1_macro:8.4f} | "
                  f"{val_accuracy:8.4f} | {bad_precision:8.4f} | {bad_recall:9.4f} | {status}")
            
            cv_results.append({
                'fold': fold + 1,
                'bad_f1': bad_f1,
                'good_f1': good_f1,
                'macro_f1': val_f1_macro,
                'accuracy': val_accuracy,
                'bad_precision': bad_precision,
                'bad_recall': bad_recall
            })

    # CV统计
    avg_bad_f1 = np.mean([r['bad_f1'] for r in cv_results])
    avg_good_f1 = np.mean([r['good_f1'] for r in cv_results])
    avg_macro_f1 = np.mean([r['macro_f1'] for r in cv_results])
    avg_accuracy = np.mean([r['accuracy'] for r in cv_results])
    avg_bad_precision = np.mean([r['bad_precision'] for r in cv_results])
    avg_bad_recall = np.mean([r['bad_recall'] for r in cv_results])

    print("-" * 80)
    print(f"Avg  | {avg_bad_f1:8.4f} | {avg_good_f1:8.4f} | {avg_macro_f1:8.4f} | "
          f"{avg_accuracy:8.4f} | {avg_bad_precision:8.4f} | {avg_bad_recall:9.4f} | Summary")

    print(f"\n🤖 优化Meta-ANN性能总结:")
    print(f"   训练集Bad F1: {meta_results['final_bad_f1']:.4f}")
    print(f"   交叉验证Bad F1: {avg_bad_f1:.4f} ± {np.std([r['bad_f1'] for r in cv_results]):.4f}")
    print(f"   Bad客户检出率: {avg_bad_recall:.4f} ± {np.std([r['bad_recall'] for r in cv_results]):.4f}")
    print(f"   Bad客户预测准确率: {avg_bad_precision:.4f} ± {np.std([r['bad_precision'] for r in cv_results]):.4f}")
    print(f"   整体准确率: {avg_accuracy:.4f} ± {np.std([r['accuracy'] for r in cv_results]):.4f}")

    generalization_gap = meta_results['final_bad_f1'] - avg_bad_f1
    if generalization_gap > 0.1:
        print(f"   ⚠️ 警告: 高过拟合 (差距: {generalization_gap:+.4f})")
    elif generalization_gap > 0.05:
        print(f"   ⚠️ 注意: 中度过拟合 (差距: {generalization_gap:+.4f})")
    else:
        print(f"   ✅ 良好: 低过拟合 (差距: {generalization_gap:+.4f})")

    # =====================================================
    # Phase 5: 测试集预测 & 生成提交文件 (修改部分)
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 PHASE 5: Test Set Prediction & Submission Generation")
    print(f"{'='*80}")
    
    # 加载测试数据
    test_df = pd.merge(all_features_df, te[['account']], on='account', how='inner')
    test_df['account_type'] = test_df.apply(classify_account_type_original, axis=1)
    
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
    
    # 4. 优化Meta-ANN测试预测
    print("🤖 优化Meta-ANN测试预测...")
    test_original_features = test_df[feature_cols].values
    test_original_features_scaled = feature_scaler.transform(test_original_features)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    X_test_base_tensor = torch.tensor(test_combined_base_predictions, dtype=torch.float32).to(device)
    X_test_feat_tensor = torch.tensor(test_original_features_scaled, dtype=torch.float32).to(device)
    
    meta_model.eval()
    with torch.no_grad():
        test_outputs = meta_model(X_test_base_tensor, X_test_feat_tensor)
        test_probabilities = F.softmax(test_outputs, dim=1)
        _, test_final_predictions = torch.max(test_outputs, 1)
        
        test_final_labels = test_final_predictions.cpu().numpy()
        test_bad_probabilities = test_probabilities[:, 1].cpu().numpy()  # Bad客户的概率
    
    # =====================================================
    # 生成提交文件部分修改
    # =====================================================
    print(f"\n💾 保存提交文件...")
    
    # 创建提交DataFrame
    submission_df = pd.DataFrame({
        'account': test_df['account'].values,
        'flag': test_final_labels
    })
    
    # 统计预测结果
    pred_counts = submission_df['flag'].value_counts()
    print(f"📊 测试预测总结:")
    print(f"   总测试账户: {len(submission_df)}")
    print(f"   预测Good (0): {pred_counts.get(0, 0)} ({pred_counts.get(0, 0)/len(submission_df)*100:.1f}%)")
    print(f"   预测Bad (1): {pred_counts.get(1, 0)} ({pred_counts.get(1, 0)/len(submission_df)*100:.1f}%)")
    
    # 生成文件名，使用交叉验证的Bad F1分数
    mean_cv_bad_f1 = avg_bad_f1
    filename = f"optimized_meta_ann_bad_f1_{mean_cv_bad_f1:.4f}.csv"
    filepath = f"/Users/mannormal/4011/Qi Zihan/result_analysis/prediction_results/{filename}"
    
    # 保存文件
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    submission_df.to_csv(filepath, index=False)
    
    print(f"✅ 提交文件已保存: {filename}")
    print(f"📁 完整路径: {filepath}")
    
    # =====================================================
    # 额外保存详细分析文件
    # =====================================================
    # 保存测试集的预测概率用于进一步分析
    detailed_test_results = pd.DataFrame({
        'account': test_df['account'].values,
        'predicted_label': test_final_labels,
        'bad_probability': test_bad_probabilities,
        'confidence': np.max([1 - test_bad_probabilities, test_bad_probabilities], axis=0)
    })
    
    detailed_filename = f"detailed_test_results_bad_f1_{mean_cv_bad_f1:.4f}.csv"
    detailed_filepath = f"/Users/mannormal/4011/Qi Zihan/result_analysis/prediction_results/{detailed_filename}"
    detailed_test_results.to_csv(detailed_filepath, index=False)
    
    print(f"📄 详细测试结果已保存: {detailed_filename}")
    
    # 置信度分析
    high_confidence_bad = detailed_test_results[
        (detailed_test_results['predicted_label'] == 1) & 
        (detailed_test_results['confidence'] > 0.8)
    ]
    high_confidence_good = detailed_test_results[
        (detailed_test_results['predicted_label'] == 0) & 
        (detailed_test_results['confidence'] > 0.8)
    ]
    
    print(f"\n📈 置信度分析:")
    print(f"   高置信度Bad预测: {len(high_confidence_bad)} 个")
    print(f"   高置信度Good预测: {len(high_confidence_good)} 个")
    print(f"   高置信度预测比例: {(len(high_confidence_bad) + len(high_confidence_good))/len(submission_df)*100:.1f}%")
    
    return {
        'meta_model': meta_model,
        'feature_scaler': feature_scaler,
        'rf_predictions': rf_predictions,
        'strategy_predictions': all_strategy_predictions,
        'cv_results': cv_results,
        'meta_results': meta_results,
        'final_bad_f1': avg_bad_f1,  # 使用交叉验证的Bad F1
        'submission_df': submission_df,
        'submission_filepath': filepath,
        'detailed_results': detailed_test_results,
        'detailed_filepath': detailed_filepath
    }

# 在imports部分添加
from sklearn.metrics import roc_curve, precision_recall_curve

# 添加阈值优化函数
def find_optimal_threshold(y_true, y_prob, metric='f1_bad', plot=False):
    """
    寻找最优分类阈值
    
    Args:
        y_true: 真实标签
        y_prob: 预测概率 (Bad客户的概率)
        metric: 优化指标 ('f1_bad', 'f1_macro', 'precision_recall_balance', 'youden')
        plot: 是否绘制阈值曲线
    
    Returns:
        最优阈值和对应的指标值
    """
    thresholds = np.arange(0.1, 0.9, 0.01)  # 从0.1到0.9，步长0.01
    metrics_scores = []
    
    best_threshold = 0.5
    best_score = 0
    threshold_details = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # 计算各种指标
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 分类别指标
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        good_f1 = f1_per_class[0] if len(f1_per_class) > 0 else 0
        bad_f1 = f1_per_class[1] if len(f1_per_class) > 1 else 0
        
        good_precision = precision_per_class[0] if len(precision_per_class) > 0 else 0
        bad_precision = precision_per_class[1] if len(precision_per_class) > 1 else 0
        
        good_recall = recall_per_class[0] if len(recall_per_class) > 0 else 0
        bad_recall = recall_per_class[1] if len(recall_per_class) > 1 else 0
        
        # 计算不同的评估指标
        if metric == 'f1_bad':
            score = bad_f1
        elif metric == 'f1_macro':
            score = f1_macro
        elif metric == 'precision_recall_balance':
            # Bad客户的精确率和召回率的调和平均
            if bad_precision + bad_recall > 0:
                score = 2 * (bad_precision * bad_recall) / (bad_precision + bad_recall)
            else:
                score = 0
        elif metric == 'youden':
            # Youden's J statistic = Sensitivity + Specificity - 1
            sensitivity = bad_recall  # 对Bad客户的召回率
            specificity = good_recall  # 对Good客户的召回率 (真负率)
            score = sensitivity + specificity - 1
        else:
            score = bad_f1  # 默认使用Bad F1
        
        metrics_scores.append(score)
        
        threshold_details.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'bad_f1': bad_f1,
            'good_f1': good_f1,
            'bad_precision': bad_precision,
            'bad_recall': bad_recall,
            'good_precision': good_precision,
            'good_recall': good_recall,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    # 找到最优阈值的详细信息
    best_details = next(d for d in threshold_details if d['threshold'] == best_threshold)
    
    print(f"\n🎯 最优阈值优化结果 (优化指标: {metric}):")
    print(f"   最优阈值: {best_threshold:.3f}")
    print(f"   优化指标分数: {best_score:.4f}")
    print(f"   Bad客户F1: {best_details['bad_f1']:.4f}")
    print(f"   Good客户F1: {best_details['good_f1']:.4f}")
    print(f"   宏平均F1: {best_details['f1_macro']:.4f}")
    print(f"   Bad客户精确率: {best_details['bad_precision']:.4f}")
    print(f"   Bad客户召回率: {best_details['bad_recall']:.4f}")
    print(f"   整体准确率: {best_details['accuracy']:.4f}")
    
    # 显示阈值选择的影响
    default_details = next(d for d in threshold_details if abs(d['threshold'] - 0.5) < 0.01)
    print(f"\n📊 相比默认阈值0.5的改进:")
    print(f"   Bad F1: {best_details['bad_f1']:.4f} vs {default_details['bad_f1']:.4f} "
          f"(改进: {best_details['bad_f1'] - default_details['bad_f1']:+.4f})")
    print(f"   宏平均F1: {best_details['f1_macro']:.4f} vs {default_details['f1_macro']:.4f} "
          f"(改进: {best_details['f1_macro'] - default_details['f1_macro']:+.4f})")
    
    return best_threshold, best_score, threshold_details

def evaluate_with_optimal_threshold(model, scaler, X_base, X_feat, y_true, 
                                  optimization_metric='f1_bad', device='cpu'):
    """
    使用最优阈值评估模型
    """
    model.eval()
    with torch.no_grad():
        X_base_tensor = torch.tensor(X_base, dtype=torch.float32).to(device)
        X_feat_scaled = scaler.transform(X_feat)
        X_feat_tensor = torch.tensor(X_feat_scaled, dtype=torch.float32).to(device)
        
        outputs = model(X_base_tensor, X_feat_tensor)
        probabilities = F.softmax(outputs, dim=1)
        bad_probabilities = probabilities[:, 1].cpu().numpy()
    
    # 寻找最优阈值
    optimal_threshold, optimal_score, threshold_details = find_optimal_threshold(
        y_true, bad_probabilities, metric=optimization_metric
    )
    
    # 使用最优阈值进行预测
    optimal_predictions = (bad_probabilities >= optimal_threshold).astype(int)
    
    return optimal_predictions, optimal_threshold, bad_probabilities, threshold_details

# 修改train_refined_meta_ann函数，在最终评估部分添加阈值优化
def train_refined_meta_ann(base_predictions, original_features, y_true, n_epochs=500, patience=20):
    """
    训练优化的Meta-ANN，重点关注Bad客户F1分数，包含阈值优化
    
    Args:
        base_predictions: (n_samples, n_models) - 基础模型预测概率
        original_features: (n_samples, n_features) - 原始特征
        y_true: (n_samples,) - 真实标签 (0: Good, 1: Bad)
    
    Returns:
        训练好的模型、预测结果和详细指标
    """
    print(f"\n🤖 训练优化Meta-ANN (重点: Bad客户F1)")
    print(f"Base predictions shape: {base_predictions.shape}")
    print(f"Original features shape: {original_features.shape}")
    print(f"Label distribution: {dict(zip(*np.unique(y_true, return_counts=True)))}")
    
    # 特征标准化
    scaler = StandardScaler()
    original_features_scaled = scaler.fit_transform(original_features)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 转换为PyTorch张量
    X_base_tensor = torch.tensor(base_predictions, dtype=torch.float32).to(device)
    X_feat_tensor = torch.tensor(original_features_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_true, dtype=torch.long).to(device)  # 注意这里用LongTensor
    
    # 计算类别权重
    class_counts = np.bincount(y_true)
    class_weights = torch.FloatTensor(len(y_true) / (len(class_counts) * class_counts)).to(device)
    print(f"Class weights: {class_weights}")
    
    # 创建优化的模型
    model = RefinedMetaANN(
        n_base_models=base_predictions.shape[1], 
        n_original_features=original_features_scaled.shape[1],
        dropout_rates=[0.3, 0.2, 0.2, 0.1, 0.1]
    ).to(device)
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = LightLabelSmoothingCE(smoothing=0.03, class_weights=class_weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=8, factor=0.5, min_lr=1e-6)
    early_stopping = SimpleEarlyStopping(patience=patience, min_delta=0.001)
    
    # 分割训练和验证集
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(base_predictions, y_true))[0]
    
    Xb_train, Xb_val = X_base_tensor[train_idx], X_base_tensor[val_idx]
    Xf_train, Xf_val = X_feat_tensor[train_idx], X_feat_tensor[val_idx]
    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
    
    # 创建数据加载器
    train_dataset = TensorDataset(Xb_train, Xf_train, y_train)
    val_dataset = TensorDataset(Xb_val, Xf_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 训练历史记录
    train_history = []
    best_bad_f1 = 0
    
    print("\nEpoch | Train Loss | Bad F1 | Macro F1 | Accuracy | Good F1 | LR       | Status")
    print("-" * 80)
    
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_xb, batch_xf, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_xb, batch_xf)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_xb, batch_xf, batch_y in val_loader:
                outputs = model(batch_xb, batch_xf)
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        # 计算详细指标
        val_f1_macro = f1_score(val_targets, val_predictions, average='macro', zero_division=0)
        val_accuracy = accuracy_score(val_targets, val_predictions)
        
        # 计算各类别F1
        val_f1_per_class = f1_score(val_targets, val_predictions, average=None, zero_division=0)
        good_f1 = val_f1_per_class[0]
        bad_f1 = val_f1_per_class[1] if len(val_f1_per_class) > 1 else 0  # 主要指标
        
        # 学习率调度
        scheduler.step(bad_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'bad_f1': bad_f1,
            'macro_f1': val_f1_macro,
            'accuracy': val_accuracy,
            'good_f1': good_f1,
            'lr': current_lr
        }
        train_history.append(epoch_data)
        
        # 早停检查
        if bad_f1 > best_bad_f1:
            best_bad_f1 = bad_f1
            status = "✅ Best"
        else:
            status = f"⏳ {early_stopping.counter+1}/{early_stopping.patience}"
        
        # 打印进度
        if epoch % 20 == 0 or epoch >= n_epochs - 5:
            print(f"{epoch+1:5d} | {avg_train_loss:10.4f} | {bad_f1:6.4f} | {val_f1_macro:8.4f} | "
                  f"{val_accuracy:8.4f} | {good_f1:7.4f} | {current_lr:.2e} | {status}")
        
        if early_stopping(bad_f1, model):
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            print(f"🏆 Best Bad F1: {best_bad_f1:.4f}")
            break
    
    # =====================================================
    # 最终评估 - 添加阈值优化
    # =====================================================
    print(f"\n{'='*60}")
    print("🎯 阈值优化与最终评估")
    print(f"{'='*60}")
    
    model.eval()
    with torch.no_grad():
        # 获取概率预测
        final_outputs = model(X_base_tensor, X_feat_tensor)
        final_probs = F.softmax(final_outputs, dim=1)
        final_probs_np = final_probs.cpu().numpy()
        bad_probabilities = final_probs_np[:, 1]
    
    # 多种阈值优化策略
    optimization_metrics = ['f1_bad', 'f1_macro', 'precision_recall_balance', 'youden']
    optimal_results = {}
    
    for opt_metric in optimization_metrics:
        print(f"\n--- 优化指标: {opt_metric} ---")
        optimal_threshold, optimal_score, threshold_details = find_optimal_threshold(
            y_true, bad_probabilities, metric=opt_metric
        )
        
        # 使用最优阈值预测
        optimal_predictions = (bad_probabilities >= optimal_threshold).astype(int)
        
        # 计算详细指标
        final_accuracy = accuracy_score(y_true, optimal_predictions)
        final_f1_macro = f1_score(y_true, optimal_predictions, average='macro', zero_division=0)
        final_f1_weighted = f1_score(y_true, optimal_predictions, average='weighted', zero_division=0)
        
        final_f1_per_class = f1_score(y_true, optimal_predictions, average=None, zero_division=0)
        final_good_f1 = final_f1_per_class[0]
        final_bad_f1 = final_f1_per_class[1] if len(final_f1_per_class) > 1 else 0
        
        precision_per_class = precision_score(y_true, optimal_predictions, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, optimal_predictions, average=None, zero_division=0)
        
        final_good_precision = precision_per_class[0]
        final_good_recall = recall_per_class[0]
        final_bad_precision = precision_per_class[1] if len(precision_per_class) > 1 else 0
        final_bad_recall = recall_per_class[1] if len(recall_per_class) > 1 else 0
        
        optimal_results[opt_metric] = {
            'threshold': optimal_threshold,
            'predictions': optimal_predictions,
            'final_bad_f1': final_bad_f1,
            'final_good_f1': final_good_f1,
            'final_macro_f1': final_f1_macro,
            'final_weighted_f1': final_f1_weighted,
            'final_accuracy': final_accuracy,
            'final_bad_precision': final_bad_precision,
            'final_bad_recall': final_bad_recall,
            'final_good_precision': final_good_precision,
            'final_good_recall': final_good_recall,
            'threshold_details': threshold_details
        }
    
    # 选择最佳的阈值优化策略 (基于Bad F1)
    best_strategy = max(optimal_results.keys(), 
                       key=lambda k: optimal_results[k]['final_bad_f1'])
    best_result = optimal_results[best_strategy]
    
    print(f"\n🏆 推荐的最佳阈值策略: {best_strategy}")
    print(f"   最优阈值: {best_result['threshold']:.3f}")
    print(f"   Bad客户F1: {best_result['final_bad_f1']:.4f}")
    print(f"   Good客户F1: {best_result['final_good_f1']:.4f}")
    print(f"   宏平均F1: {best_result['final_macro_f1']:.4f}")
    print(f"   整体准确率: {best_result['final_accuracy']:.4f}")
    print(f"   Bad客户检出率: {best_result['final_bad_recall']:.4f}")
    print(f"   Bad客户预测准确率: {best_result['final_bad_precision']:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, best_result['predictions'])
    print(f"\n   📋 混淆矩阵 (最优阈值 {best_result['threshold']:.3f}):")
    print(f"              预测Good  预测Bad")
    print(f"   实际Good    {cm[0,0]:6d}   {cm[0,1]:6d}")
    if cm.shape[0] > 1:
        print(f"   实际Bad     {cm[1,0]:6d}   {cm[1,1]:6d}")
    
    return bad_probabilities, model, scaler, {
        'optimal_results': optimal_results,
        'best_strategy': best_strategy,
        'best_threshold': best_result['threshold'],
        'final_bad_f1': best_result['final_bad_f1'],
        'final_good_f1': best_result['final_good_f1'],
        'final_macro_f1': best_result['final_macro_f1'],
        'final_weighted_f1': best_result['final_weighted_f1'],
        'final_accuracy': best_result['final_accuracy'],
        'final_bad_precision': best_result['final_bad_precision'],
        'final_bad_recall': best_result['final_bad_recall'],
        'best_val_bad_f1': best_bad_f1,
        'train_history': train_history,
        'confusion_matrix': cm,
        'best_predictions': best_result['predictions']
    }

# 修改交叉验证部分，使用阈值优化
# 在main函数的Phase 4部分，添加阈值优化的交叉验证
def enhanced_cross_validation_with_threshold_optimization(combined_base_predictions, original_features, y_true, n_folds=5):
    """
    带阈值优化的增强交叉验证
    """
    print(f"\n🎯 Phase 4: 带阈值优化的增强交叉验证")
    print(f"{'='*80}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_results = []
    all_threshold_strategies = ['f1_bad', 'f1_macro', 'precision_recall_balance']
    
    print("\nFold | Strategy           | Threshold | Bad F1   | Good F1  | Macro F1 | Accuracy | Bad Prec | Bad Recall")
    print("-" * 105)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(combined_base_predictions, y_true)):
        # 数据分割
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
        
        # 计算类别权重
        class_counts = np.bincount(y_train_fold)
        class_weights = torch.FloatTensor(len(y_train_fold) / (len(class_counts) * class_counts))
        
        # 训练Meta-ANN
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model_fold = RefinedMetaANN(
            n_base_models=X_base_train.shape[1], 
            n_original_features=X_feat_train_scaled.shape[1]
        ).to(device)
        
        optimizer = optim.AdamW(model_fold.parameters(), lr=1e-3, weight_decay=5e-4)
        criterion = LightLabelSmoothingCE(smoothing=0.03, class_weights=class_weights.to(device))
        
        # 转换为张量
        X_base_train_t = torch.tensor(X_base_train, dtype=torch.float32).to(device)
        X_feat_train_t = torch.tensor(X_feat_train_scaled, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train_fold, dtype=torch.long).to(device)
        
        X_base_val_t = torch.tensor(X_base_val, dtype=torch.float32).to(device)
        X_feat_val_t = torch.tensor(X_feat_val_scaled, dtype=torch.float32).to(device)
        
        # 快速训练
        train_dataset = TensorDataset(X_base_train_t, X_feat_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        for epoch in range(80):  # 减少epoch用于CV
            model_fold.train()
            for batch_xb, batch_xf, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model_fold(batch_xb, batch_xf)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # 获取验证集概率预测
        model_fold.eval()
        with torch.no_grad():
            val_outputs = model_fold(X_base_val_t, X_feat_val_t)
            val_probabilities = F.softmax(val_outputs, dim=1)
            val_bad_probs = val_probabilities[:, 1].cpu().numpy()
        
        # 对每种阈值优化策略进行评估
        fold_results = {'fold': fold + 1}
        
        for strategy in all_threshold_strategies:
            # 寻找最优阈值
            optimal_threshold, _, _ = find_optimal_threshold(
                y_val_fold, val_bad_probs, metric=strategy
            )
            
            # 使用最优阈值预测
            val_predictions = (val_bad_probs >= optimal_threshold).astype(int)
            
            # 计算指标
            val_accuracy = accuracy_score(y_val_fold, val_predictions)
            val_f1_macro = f1_score(y_val_fold, val_predictions, average='macro', zero_division=0)
            
            val_f1_per_class = f1_score(y_val_fold, val_predictions, average=None, zero_division=0)
            good_f1 = val_f1_per_class[0]
            bad_f1 = val_f1_per_class[1] if len(val_f1_per_class) > 1 else 0
            
            precision_per_class = precision_score(y_val_fold, val_predictions, average=None, zero_division=0)
            recall_per_class = recall_score(y_val_fold, val_predictions, average=None, zero_division=0)
            
            bad_precision = precision_per_class[1] if len(precision_per_class) > 1 else 0
            bad_recall = recall_per_class[1] if len(recall_per_class) > 1 else 0
            
            fold_results[f'{strategy}_threshold'] = optimal_threshold
            fold_results[f'{strategy}_bad_f1'] = bad_f1
            fold_results[f'{strategy}_good_f1'] = good_f1
            fold_results[f'{strategy}_macro_f1'] = val_f1_macro
            fold_results[f'{strategy}_accuracy'] = val_accuracy
            fold_results[f'{strategy}_bad_precision'] = bad_precision
            fold_results[f'{strategy}_bad_recall'] = bad_recall
            
            print(f"{fold+1:4d} | {strategy:18s} | {optimal_threshold:9.3f} | {bad_f1:8.4f} | {good_f1:8.4f} | "
                  f"{val_f1_macro:8.4f} | {val_accuracy:8.4f} | {bad_precision:8.4f} | {bad_recall:9.4f}")
        
        cv_results.append(fold_results)
    
    # 汇总CV结果
    print("-" * 105)
    
    summary_results = {}
    for strategy in all_threshold_strategies:
        avg_threshold = np.mean([r[f'{strategy}_threshold'] for r in cv_results])
        avg_bad_f1 = np.mean([r[f'{strategy}_bad_f1'] for r in cv_results])
        avg_good_f1 = np.mean([r[f'{strategy}_good_f1'] for r in cv_results])
        avg_macro_f1 = np.mean([r[f'{strategy}_macro_f1'] for r in cv_results])
        avg_accuracy = np.mean([r[f'{strategy}_accuracy'] for r in cv_results])
        avg_bad_precision = np.mean([r[f'{strategy}_bad_precision'] for r in cv_results])
        avg_bad_recall = np.mean([r[f'{strategy}_bad_recall'] for r in cv_results])
        
        summary_results[strategy] = {
            'avg_threshold': avg_threshold,
            'avg_bad_f1': avg_bad_f1,
            'avg_good_f1': avg_good_f1,
            'avg_macro_f1': avg_macro_f1,
            'avg_accuracy': avg_accuracy,
            'avg_bad_precision': avg_bad_precision,
            'avg_bad_recall': avg_bad_recall,
            'std_bad_f1': np.std([r[f'{strategy}_bad_f1'] for r in cv_results])
        }
        
        print(f"Avg  | {strategy:18s} | {avg_threshold:9.3f} | {avg_bad_f1:8.4f} | {avg_good_f1:8.4f} | "
              f"{avg_macro_f1:8.4f} | {avg_accuracy:8.4f} | {avg_bad_precision:8.4f} | {avg_bad_recall:9.4f}")
    
    # 找到最佳策略
    best_cv_strategy = max(summary_results.keys(), 
                          key=lambda k: summary_results[k]['avg_bad_f1'])
    best_cv_result = summary_results[best_cv_strategy]
    
    print(f"\n🏆 交叉验证最佳阈值策略: {best_cv_strategy}")
    print(f"   平均最优阈值: {best_cv_result['avg_threshold']:.3f}")
    print(f"   平均Bad F1: {best_cv_result['avg_bad_f1']:.4f} ± {best_cv_result['std_bad_f1']:.4f}")
    print(f"   平均宏F1: {best_cv_result['avg_macro_f1']:.4f}")
    print(f"   平均准确率: {best_cv_result['avg_accuracy']:.4f}")
    
    return cv_results, summary_results, best_cv_strategy, best_cv_result

# 修改测试集预测部分，使用最优阈值
# 在Phase 5的最后，使用最优阈值进行测试集预测
def predict_test_with_optimal_threshold(meta_model, feature_scaler, test_combined_base_predictions, 
                                      test_original_features, optimal_threshold, device):
    """
    使用最优阈值对测试集进行预测
    """
    print(f"🤖 使用最优阈值 {optimal_threshold:.3f} 进行测试预测...")
    
    test_original_features_scaled = feature_scaler.transform(test_original_features)
    
    X_test_base_tensor = torch.tensor(test_combined_base_predictions, dtype=torch.float32).to(device)
    X_test_feat_tensor = torch.tensor(test_original_features_scaled, dtype=torch.float32).to(device)
    
    meta_model.eval()
    with torch.no_grad():
        test_outputs = meta_model(X_test_base_tensor, X_test_feat_tensor)
        test_probabilities = F.softmax(test_outputs, dim=1)
        test_bad_probabilities = test_probabilities[:, 1].cpu().numpy()
        
        # 使用最优阈值进行分类
        test_final_labels = (test_bad_probabilities >= optimal_threshold).astype(int)
    
    return test_final_labels, test_bad_probabilities