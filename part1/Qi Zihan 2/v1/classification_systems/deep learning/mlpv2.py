import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from datetime import datetime
import math

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ========== 渐进式复杂度的模型设计 ==========
class BaselineMLP(nn.Module):
    """简单版: 30→128→64→32→2 (约4k参数)"""
    
    def __init__(self, input_dim=30, dropout_rates=[0.3, 0.2, 0.1]):
        super(BaselineMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rates[2]),
            
            nn.Linear(32, 2)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class ModerateMLP(nn.Module):
    """中等版: 30→192→96→48→2 (约12k参数)"""
    
    def __init__(self, input_dim=30, dropout_rates=[0.3, 0.25, 0.2, 0.1]):
        super(ModerateMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            
            nn.Linear(192, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            
            nn.Linear(96, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(dropout_rates[2]),
            
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(dropout_rates[3]),
            
            nn.Linear(24, 2)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class DeepMLP(nn.Module):
    """复杂版: 30→256→128→64→32→2 (约25k参数)"""
    
    def __init__(self, input_dim=30, dropout_rates=[0.35, 0.3, 0.25, 0.2, 0.1]):
        super(DeepMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            
            nn.Linear(128, 64),
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
            
            nn.Linear(16, 2)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

# ========== 修正的Label Smoothing损失函数 ==========
class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing交叉熵损失"""
    def __init__(self, num_classes=2, smoothing=0.1, class_weights=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.class_weights = class_weights
        
    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=1)
        
        # Label smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = torch.mean(torch.sum(-true_dist * log_prob, dim=1))
        
        # 类别权重
        if self.class_weights is not None:
            weights = self.class_weights[target]
            loss = loss * weights.mean()
        
        return loss

# ========== 修正的Dropout调度器 ==========
class ImprovedDropoutScheduler:
    """改进的Dropout调度器 - 只在验证指标改善时降低dropout"""
    def __init__(self, model, initial_rates, min_rates=None, decay_factor=0.9):
        self.model = model
        self.initial_rates = initial_rates.copy()
        self.min_rates = min_rates or [rate * 0.6 for rate in initial_rates]
        self.decay_factor = decay_factor
        self.current_rates = initial_rates.copy()
        self.last_improvement_epoch = 0
        
    def step(self, epoch, val_improving=True):
        """只在验证指标持续改善且经过足够轮数时才降低dropout"""
        if val_improving:
            self.last_improvement_epoch = epoch
        
        # 只有在持续改善且经过热身期后才降低dropout
        epochs_since_improvement = epoch - self.last_improvement_epoch
        if val_improving and epoch > 20 and epochs_since_improvement < 5:
            for i, (current, minimum) in enumerate(zip(self.current_rates, self.min_rates)):
                self.current_rates[i] = max(current * self.decay_factor, minimum)
            self._apply_dropout_rates()
    
    def _apply_dropout_rates(self):
        """应用新的dropout率到模型"""
        dropout_idx = 0
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                if dropout_idx < len(self.current_rates):
                    module.p = self.current_rates[dropout_idx]
                    dropout_idx += 1

# ========== 余弦退火学习率调度器 ==========
class CosineAnnealingWarmupScheduler:
    """带热身的余弦退火学习率调度器"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0, warmup_start_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# ========== 早停法 ==========
class EarlyStopping:
    """简化的早停法"""
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0
        
    def __call__(self, epoch, score, model):
        is_better = False
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            is_better = True
        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            is_better = True
        else:
            self.counter += 1
            
        if is_better and self.restore_best_weights:
            self.save_checkpoint(model)
            
        should_stop = self.counter >= self.patience
        if should_stop and self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            
        return should_stop, is_better
    
    def save_checkpoint(self, model):
        self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}

# ========== 特征选择 ==========
def feature_selection_analysis(X, y, feature_names, top_k=20):
    """特征重要性分析和选择"""
    print(f"\n🔍 特征选择分析 (选择前{top_k}个特征)...")
    
    # 使用F统计量和互信息两种方法
    print("计算F统计量重要性...")
    f_selector = SelectKBest(f_classif, k=top_k)
    X_f_selected = f_selector.fit_transform(X, y)
    f_scores = f_selector.scores_
    f_selected_features = [feature_names[i] for i in f_selector.get_support(indices=True)]
    
    print("计算互信息重要性...")
    mi_selector = SelectKBest(mutual_info_classif, k=top_k)
    X_mi_selected = mi_selector.fit_transform(X, y)
    mi_scores = mi_selector.scores_
    mi_selected_features = [feature_names[i] for i in mi_selector.get_support(indices=True)]
    
    # 找到两种方法都选中的特征
    common_features = list(set(f_selected_features) & set(mi_selected_features))
    
    print(f"\n📊 特征选择结果:")
    print(f"F统计量选择: {len(f_selected_features)} 个特征")
    print(f"互信息选择: {len(mi_selected_features)} 个特征")
    print(f"共同选择: {len(common_features)} 个特征")
    
    # 显示前10个最重要的特征
    f_importance = list(zip(feature_names, f_scores))
    f_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 F统计量 Top 10 特征:")
    for i, (name, score) in enumerate(f_importance[:10]):
        print(f"  {i+1:2d}. {name:30} | F-Score: {score:8.2f}")
    
    # 返回共同特征的索引，如果不够则补充F统计量高的
    if len(common_features) >= top_k // 2:
        selected_indices = [i for i, name in enumerate(feature_names) if name in common_features]
    else:
        selected_indices = f_selector.get_support(indices=True)
    
    return selected_indices[:top_k], [feature_names[i] for i in selected_indices[:top_k]]

# ========== 快速超参数搜索 ==========
def quick_hyperparameter_search(X, y, feature_names, model_configs):
    """用3折CV快速筛选最佳配置"""
    print(f"\n⚡ 快速超参数搜索 (3折CV预筛选)...")
    
    # 配置候选
    search_configs = [
        {'weight_decay': 1e-4, 'lr': 0.001, 'smoothing': 0.05},
        {'weight_decay': 5e-4, 'lr': 0.001, 'smoothing': 0.1},
        {'weight_decay': 1e-3, 'lr': 0.0008, 'smoothing': 0.1},
        {'weight_decay': 1e-4, 'lr': 0.0012, 'smoothing': 0.15},
    ]
    
    best_results = {}
    
    # 计算类别权重
    class_counts = np.bincount(y)
    class_weights = torch.FloatTensor(len(y) / (len(class_counts) * class_counts))
    
    for model_name, model_class in model_configs.items():
        print(f"\n🎯 测试模型: {model_name}")
        model_results = []
        
        for i, config in enumerate(search_configs):
            print(f"  配置 {i+1}/{len(search_configs)}: {config}")
            
            # 3折交叉验证
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            fold_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # 转换为tensor
                X_train_tensor = torch.FloatTensor(X_train.values)
                y_train_tensor = torch.LongTensor(y_train)
                X_val_tensor = torch.FloatTensor(X_val.values)
                y_val_tensor = torch.LongTensor(y_val)
                
                # 数据加载器
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
                
                # 模型训练
                model = model_class(input_dim=X.shape[1]).to(device)
                fold_f1 = train_quick_model(model, train_loader, val_loader, config, class_weights)
                fold_scores.append(fold_f1)
            
            avg_f1 = np.mean(fold_scores)
            model_results.append({
                'config': config,
                'f1_mean': avg_f1,
                'f1_std': np.std(fold_scores)
            })
            
            print(f"    平均F1: {avg_f1:.4f} ± {np.std(fold_scores):.4f}")
        
        # 找到该模型的最佳配置
        best_config = max(model_results, key=lambda x: x['f1_mean'])
        best_results[model_name] = best_config
        
        print(f"\n✅ {model_name} 最佳配置:")
        print(f"   F1: {best_config['f1_mean']:.4f} ± {best_config['f1_std']:.4f}")
        print(f"   配置: {best_config['config']}")
    
    return best_results

def train_quick_model(model, train_loader, val_loader, config, class_weights, max_epochs=50):
    """快速训练模型用于超参数搜索"""
    
    criterion = LabelSmoothingCrossEntropy(
        num_classes=2, smoothing=config['smoothing'], class_weights=class_weights.to(device)
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer, warmup_epochs=5, max_epochs=max_epochs, eta_min=config['lr']*0.01
    )
    
    early_stopping = EarlyStopping(patience=8, min_delta=0.001)
    
    best_f1 = 0
    
    for epoch in range(max_epochs):
        # 训练
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # 验证
        model.eval()
        val_predictions, val_targets = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_f1 = f1_score(val_targets, val_predictions, average='weighted')
        best_f1 = max(best_f1, val_f1)
        
        scheduler.step(epoch)
        
        should_stop, _ = early_stopping(epoch, val_f1, model)
        if should_stop:
            break
    
    return best_f1

# ========== 完整训练函数 ==========
def train_optimized_model(model, train_loader, val_loader, config, class_weights, epochs=100):
    """优化的训练函数"""
    
    print(f"🚀 开始优化训练...")
    print(f"   模型: {model.__class__.__name__}")
    print(f"   配置: {config}")
    
    criterion = LabelSmoothingCrossEntropy(
        num_classes=2, smoothing=config['smoothing'], class_weights=class_weights.to(device)
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer, warmup_epochs=10, max_epochs=epochs, eta_min=config['lr']*0.01
    )
    
    # Dropout调度器
    if hasattr(model, 'network'):
        initial_dropout = [0.3, 0.25, 0.2, 0.1]  # 根据模型调整
        dropout_scheduler = ImprovedDropoutScheduler(model, initial_dropout)
    else:
        dropout_scheduler = None
    
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    train_losses = []
    val_f1_scores = []
    learning_rates = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_predictions, val_targets = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_f1 = f1_score(val_targets, val_predictions, average='weighted')
        avg_train_loss = train_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_f1_scores.append(val_f1)
        
        # 学习率调度
        current_lr = scheduler.step(epoch)
        learning_rates.append(current_lr)
        
        # Dropout调度
        if dropout_scheduler is not None:
            is_improving = len(val_f1_scores) < 2 or val_f1 > max(val_f1_scores[:-1])
            dropout_scheduler.step(epoch, is_improving)
        
        # 早停检查
        should_stop, is_improving = early_stopping(epoch, val_f1, model)
        if should_stop:
            print(f"Early stopping at epoch {epoch+1}, best epoch: {early_stopping.best_epoch+1}")
            break
        
        if epoch % 15 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val F1: {val_f1:.4f}, '
                  f'LR: {current_lr:.2e}')
    
    return train_losses, val_f1_scores, learning_rates

# ========== 主函数 ==========
def main_optimized():
    """优化版主函数 - 渐进式改进"""
    
    print("="*80)
    print("🎯 MLP渐进式优化训练")
    print("="*80)
    
    # 1. 数据加载
    print("\n📂 加载数据...")
    data_path = "/Users/mannormal/4011/Qi Zihan/feature_extraction/generated_features/all_features_complete.csv"
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
    
    print(f"训练数据: {df_train.shape}")
    print(f"标签分布: {np.bincount(df_train['label'])}")
    
    # 2. 特征预处理
    feature_cols = [col for col in df.columns if col != 'account']
    from mlp import preprocess_features  # 复用原有预处理
    X_train, scaler = preprocess_features(df_train, feature_cols)
    y_train = df_train['label'].values
    
    print(f"预处理后特征数: {X_train.shape[1]}")
    
    # 3. 特征选择分析
    selected_indices, selected_features = feature_selection_analysis(
        X_train.values, y_train, X_train.columns.tolist(), top_k=25
    )
    
    # 创建特征选择版本的数据
    X_train_selected = X_train.iloc[:, selected_indices]
    print(f"特征选择后: {X_train_selected.shape[1]} 个特征")
    
    # 4. 定义模型配置 - 渐进式复杂度
    model_configs = {
        'baseline': BaselineMLP,     # ~4k 参数
        'moderate': ModerateMLP,     # ~12k 参数
        'deep': DeepMLP,            # ~25k 参数
    }
    
    # 5. Phase 1: 快速超参数搜索 (3折CV)
    print(f"\n{'='*60}")
    print("🔬 Phase 1: 快速超参数搜索")
    print(f"{'='*60}")
    
    # 测试全特征版本
    print("\n📊 测试全特征版本...")
    best_configs_full = quick_hyperparameter_search(X_train, y_train, X_train.columns.tolist(), model_configs)
    
    # 测试特征选择版本
    print("\n📊 测试特征选择版本...")
    best_configs_selected = quick_hyperparameter_search(X_train_selected, y_train, selected_features, model_configs)
    
    # 6. Phase 2: 选择最佳配置进行完整10折CV
    print(f"\n{'='*60}")
    print("🎯 Phase 2: 最佳配置的完整评估")
    print(f"{'='*60}")
    
    # 比较全特征和特征选择的最佳结果
    best_full = max(best_configs_full.values(), key=lambda x: x['f1_mean'])
    best_selected = max(best_configs_selected.values(), key=lambda x: x['f1_mean'])
    
    print(f"\n📊 3折CV预筛选结果对比:")
    print(f"全特征最佳: F1={best_full['f1_mean']:.4f}")
    print(f"特征选择最佳: F1={best_selected['f1_mean']:.4f}")
    
    # 选择更好的版本进行最终训练
    if best_selected['f1_mean'] > best_full['f1_mean']:
        print("✅ 选择特征选择版本进行最终训练")
        X_final = X_train_selected
        final_configs = best_configs_selected
        final_features = selected_features
    else:
        print("✅ 选择全特征版本进行最终训练")
        X_final = X_train
        final_configs = best_configs_full
        final_features = X_train.columns.tolist()
    
    # 找到最佳模型和配置
    best_model_name = max(final_configs.keys(), key=lambda k: final_configs[k]['f1_mean'])
    best_config = final_configs[best_model_name]['config']
    best_model_class = model_configs[best_model_name]
    
    print(f"\n🏆 最终选择:")
    print(f"   模型: {best_model_name}")
    print(f"   特征数: {X_final.shape[1]}")
    print(f"   配置: {best_config}")
    print(f"   预期F1: {final_configs[best_model_name]['f1_mean']:.4f}")
    
    # 7. 10折交叉验证最终评估
    print(f"\n🔄 10折交叉验证最终评估...")
    
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(len(y_train) / (len(class_counts) * class_counts))
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_final, y_train)):
        print(f"\n--- Fold {fold_idx+1}/10 ---")
        
        X_fold_train, X_fold_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # 数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_fold_train.values),
            torch.LongTensor(y_fold_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_fold_val.values),
            torch.LongTensor(y_fold_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # 训练模型
        model = best_model_class(input_dim=X_final.shape[1]).to(device)
        train_losses, val_f1_scores, learning_rates = train_optimized_model(
            model, train_loader, val_loader, best_config, class_weights, epochs=120
        )
        
        best_f1 = max(val_f1_scores)
        fold_results.append({
            'fold': fold_idx + 1,
            'f1': best_f1,
            'epochs': len(train_losses)
        })
        
        print(f"Fold {fold_idx+1} 最佳F1: {best_f1:.4f}")
    
    # 8. 最终结果
    final_f1_scores = [r['f1'] for r in fold_results]
    mean_f1 = np.mean(final_f1_scores)
    std_f1 = np.std(final_f1_scores)
    
    print(f"\n🎉 最终10折CV结果:")
    print(f"   平均F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"   各折F1: {[f'{f:.4f}' for f in final_f1_scores]}")
    print(f"   平均训练轮数: {np.mean([r['epochs'] for r in fold_results]):.1f}")
    
    # 9. 训练最终模型并预测
    print(f"\n🚀 训练最终模型...")
    
    final_model = best_model_class(input_dim=X_final.shape[1]).to(device)
    
    # 使用全部数据训练
    full_dataset = TensorDataset(
        torch.FloatTensor(X_final.values),
        torch.LongTensor(y_train)
    )
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)
    
    # 训练最优轮数
    optimal_epochs = int(np.mean([r['epochs'] for r in fold_results]))
    
    criterion = LabelSmoothingCrossEntropy(
        num_classes=2, smoothing=best_config['smoothing'], class_weights=class_weights.to(device)
    )
    optimizer = optim.AdamW(
        final_model.parameters(), lr=best_config['lr'], weight_decay=best_config['weight_decay']
    )
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer, warmup_epochs=5, max_epochs=optimal_epochs, eta_min=best_config['lr']*0.01
    )
    
    print(f"训练轮数: {optimal_epochs}")
    final_model.train()
    
    for epoch in range(optimal_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in full_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = final_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step(epoch)
        
        if epoch % 20 == 0:
            print(f'Epoch [{epoch+1}/{optimal_epochs}], Loss: {epoch_loss/len(full_loader):.4f}')
    
    print("✅ 最终模型训练完成!")
    
    # 10. 预测测试集
    print(f"\n🔮 预测测试集...")
    
    # 处理测试集 - 使用相同的特征选择
    df_test = df[df['account'].isin(set(test_df['account']))].copy()
    X_test = df_test[feature_cols].copy()
    X_test = X_test[X_train.columns]  # 保持一致性
    
    # 预处理
    for col in X_test.columns:
        if 'profit' in col:
            X_test[col] = np.sign(X_test[col]) * np.log1p(np.abs(X_test[col]))
            Q01 = X_train[col].quantile(0.01)
            Q99 = X_train[col].quantile(0.99)
            X_test[col] = np.clip(X_test[col], Q01, Q99)
        elif 'ratio' in col:
            X_test[col] = np.clip(X_test[col], 0, 50)
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # 应用相同的特征选择
    if X_final.shape[1] != X_train.shape[1]:  # 如果使用了特征选择
        X_test_final = X_test_scaled.iloc[:, selected_indices]
    else:
        X_test_final = X_test_scaled
    
    # 预测
    final_model.eval()
    X_test_tensor = torch.FloatTensor(X_test_final.values).to(device)
    
    with torch.no_grad():
        outputs = final_model(X_test_tensor)
        _, predictions = torch.max(outputs, 1)
    
    predictions = predictions.cpu().numpy()
    
    # 保存结果
    result_dir = "/Users/mannormal/4011/Qi Zihan/result_analysis/prediction_results/"
    os.makedirs(result_dir, exist_ok=True)
    
    submission_df = pd.DataFrame({
        'account': df_test['account'].values,
        'Predict': predictions
    })
    
    filename = f"MLP_OPTIMIZED_{best_model_name}_f1_{mean_f1:.4f}_features_{X_final.shape[1]}.csv"
    submission_df.to_csv(os.path.join(result_dir, filename), index=False)
    
    print(f"\n🎉 优化版MLP完成!")
    print(f"📊 最佳模型: {best_model_name}")
    print(f"🔧 特征数: {X_final.shape[1]}")
    print(f"📈 10折CV F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"📁 文件: {filename}")
    print(f"📊 预测分布: {np.bincount(predictions)}")
    
    return {
        'model_name': best_model_name,
        'config': best_config,
        'cv_f1': mean_f1,
        'cv_std': std_f1,
        'feature_count': X_final.shape[1],
        'selected_features': final_features,
        'submission': submission_df
    }

if __name__ == "__main__":
    results = main_optimized()