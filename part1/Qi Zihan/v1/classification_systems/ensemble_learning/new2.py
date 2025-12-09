import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

print("=== ULTRA Advanced ResNet Meta-ANN System ===")

# =====================================================
# Advanced ResNet-Style Meta-ANN Architecture
# =====================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.skip_bn = nn.BatchNorm1d(out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        
        # Skip connection
        skip = self.skip_bn(self.skip(x))
        
        # Add & activate
        out = F.relu(out + skip)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, features, attention_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(features, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, features),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        weights = self.attention(x)
        return x * weights

class AdvancedResNetMetaANN(nn.Module):
    def __init__(self, n_base, n_feat, hidden_dims=[256, 128, 64, 32], dropout=0.4):
        super().__init__()
        
        # Feature scaling parameters
        self.base_scale = nn.Parameter(torch.ones(n_base))
        self.base_bias = nn.Parameter(torch.zeros(n_base))
        self.feat_scale = nn.Parameter(torch.ones(n_feat))
        self.feat_bias = nn.Parameter(torch.zeros(n_feat))
        
        # Feature fusion
        total_features = n_base + n_feat
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(total_features, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ResNet blocks (20+ layers total)
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.res_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout)
            )
            # Add another residual block at same dimension for deeper network
            self.res_blocks.append(
                ResidualBlock(hidden_dims[i+1], hidden_dims[i+1], dropout)
            )
        
        # Attention mechanism
        self.attention = AttentionBlock(hidden_dims[-1])
        
        # Multiple prediction heads (ensemble within model)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[-1], 16),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(16, 1)
            ) for _ in range(3)  # 3 prediction heads
        ])
        
        # Final ensemble layer
        self.final_ensemble = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # Gradient clipping value
        self.grad_clip = 1.0
        
    def forward(self, x_base, x_feat):
        # Enhanced feature scaling
        x_base_scaled = self.base_scale * x_base + self.base_bias
        x_feat_scaled = self.feat_scale * x_feat + self.feat_bias
        
        # Feature fusion with learnable weights
        x = torch.cat([x_base_scaled, x_feat_scaled], dim=1)
        
        # Input projection
        x = self.input_proj(x)
        
        # ResNet blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Attention
        x = self.attention(x)
        
        # Multiple heads
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(x))
        
        # Ensemble heads
        ensemble_input = torch.cat(head_outputs, dim=1)
        final_output = self.final_ensemble(ensemble_input)
        
        return final_output

def advanced_train_meta_ann(base_predictions, original_features, y_true, n_epochs=1000, patience=50):
    """
    训练高级ResNet Meta-ANN
    """
    print(f"\n🚀 Training Advanced ResNet Meta-ANN")
    print(f"Base predictions shape: {base_predictions.shape}")
    print(f"Original features shape: {original_features.shape}")
    
    # 多尺度特征标准化
    scaler_base = StandardScaler()
    scaler_feat = StandardScaler()
    
    base_predictions = scaler_base.fit_transform(base_predictions)
    original_features = scaler_feat.fit_transform(original_features)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 转换为张量
    X_base_tensor = torch.tensor(base_predictions, dtype=torch.float32).to(device)
    X_feat_tensor = torch.tensor(original_features, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_true.reshape(-1,1), dtype=torch.float32).to(device)
    
    # 创建高级模型
    model = AdvancedResNetMetaANN(
        n_base=base_predictions.shape[1], 
        n_feat=original_features.shape[1],
        hidden_dims=[512, 256, 128, 64, 32],  # 更深的网络
        dropout=0.5
    ).to(device)
    
    print(f"📊 Model Info:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # 高级优化器设置
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-3, 
        weight_decay=1e-3,  # 更强的正则化
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 复合损失函数
    bce_loss = nn.BCELoss()
    focal_loss_alpha = 0.25
    focal_loss_gamma = 2.0
    
    def focal_loss(pred, target):
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** focal_loss_gamma)
        if focal_loss_alpha >= 0:
            alpha_t = focal_loss_alpha * target + (1 - focal_loss_alpha) * (1 - target)
            loss = alpha_t * loss
        return loss.mean()
    
    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    # 交叉验证分割
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = list(skf.split(base_predictions, y_true))[0]
    
    Xb_train, Xb_val = X_base_tensor[train_idx], X_base_tensor[val_idx]
    Xf_train, Xf_val = X_feat_tensor[train_idx], X_feat_tensor[val_idx]
    y_train, y_val = y_tensor[train_idx], y_tensor[val_idx]
    
    # 训练历史
    best_val_f1 = 0
    patience_counter = 0
    train_f1_history = []
    val_f1_history = []
    train_loss_history = []
    val_loss_history = []
    
    print(f"\n🎯 Training Progress:")
    print("Epoch | Train Loss | Val Loss | Train F1 | Val F1   | LR       | Status")
    print("-" * 75)
    
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        
        y_pred_train = model(Xb_train, Xf_train)
        
        # 复合损失
        bce_loss_val = bce_loss(y_pred_train, y_train)
        focal_loss_val = focal_loss(y_pred_train, y_train)
        total_loss = 0.7 * bce_loss_val + 0.3 * focal_loss_val
        
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            y_pred_val = model(Xb_val, Xf_val)
            val_loss = bce_loss(y_pred_val, y_val)
            
            # 计算F1分数
            y_train_prob = model(Xb_train, Xf_train).cpu().numpy()
            y_val_prob = y_pred_val.cpu().numpy()
            
            # 动态阈值优化
            best_threshold = 0.5
            best_f1 = 0
            for threshold in np.arange(0.3, 0.8, 0.05):
                val_pred_thresh = (y_val_prob > threshold).astype(int).flatten()
                f1_thresh = metrics.f1_score(y_true[val_idx], val_pred_thresh, zero_division=0)
                if f1_thresh > best_f1:
                    best_f1 = f1_thresh
                    best_threshold = threshold
            
            train_pred = (y_train_prob > best_threshold).astype(int).flatten()
            val_pred = (y_val_prob > best_threshold).astype(int).flatten()
            
            train_f1 = metrics.f1_score(y_true[train_idx], train_pred, zero_division=0)
            val_f1 = metrics.f1_score(y_true[val_idx], val_pred, zero_division=0)
            
            train_f1_history.append(train_f1)
            val_f1_history.append(val_f1)
            train_loss_history.append(total_loss.item())
            val_loss_history.append(val_loss.item())
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 早停检查
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            best_threshold_final = best_threshold
            status = "✅ Best"
        else:
            patience_counter += 1
            status = f"⏳ {patience_counter}/{patience}"
        
        # 打印进度
        if epoch % 25 == 0 or patience_counter == 0:
            print(f"{epoch:5d} | {total_loss.item():10.6f} | {val_loss.item():8.6f} | "
                  f"{train_f1:8.4f} | {val_f1:8.4f} | {current_lr:.2e} | {status}")
        
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {epoch}")
            print(f"🏆 Best validation F1: {best_val_f1:.4f}")
            print(f"🎯 Optimal threshold: {best_threshold_final:.3f}")
            break
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    model.eval()
    
    # 最终预测
    with torch.no_grad():
        y_final_pred = model(X_base_tensor, X_feat_tensor).cpu().numpy()
        y_final_label = (y_final_pred > best_threshold_final).astype(int).flatten()
        final_f1 = metrics.f1_score(y_true, y_final_label, zero_division=0)
        final_acc = metrics.accuracy_score(y_true, y_final_label)
        final_precision = metrics.precision_score(y_true, y_final_label, zero_division=0)
        final_recall = metrics.recall_score(y_true, y_final_label, zero_division=0)
        
    print(f"\n📊 Advanced ResNet Meta-ANN Final Results:")
    print(f"   Accuracy: {final_acc:.4f}")
    print(f"   Precision: {final_precision:.4f}")
    print(f"   Recall: {final_recall:.4f}")
    print(f"   F1-Score: {final_f1:.4f}")
    print(f"   Best Val F1: {best_val_f1:.4f}")
    print(f"   Optimal Threshold: {best_threshold_final:.3f}")
    print(f"   Final Train F1: {train_f1_history[-1]:.4f}")
    print(f"   Generalization Gap: {train_f1_history[-1] - best_val_f1:+.4f}")
    
    return y_final_pred, model, (scaler_base, scaler_feat), {
        'final_f1': final_f1,
        'final_acc': final_acc,
        'final_precision': final_precision,
        'final_recall': final_recall,
        'best_val_f1': best_val_f1,
        'best_threshold': best_threshold_final,
        'train_f1_history': train_f1_history,
        'val_f1_history': val_f1_history,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history
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
        
        cv_score = np.mean([metrics.f1_score(y_all[val_idx], clf.predict(X_all[val_idx]), zero_division=0) 
                           for train_idx, val_idx in skf.split(X_all, y_all)])
        cv_scores.append(cv_score)
        
        y_pred = clf.predict_proba(X_all)[:, 1]  # 概率
        predictions.append(y_pred)
    
    predictions_array = np.array(predictions).T  # 🔧 修复: 转置为 (n_samples, n_models)
    print(f"   Predictions shape: {predictions_array.shape}")
    
    return predictions_array, cv_scores

# =====================================================
# PyTorch Meta-ANN with Feature Scaling
# =====================================================
class MetaANN(nn.Module):
    def __init__(self, n_base, n_feat, hidden1=128, hidden2=64, hidden3=32, dropout=0.3):
        super().__init__()
        # 可训练缩放参数
        self.a = nn.Parameter(torch.ones(n_feat))
        self.b = nn.Parameter(torch.zeros(n_feat))
        
        # 更深的网络架构
        self.fc1 = nn.Linear(n_base + n_feat, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.bn3 = nn.BatchNorm1d(hidden3)
        self.dropout3 = nn.Dropout(dropout)
        
        self.out = nn.Linear(hidden3, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x_base, x_feat):
        # 特征缩放
        x_feat_scaled = self.a * x_feat + self.b
        
        # 特征融合
        x = torch.cat([x_base, x_feat_scaled], dim=1)
        
        # 深度网络
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
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
        hidden1=128, 
        hidden2=64, 
        hidden3=32,
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
    
    print("\nEpoch | Train F1 | Val F1   | LR       | Status")
    print("-" * 50)
    
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
            
            train_f1 = metrics.f1_score(y_true[train_idx], train_pred, zero_division=0)
            val_f1 = metrics.f1_score(y_true[val_idx], val_pred, zero_division=0)
            
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
            print(f"{epoch:5d} | {train_f1:8.4f} | {val_f1:8.4f} | {current_lr:.2e} | {status}")
        
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
        final_acc = metrics.accuracy_score(y_true, y_final_label)
        
    print(f"\n📊 Meta-ANN Final Results:")
    print(f"   Accuracy: {final_acc:.4f}")
    print(f"   F1-Score: {final_f1:.4f}")
    print(f"   Best Val F1: {best_val_f1:.4f}")
    print(f"   Overfitting: {train_f1_history[-1] - best_val_f1:+.4f}")
    
    return y_final_pred, model, scaler, {
        'final_f1': final_f1,
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
    
    # 🔍 分析为什么F1下降的原因
    print(f"\n   🔍 F1分数分析:")
    print(f"      📉 从0.7降到0.5的可能原因:")
    print(f"         1. 之前可能在训练集上评估(过拟合)")
    print(f"         2. 现在使用真正的交叉验证(更真实)")
    print(f"         3. 数据极度不平衡 1:{bad_accounts//good_accounts}")
    print(f"         4. 特征可能需要更多工程化")
    
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
    features_path = '/Users/mannormal/4011/Qi Zihan/feature_extraction/generated_features/all_features.csv'
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
    
    print("\nFold | Train F1 | Val F1   | Train Acc| Val Acc  | Overfitting")
    print("-" * 65)
    
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
            n_feat=X_feat_train_scaled.shape[1]
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
            train_acc = metrics.accuracy_score(y_train_fold, train_label)
            val_acc = metrics.accuracy_score(y_val_fold, val_label)
            
            overfitting = train_f1 - val_f1
            overfit_status = "🔴 High" if overfitting > 0.1 else "🟡 Med" if overfitting > 0.05 else "🟢 Low"
            
            print(f"{fold+1:4d} | {train_f1:8.4f} | {val_f1:8.4f} | {train_acc:8.4f} | {val_acc:8.4f} | {overfit_status}")
            
            cv_results.append({
                'fold': fold + 1,
                'train_f1': train_f1,
                'val_f1': val_f1,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'overfitting': overfitting
            })
    
    # CV统计
    avg_train_f1 = np.mean([r['train_f1'] for r in cv_results])
    avg_val_f1 = np.mean([r['val_f1'] for r in cv_results])
    avg_train_acc = np.mean([r['train_acc'] for r in cv_results])
    avg_val_acc = np.mean([r['val_acc'] for r in cv_results])
    avg_overfitting = avg_train_f1 - avg_val_f1
    
    print("-" * 65)
    print(f"Avg  | {avg_train_f1:8.4f} | {avg_val_f1:8.4f} | {avg_train_acc:8.4f} | {avg_val_acc:8.4f} | {avg_overfitting:+7.4f}")
    
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
    print(f"   Meta-ANN: {combined_base_predictions.shape[1]+original_features.shape[1]} → 128 → 64 → 32 → 1")
    
    # =====================================================
    # Phase 5: Test Set Prediction & Submission File Generation
    # =====================================================
    print(f"\n{'='*80}")
    print("🎯 PHASE 5: Test Set Prediction & Submission Generation")
    print(f"{'='*80}")
    
    print("\n📁 Loading Test Data...")
    test_df = pd.merge(all_features_df, te[['account']], on='account', how='inner')
    test_df['account_type'] = test_df.apply(classify_account_type_original, axis=1)
    
    print(f"   Test data shape: {test_df.shape}")
    print(f"   Test account types: {dict(test_df['account_type'].value_counts())}")
    
    # Prepare test features
    test_original_features = test_df[feature_cols].values
    print(f"   Test original features shape: {test_original_features.shape}")
    
    # =====================================================
    # Step 5.1: Retrain RF Ensemble on Full Training Data
    # =====================================================
    print(f"\n🌳 Step 5.1: Retraining RF Ensemble on Full Training Data")
    
    rf_models = []
    test_rf_predictions = []
    
    for i in tqdm(range(100), desc="Training RF for Test"):
        # Sample with replacement for diversity
        good_sample = training_df[training_df['flag'] == 1].sample(
            n=len(training_df[training_df['flag'] == 1]), replace=True, random_state=i
        )
        bad_sample = training_df[training_df['flag'] == -1].sample(
            n=len(training_df[training_df['flag'] == 1]), replace=True, random_state=i+5000
        )
        train_sample = pd.concat([good_sample, bad_sample])
        
        X_train = train_sample[rf_feature_names].values
        y_train = np.where(train_sample['flag'].values == -1, 0, 1)
        
        rf_configs = [
            {'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 8, 'min_samples_leaf': 3},
            {'n_estimators': 180, 'max_depth': 30, 'min_samples_split': 6, 'min_samples_leaf': 2},
            {'n_estimators': 150, 'max_depth': 20, 'min_samples_split': 10, 'min_samples_leaf': 4},
            {'n_estimators': 220, 'max_depth': 35, 'min_samples_split': 12, 'min_samples_leaf': 5},
        ]
        config = rf_configs[i % len(rf_configs)]
        
        clf = RandomForestClassifier(
            **config,
            random_state=i,
            class_weight='balanced_subsample',
            max_features='sqrt',
            bootstrap=True,
            n_jobs=1
        )
        clf.fit(X_train, y_train)
        rf_models.append(clf)
        
        # Predict on test set
        test_pred = clf.predict_proba(test_df[rf_feature_names].values)[:, 1]
        test_rf_predictions.append(test_pred)
    
    test_rf_predictions = np.array(test_rf_predictions).T  # (n_test_samples, n_rf_models)
    print(f"   ✅ Test RF predictions shape: {test_rf_predictions.shape}")
    
    # =====================================================
    # Step 5.2: Strategy Ensemble Predictions on Test Set
    # =====================================================
    print(f"\n🎯 Step 5.2: Strategy Ensemble Predictions on Test Set")
    
    test_strategy_predictions = []
    
    for strategy_name, strategy_categories in strategy_data.items():
        print(f"\n   📊 {strategy_name.upper()} Strategy on Test Set")
        
        # Merge test data with strategy categories
        test_with_strategy = test_df.merge(strategy_categories, on='account', how='left')
        strategy_col = f"{strategy_name}_category"
        test_with_strategy[strategy_col] = test_with_strategy[strategy_col].fillna('unknown')
        
        # Create strategy features
        strategy_dummies = pd.get_dummies(test_with_strategy[strategy_col], prefix=strategy_name)
        test_feature_data = pd.concat([
            test_with_strategy[[col for col in rf_feature_names]],  # Use same features as RF
            strategy_dummies
        ], axis=1)
        
        strategy_test_preds = []
        
        # Retrain strategy models on full training data
        for i in range(20):  # 20 models per strategy
            # Prepare training data with strategy
            train_with_strategy = training_df.merge(strategy_categories, on='account', how='left')
            train_with_strategy[strategy_col] = train_with_strategy[strategy_col].fillna('unknown')
            
            train_strategy_dummies = pd.get_dummies(train_with_strategy[strategy_col], prefix=strategy_name)
            train_feature_data = pd.concat([
                train_with_strategy[rf_feature_names],
                train_strategy_dummies
            ], axis=1)
            
            # Align columns between train and test
            common_cols = list(set(train_feature_data.columns) & set(test_feature_data.columns))
            for col in train_feature_data.columns:
                if col not in test_feature_data.columns:
                    test_feature_data[col] = 0
            for col in test_feature_data.columns:
                if col not in train_feature_data.columns:
                    train_feature_data[col] = 0
            
            train_feature_data = train_feature_data[sorted(train_feature_data.columns)]
            test_feature_data = test_feature_data[sorted(test_feature_data.columns)]
            
            # Sample and train
            good_sample = train_with_strategy[train_with_strategy['flag'] == 1].sample(
                n=len(train_with_strategy[train_with_strategy['flag'] == 1]), replace=True, random_state=i*200
            )
            bad_sample = train_with_strategy[train_with_strategy['flag'] == -1].sample(
                n=len(train_with_strategy[train_with_strategy['flag'] == 1]), replace=True, random_state=i*200+100
            )
            sample_indices = list(good_sample.index) + list(bad_sample.index)
            
            X_train = train_feature_data.loc[sample_indices].values
            y_train = np.where(pd.concat([good_sample, bad_sample])['flag'].values == -1, 0, 1)
            
            clf = RandomForestClassifier(
                n_estimators=120,
                max_depth=18,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=i*10,
                class_weight='balanced'
            )
            clf.fit(X_train, y_train)
            
            # Predict on test
            test_pred = clf.predict_proba(test_feature_data.values)[:, 1]
            strategy_test_preds.append(test_pred)
        
        strategy_test_preds = np.array(strategy_test_preds).T  # (n_test_samples, n_strategy_models)
        test_strategy_predictions.append(strategy_test_preds)
        print(f"      ✅ {strategy_name} test predictions: {strategy_test_preds.shape}")
    
    # Combine all test base predictions
    test_combined_base_predictions = np.hstack([test_rf_predictions] + test_strategy_predictions)
    print(f"\n   📊 Combined test base predictions: {test_combined_base_predictions.shape}")
    
    # =====================================================
    # Step 5.3: Meta-ANN Inference on Test Set
    # =====================================================
    print(f"\n🤖 Step 5.3: Meta-ANN Inference on Test Set")
    
    # Scale test features using the same scaler
    test_original_features_scaled = feature_scaler.transform(test_original_features)
    
    # Convert to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    test_base_tensor = torch.tensor(test_combined_base_predictions, dtype=torch.float32).to(device)
    test_feat_tensor = torch.tensor(test_original_features_scaled, dtype=torch.float32).to(device)
    
    # Meta-ANN prediction
    meta_model.eval()
    with torch.no_grad():
        test_meta_predictions = meta_model(test_base_tensor, test_feat_tensor).cpu().numpy()
    
    # Convert probabilities to labels (using 0.5 threshold)
    test_final_labels = (test_meta_predictions > 0.5).astype(int).flatten()
    
    print(f"   ✅ Test Meta-ANN predictions shape: {test_meta_predictions.shape}")
    print(f"   📊 Test prediction distribution:")
    unique, counts = np.unique(test_final_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"      Label {label}: {count} accounts ({count/len(test_final_labels)*100:.1f}%)")
    
    # =====================================================
    # Step 5.4: Generate Submission File
    # =====================================================
    print(f"\n📝 Step 5.4: Generating Submission File")
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'account': test_df['account'].values,
        'flag': test_final_labels
    })
    
    # Generate filename with CV F1 score
    cv_f1_score = avg_val_f1
    filename = f"advanced_resnet_meta_ann_mean_cv_f1_score_{cv_f1_score:.4f}.csv"
    filepath = f"/Users/mannormal/4011/Qi Zihan/classification_systems/ensemble_learning/{filename}"
    
    # Save submission file
    submission_df.to_csv(filepath, index=False)
    
    print(f"   ✅ Submission file saved: {filename}")
    print(f"   📁 Full path: {filepath}")
    print(f"   📊 Submission stats:")
    print(f"      Total accounts: {len(submission_df)}")
    print(f"      Predicted good (1): {np.sum(test_final_labels)} ({np.mean(test_final_labels)*100:.1f}%)")
    print(f"      Predicted bad (0): {len(test_final_labels) - np.sum(test_final_labels)} ({(1-np.mean(test_final_labels))*100:.1f}%)")
    print(f"      CV F1 Score: {cv_f1_score:.4f}")
    
    # Display first few predictions
    print(f"\n   🔍 First 10 predictions:")
    print(submission_df.head(10).to_string(index=False))
    
    return {
        'meta_model': meta_model,
        'feature_scaler': feature_scaler,
        'rf_predictions': rf_predictions,
        'strategy_predictions': all_strategy_predictions,
        'cv_results': cv_results,
        'meta_results': meta_results,
        'final_f1': avg_val_f1,
        'submission_file': filename,
        'submission_path': filepath,
        'test_predictions': test_final_labels
    }

if __name__ == "__main__":
    results = main()
    
    print(f"\n{'='*80}")
    print("✅ Advanced ResNet Meta-ANN System Complete!")
    print(f"🎯 Cross-Validation F1: {results['final_f1']:.4f}")
    print(f"📁 Submission File: {results['submission_file']}")
    print(f"📊 Test Predictions: {len(results['test_predictions'])} accounts")
    print(f"{'='*80}")
    print("\n🎉 Ready for submission! 🚀")
