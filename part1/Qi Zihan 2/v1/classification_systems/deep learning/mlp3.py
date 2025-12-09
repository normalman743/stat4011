import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
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

# ========== 回归基础的MLP架构 ==========
class RefinedMLP(nn.Module):
    """精简版：回归原始架构，但适配新的特征数量并加入有效改进"""
    
    def __init__(self, input_dim=53, dropout_rates=[0.3, 0.2, 0.2, 0.1, 0.1]):  # 更新默认输入维度
        super(RefinedMLP, self).__init__()
        
        self.network = nn.Sequential(
            # input_dim -> 128
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0]),
            
            # 128 -> 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rates[1]),
            
            # 64 -> 64
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rates[2]),
            
            # 64 -> 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rates[3]),
            
            # 32 -> 16
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rates[4]),
            
            # 16 -> 2
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

# ========== 轻量级Label Smoothing ==========
class LightLabelSmoothingCE(nn.Module):
    """轻量级Label Smoothing - 只保留核心功能"""
    def __init__(self, smoothing=0.05, class_weights=None):
        super(LightLabelSmoothingCE, self).__init__()
        self.smoothing = smoothing
        self.class_weights = class_weights
        
    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=1)
        
        # 简化的Label smoothing
        with torch.no_grad():
            smooth_target = torch.zeros_like(pred)
            smooth_target.fill_(self.smoothing / (pred.size(1) - 1))  # 平滑分布
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)  # 真实标签
        
        loss = -torch.sum(smooth_target * log_prob, dim=1)
        
        # 应用类别权重
        if self.class_weights is not None:
            weights = self.class_weights[target]
            loss = loss * weights
        
        return loss.mean()

# ========== 简化的早停法 ==========
class SimpleEarlyStopping:
    """简化的早停法 - 专注核心功能"""
    def __init__(self, patience=12, min_delta=0.001):
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
                # 恢复最佳权重
                if self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}

# ========== 精简的训练函数 ==========
def train_refined_model(model, train_loader, val_loader, config, class_weights, epochs=100):
    """精简的训练函数 - 只保留有效的组件"""
    
    print(f"🎯 精简训练配置:")
    print(f"   Weight Decay: {config['weight_decay']}")
    print(f"   Learning Rate: {config['lr']}")
    print(f"   Label Smoothing: {config['smoothing']}")
    
    # 损失函数和优化器
    criterion = LightLabelSmoothingCE(
        smoothing=config['smoothing'], 
        class_weights=class_weights.to(device)
    )
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    # 简单的学习率调度 - 验证停滞时减半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=6, min_lr=config['lr']*0.01
    )
    
    # 早停法
    early_stopping = SimpleEarlyStopping(patience=12, min_delta=0.001)
    
    train_losses = []
    val_f1_scores = []
    
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
            
            # 轻微的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        # 计算指标
        val_f1 = f1_score(val_targets, val_predictions, average='weighted')
        avg_train_loss = train_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_f1_scores.append(val_f1)
        
        # 学习率调度
        scheduler.step(val_f1)
        
        # 早停检查
        if early_stopping(val_f1, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val F1: {val_f1:.4f}, '
                  f'LR: {current_lr:.2e}')
    
    return train_losses, val_f1_scores

# ========== 聚焦的配置测试 ==========
def focused_config_search(X, y):
    """聚焦的配置搜索 - 基于你的发现"""
    
    print("\n🎯 聚焦配置搜索 - 基于发现的有效模式")
    print("=" * 60)
    
    # 基于你的分析，重点测试高正则化配置
    focused_configs = [
        # 配置1: 你发现的最佳配置
        {'weight_decay': 0.001, 'lr': 0.0008, 'smoothing': 0.05, 'name': '最佳发现'},
        
        # 配置2: 稍微降低正则化
        {'weight_decay': 0.0008, 'lr': 0.001, 'smoothing': 0.05, 'name': '平衡版本'},
        
        # 配置3: 原始基础 + 轻微改进
        {'weight_decay': 0.0005, 'lr': 0.001, 'smoothing': 0.03, 'name': '保守改进'},
        
        # 配置4: 最小改动
        {'weight_decay': 0.0001, 'lr': 0.001, 'smoothing': 0.02, 'name': '最小改动'},
    ]
    
    # 计算类别权重
    class_counts = np.bincount(y)
    class_weights = torch.FloatTensor(len(y) / (len(class_counts) * class_counts))
    print(f"类别权重: {class_weights}")
    
    results = []
    
    # 5折快速验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for config_idx, config in enumerate(focused_configs):
        print(f"\n📊 测试配置 {config_idx+1}: {config['name']}")
        print(f"   {config}")
        
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 数据准备
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train.values),
                torch.LongTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val.values),
                torch.LongTensor(y_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            
            # 模型训练
            model = RefinedMLP(input_dim=X.shape[1]).to(device)
            train_losses, val_f1_scores = train_refined_model(
                model, train_loader, val_loader, config, class_weights, epochs=80
            )
            
            fold_f1 = max(val_f1_scores) if val_f1_scores else 0
            fold_scores.append(fold_f1)
            
            print(f"    Fold {fold_idx+1}: F1 = {fold_f1:.4f}")
        
        # 统计结果
        mean_f1 = np.mean(fold_scores)
        std_f1 = np.std(fold_scores)
        
        results.append({
            'config': config,
            'f1_mean': mean_f1,
            'f1_std': std_f1,
            'fold_scores': fold_scores
        })
        
        print(f"✅ {config['name']}: F1 = {mean_f1:.4f} ± {std_f1:.4f}")
    
    # 选择最佳配置
    best_result = max(results, key=lambda x: x['f1_mean'])
    
    print(f"\n🏆 最佳配置: {best_result['config']['name']}")
    print(f"   F1分数: {best_result['f1_mean']:.4f} ± {best_result['f1_std']:.4f}")
    print(f"   详细配置: {best_result['config']}")
    
    return best_result, results

# ========== 最终完整验证 ==========
def final_validation(X, y, best_config, cv_folds=10):
    """最终10折交叉验证"""
    
    print(f"\n🔄 最终{cv_folds}折交叉验证")
    print("=" * 60)
    print(f"使用配置: {best_config['name']}")
    
    # 计算类别权重
    class_counts = np.bincount(y)
    class_weights = torch.FloatTensor(len(y) / (len(class_counts) * class_counts))
    
    # 10折交叉验证
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_idx+1}/{cv_folds} ---")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 数据准备
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train.values),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val.values),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # 模型训练
        model = RefinedMLP(input_dim=X.shape[1]).to(device)
        train_losses, val_f1_scores = train_refined_model(
            model, train_loader, val_loader, best_config, class_weights, epochs=100
        )
        
        best_f1 = max(val_f1_scores)
        fold_results.append({
            'fold': fold_idx + 1,
            'f1': best_f1,
            'epochs': len(train_losses)
        })
        
        print(f"Fold {fold_idx+1} 最佳F1: {best_f1:.4f}")
    
    # 统计最终结果
    final_f1_scores = [r['f1'] for r in fold_results]
    mean_f1 = np.mean(final_f1_scores)
    std_f1 = np.std(final_f1_scores)
    avg_epochs = np.mean([r['epochs'] for r in fold_results])
    
    print(f"\n🎉 最终结果:")
    print(f"   平均F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"   各折F1: {[f'{f:.4f}' for f in final_f1_scores]}")
    print(f"   平均训练轮数: {avg_epochs:.1f}")
    
    return mean_f1, std_f1, fold_results

# ========== 主函数 ==========
def main_refined():
    """精简主函数 - 回归基础但有针对性改进"""
    
    print("="*80)
    print("🎯 MLP精简优化 - 回归基础 + 有效改进")
    print("="*80)
    
    # 1. 数据加载和预处理
    print("\n📂 数据加载...")
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
    
    print(f"训练数据: {df_train.shape}")
    print(f"标签分布: {np.bincount(df_train['label'])}")
    
    # 特征预处理 - 排除时间字符串特征，保留数值特征
    # 排除account和时间字符串列
    time_cols = ['first_transaction_time', 'last_transaction_time'] 
    feature_cols = [col for col in df.columns if col not in ['account'] + time_cols]
    
    print(f"📊 原始特征数: {len(df.columns)-1}")
    print(f"📊 排除时间字符串后: {len(feature_cols)} 个数值特征")
    print(f"📊 排除的时间列: {time_cols}")
    
    # 使用简化的预处理（避免导入mlp模块的依赖问题）
    X_train = df_train[feature_cols].copy()
    
    # 基础预处理
    for col in X_train.columns:
        if 'profit' in col.lower():
            # 对profit类特征进行对数变换和截断
            X_train[col] = np.sign(X_train[col]) * np.log1p(np.abs(X_train[col]))
            Q01, Q99 = X_train[col].quantile([0.01, 0.99])
            X_train[col] = np.clip(X_train[col], Q01, Q99)
        elif 'ratio' in col.lower():
            # 对ratio类特征进行截断
            X_train[col] = np.clip(X_train[col], 0, 50)
    
    # 标准化
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    y_train = df_train['label'].values
    
    print(f"🔧 使用 {X_train.shape[1]} 个数值特征 (排除时间字符串)")
    
    # 2. 聚焦配置搜索
    best_result, all_results = focused_config_search(X_train, y_train)
    
    # 3. 与原始基线对比
    print(f"\n📊 与原始性能对比:")
    print(f"   原始MLP基线: ~0.8900")
    print(f"   当前最佳配置: {best_result['f1_mean']:.4f}")
    
    if best_result['f1_mean'] >= 0.888:  # 设定一个合理的阈值
        print("✅ 配置有效，进行最终验证")
        
        # 4. 最终10折验证
        final_mean_f1, final_std_f1, fold_results = final_validation(
            X_train, y_train, best_result['config'], cv_folds=10
        )
        
        # 5. 训练最终模型
        print(f"\n🚀 训练最终模型...")
        
        # 使用全部数据训练
        full_dataset = TensorDataset(
            torch.FloatTensor(X_train.values),
            torch.LongTensor(y_train)
        )
        full_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)
        
        # 训练最优轮数
        optimal_epochs = int(np.mean([r['epochs'] for r in fold_results]))
        
        # 计算类别权重
        class_counts = np.bincount(y_train)
        class_weights = torch.FloatTensor(len(y_train) / (len(class_counts) * class_counts))
        
        # 最终模型
        final_model = RefinedMLP(input_dim=X_train.shape[1]).to(device)
        
        criterion = LightLabelSmoothingCE(
            smoothing=best_result['config']['smoothing'], 
            class_weights=class_weights.to(device)
        )
        optimizer = optim.AdamW(
            final_model.parameters(), 
            lr=best_result['config']['lr'], 
            weight_decay=best_result['config']['weight_decay']
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
            
            if epoch % 20 == 0:
                print(f'Epoch [{epoch+1}/{optimal_epochs}], Loss: {epoch_loss/len(full_loader):.4f}')
        
        print("✅ 最终模型训练完成!")
        
        # 6. 预测测试集
        print(f"\n🔮 预测测试集...")
        
        # 处理测试集
        df_test = df[df['account'].isin(set(test_df['account']))].copy()
        X_test = df_test[feature_cols].copy()  # 使用相同的数值特征列
        
        # 预处理 - 与训练集保持一致
        for col in X_test.columns:
            if 'profit' in col.lower():
                X_test[col] = np.sign(X_test[col]) * np.log1p(np.abs(X_test[col]))
                Q01 = X_train[col].quantile(0.01)
                Q99 = X_train[col].quantile(0.99)
                X_test[col] = np.clip(X_test[col], Q01, Q99)
            elif 'ratio' in col.lower():
                X_test[col] = np.clip(X_test[col], 0, 50)
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # 预测
        final_model.eval()
        X_test_tensor = torch.FloatTensor(X_test_scaled.values).to(device)
        
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
        
        config_name = best_result['config']['name'].replace(' ', '_')
        filename = f"MLP_REFINED_{config_name}_f1_{final_mean_f1:.4f}.csv"
        submission_df.to_csv(os.path.join(result_dir, filename), index=False)
        
        print(f"\n🎉 精简版MLP完成!")
        print(f"📊 最佳配置: {best_result['config']['name']}")
        print(f"📈 最终F1: {final_mean_f1:.4f} ± {final_std_f1:.4f}")
        print(f"📁 文件: {filename}")
        print(f"📊 预测分布: {np.bincount(predictions)}")
        
        return {
            'config': best_result['config'],
            'cv_f1': final_mean_f1,
            'cv_std': final_std_f1,
            'submission': submission_df,
            'all_configs': all_results
        }
        
    else:
        print("❌ 配置效果不佳，建议保持原始版本")
        return None

if __name__ == "__main__":
    results = main_refined()