import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from datetime import datetime

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class PositionalEncoding(nn.Module):
    """位置编码 - 为每个特征位置添加编码"""
    
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]

class FeatureEmbedding(nn.Module):
    """特征嵌入层 - 将每个特征映射到d_model维度"""
    
    def __init__(self, input_dim, d_model, dropout=0.1):
        super(FeatureEmbedding, self).__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        
        # 每个特征一个线性变换
        self.feature_projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(input_dim)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        batch_size = x.size(0)
        
        # 将每个特征投影到d_model维度
        feature_embeddings = []
        for i in range(self.input_dim):
            feature_val = x[:, i:i+1]  # [batch_size, 1]
            embedded = self.feature_projections[i](feature_val)  # [batch_size, d_model]
            feature_embeddings.append(embedded)
        
        # Stack to create sequence: [input_dim, batch_size, d_model]
        embeddings = torch.stack(feature_embeddings, dim=0)
        
        # Apply layer norm and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class TransformerBlock(nn.Module):
    """标准Transformer编码器块"""
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=nhead, 
            dropout=dropout,
            batch_first=False  # 使用 [seq_len, batch, embed_dim] 格式
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class FinancialTransformer(nn.Module):
    """金融账户分类Transformer模型"""
    
    def __init__(self, input_dim=30, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, num_classes=2):
        super(FinancialTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 特征嵌入
        self.feature_embedding = FeatureEmbedding(input_dim, d_model, dropout)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=input_dim)
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # 全局池化策略
        self.pooling_strategy = "mean"  # 可选: "mean", "max", "cls"
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        
        # 特征嵌入: [input_dim, batch_size, d_model]
        embedded = self.feature_embedding(x)
        
        # 位置编码
        embedded = self.pos_encoding(embedded)
        
        # 通过Transformer层
        for transformer_layer in self.transformer_layers:
            embedded = transformer_layer(embedded)
        
        # 全局池化: [input_dim, batch_size, d_model] -> [batch_size, d_model]
        if self.pooling_strategy == "mean":
            pooled = embedded.mean(dim=0)  # 对序列维度求平均
        elif self.pooling_strategy == "max":
            pooled, _ = embedded.max(dim=0)  # 对序列维度求最大值
        else:  # "cls" - 使用第一个位置
            pooled = embedded[0]
        
        # 分类
        output = self.classifier(pooled)
        
        return output
    
    def get_attention_weights(self, x):
        """获取注意力权重用于可解释性分析"""
        
        embedded = self.feature_embedding(x)
        embedded = self.pos_encoding(embedded)
        
        attention_weights = []
        
        for transformer_layer in self.transformer_layers:
            # 手动计算注意力权重
            attn_output, attn_weights = transformer_layer.self_attention(
                embedded, embedded, embedded
            )
            attention_weights.append(attn_weights.detach())
            
            # 继续前向传播
            embedded = transformer_layer.norm1(embedded + transformer_layer.dropout(attn_output))
            ff_output = transformer_layer.feed_forward(embedded)
            embedded = transformer_layer.norm2(embedded + transformer_layer.dropout(ff_output))
        
        return attention_weights

class AdvancedFinancialTransformer(nn.Module):
    """高级版本：加入特征类型感知的Transformer"""
    
    def __init__(self, input_dim=30, d_model=128, nhead=8, num_layers=3,
                 dim_feedforward=256, dropout=0.1, num_classes=2):
        super(AdvancedFinancialTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 特征类型嵌入（为不同类型的特征学习不同的嵌入）
        self.feature_type_embedding = nn.Embedding(4, d_model // 4)  # 4种特征类型
        
        # 特征值嵌入
        self.feature_value_projection = nn.Linear(1, d_model - d_model // 4)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=input_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 注意力池化
        self.attention_pooling = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=0)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 预定义特征类型映射 - 注册为buffer
        feature_type_map = self._create_feature_type_map()
        self.register_buffer('feature_type_map', feature_type_map)
    
    def _create_feature_type_map(self):
        """创建特征类型映射"""
        # 0: degree类, 1: profit类, 2: size类, 3: length/count类
        feature_types = []
        
        # 假设30个特征的类型分布（需要根据实际特征调整）
        for i in range(self.input_dim):
            if i < 3:  # degree相关
                feature_types.append(0)
            elif i < 18:  # profit相关
                feature_types.append(1)
            elif i < 24:  # size相关
                feature_types.append(2)
            else:  # length/count相关
                feature_types.append(3)
        
        return torch.tensor(feature_types, dtype=torch.long)
    
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        batch_size = x.size(0)
        
        # 创建嵌入
        embeddings = []
        for i in range(self.input_dim):
            # 特征值嵌入
            feature_val = x[:, i:i+1]  # [batch_size, 1]
            value_embed = self.feature_value_projection(feature_val)  # [batch_size, d_model-d_model//4]
            
            # 特征类型嵌入
            feature_type = self.feature_type_map[i]
            type_embed = self.feature_type_embedding(feature_type)  # [d_model//4]
            type_embed = type_embed.unsqueeze(0).expand(batch_size, -1)  # [batch_size, d_model//4]
            
            # 连接值嵌入和类型嵌入
            combined_embed = torch.cat([value_embed, type_embed], dim=1)  # [batch_size, d_model]
            embeddings.append(combined_embed)
        
        # Stack为序列: [input_dim, batch_size, d_model]
        embedded = torch.stack(embeddings, dim=0)
        
        # 位置编码
        embedded = self.pos_encoding(embedded)
        
        # Transformer编码
        encoded = self.transformer_encoder(embedded)
        
        # 注意力池化
        attention_weights = self.attention_pooling(encoded)  # [input_dim, batch_size, 1]
        weighted_features = encoded * attention_weights  # 广播乘法
        pooled = weighted_features.sum(dim=0)  # [batch_size, d_model]
        
        # 分类
        output = self.classifier(pooled)
        
        return output

class EarlyStopping:
    """早停法类"""
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def preprocess_features(df, feature_cols, show_details=True):
    """数据预处理 - 增强版带详细打印"""
    
    print("=" * 60)
    print("🔧 开始数据预处理...")
    print(f"原始特征数: {len(feature_cols)}")
    
    X = df[feature_cols].copy()
    
    # ===== 1. 检查和删除常数特征 =====
    print("\n📊 检查常数特征...")
    constant_features = []
    feature_stats = {}
    
    for col in X.columns:
        unique_count = X[col].nunique()
        unique_values = X[col].unique()
        
        feature_stats[col] = {
            'unique_count': unique_count,
            'unique_values': unique_values[:5] if len(unique_values) > 5 else unique_values  # 只显示前5个
        }
        
        if unique_count <= 1:
            constant_features.append(col)
            print(f"  ❌ {col}: 只有 {unique_count} 个唯一值 -> {unique_values}")
    
    if constant_features:
        print(f"\n🗑️  删除 {len(constant_features)} 个常数特征: {constant_features}")
        X = X.drop(columns=constant_features)
    else:
        print("✅ 没有发现常数特征")
    
    # ===== 2. 显示数据范围情况 =====
    print(f"\n📈 原始数据范围分析:")
    print("-" * 50)
    
    extreme_features = []
    for col in X.columns:
        min_val, max_val = X[col].min(), X[col].max()
        std_val = X[col].std()
        
        print(f"{col:25} | 范围: [{min_val:>12.2e}, {max_val:>12.2e}] | 标准差: {std_val:>10.2e}")
        
        # 检查极端值
        if max_val > 1e10 or min_val < -1e10:
            extreme_features.append(col)
    
    if extreme_features:
        print(f"\n⚠️  发现极端数值特征: {extreme_features}")
    
    # ===== 3. 处理极端异常值 =====
    print(f"\n🛠️  处理异常值...")
    processing_log = {}
    
    for col in X.columns:
        original_min, original_max = X[col].min(), X[col].max()
        
        if 'profit' in col:
            print(f"\n  处理金额特征: {col}")
            
            # Step 1: log变换
            print(f"    原始范围: [{original_min:.2e}, {original_max:.2e}]")
            X[col] = np.sign(X[col]) * np.log1p(np.abs(X[col]))
            log_min, log_max = X[col].min(), X[col].max()
            print(f"    Log变换后: [{log_min:.4f}, {log_max:.4f}]")
            
            # Step 2: clip异常值
            Q01 = X[col].quantile(0.01)
            Q99 = X[col].quantile(0.99)
            print(f"    1%分位数: {Q01:.4f}, 99%分位数: {Q99:.4f}")
            
            # 统计会被clip的数据
            will_be_clipped = ((X[col] < Q01) | (X[col] > Q99)).sum()
            clip_percentage = will_be_clipped / len(X) * 100
            
            X[col] = np.clip(X[col], Q01, Q99)
            print(f"    Clip后范围: [{X[col].min():.4f}, {X[col].max():.4f}]")
            print(f"    被裁剪数据: {will_be_clipped}/{len(X)} ({clip_percentage:.2f}%)")
            
            processing_log[col] = {
                'type': 'profit',
                'original_range': (original_min, original_max),
                'log_range': (log_min, log_max),
                'clip_range': (Q01, Q99),
                'clipped_count': will_be_clipped
            }
            
        elif 'ratio' in col:
            print(f"\n  处理比例特征: {col}")
            print(f"    原始范围: [{original_min:.4f}, {original_max:.4f}]")
            
            # 统计会被clip的数据
            will_be_clipped = ((X[col] < 0) | (X[col] > 50)).sum()
            clip_percentage = will_be_clipped / len(X) * 100
            
            X[col] = np.clip(X[col], 0, 50)
            print(f"    Clip到[0, 50]: [{X[col].min():.4f}, {X[col].max():.4f}]")
            print(f"    被裁剪数据: {will_be_clipped}/{len(X)} ({clip_percentage:.2f}%)")
            
            processing_log[col] = {
                'type': 'ratio',
                'original_range': (original_min, original_max),
                'clip_range': (0, 50),
                'clipped_count': will_be_clipped
            }
        
        else:
            # 其他特征不处理，只记录
            processing_log[col] = {
                'type': 'other',
                'original_range': (original_min, original_max)
            }
    
    # ===== 4. 标准化 =====
    print(f"\n📏 应用RobustScaler标准化...")
    scaler = RobustScaler()
    
    # 显示标准化前后的统计
    print("标准化前后对比 (前5个特征):")
    print("-" * 60)
    
    for i, col in enumerate(X.columns[:5]):
        before_mean, before_std = X[col].mean(), X[col].std()
        print(f"  {col:20} | 均值: {before_mean:>8.3f} | 标准差: {before_std:>8.3f}")
    
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), 
        columns=X.columns, 
        index=X.index
    )
    
    print("\n标准化后:")
    for i, col in enumerate(X_scaled.columns[:5]):
        after_mean, after_std = X_scaled[col].mean(), X_scaled[col].std()
        print(f"  {col:20} | 均值: {after_mean:>8.3f} | 标准差: {after_std:>8.3f}")
    
    # ===== 5. 最终摘要 =====
    print("\n" + "=" * 60)
    print("📋 预处理完成摘要:")
    print(f"  • 原始特征数: {len(feature_cols)}")
    print(f"  • 删除常数特征: {len(constant_features)}")
    print(f"  • 最终特征数: {X_scaled.shape[1]}")
    print(f"  • 样本数: {X_scaled.shape[0]}")
    
    # 统计各类处理的特征数
    profit_features = [k for k, v in processing_log.items() if v.get('type') == 'profit']
    ratio_features = [k for k, v in processing_log.items() if v.get('type') == 'ratio']
    other_features = [k for k, v in processing_log.items() if v.get('type') == 'other']
    
    print(f"  • 金额特征(log+clip): {len(profit_features)}")
    print(f"  • 比例特征(clip): {len(ratio_features)}")
    print(f"  • 其他特征: {len(other_features)}")
    
    print("=" * 60)
    
    return X_scaled, scaler

def train_transformer_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001, 
                           weight_decay=1e-4, class_weights=None):
    """训练Transformer模型"""
    
    # 损失函数和优化器
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Transformer通常需要更小的学习率
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate*0.5, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # 早停法
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
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
            
            # 梯度裁剪（Transformer训练中很有用）
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
        
        # 计算F1分数
        val_f1 = f1_score(val_targets, val_predictions, average='weighted')
        
        train_losses.append(train_loss / len(train_loader))
        val_f1_scores.append(val_f1)
        
        # 学习率调度
        scheduler.step(val_f1)
        
        # 早停检查
        if early_stopping(val_f1, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Val F1: {val_f1:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
    
    return train_losses, val_f1_scores

def cross_validation_training(X, y, model_type="basic", cv_folds=10, epochs=100):
    """交叉验证训练 - 支持Transformer模型"""
    
    # 计算类别权重
    class_counts = np.bincount(y)
    class_weights = torch.FloatTensor(len(y) / (len(class_counts) * class_counts))
    print(f"类别权重: {class_weights}")
    
    # 分层交叉验证
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    all_folds = list(skf.split(X, y))
    
    print(f"📊 策略: {cv_folds}折交叉验证，模型类型: {model_type}")
    print(f"   每折验证集大小: ~{len(y)//cv_folds} ({100/cv_folds:.1f}%)")
    print(f"   每折训练集大小: ~{len(y)*(cv_folds-1)//cv_folds} ({100*(cv_folds-1)/cv_folds:.1f}%)")
    
    fold_results = []
    
    print(f"\n🚀 开始训练全部 {cv_folds} 折...")
    
    for fold_idx in range(cv_folds):
        train_idx, val_idx = all_folds[fold_idx]
        print(f"\n=== Fold {fold_idx+1}/{cv_folds} ===")
        
        # 分割数据
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")
        
        # 转换为tensor
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.LongTensor(y_val)
        
        # 创建DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # 初始化模型
        if model_type == "basic":
            model = FinancialTransformer(
                input_dim=X.shape[1], 
                d_model=128, 
                nhead=8, 
                num_layers=3,
                dropout=0.1
            ).to(device)
        elif model_type == "advanced":
            model = AdvancedFinancialTransformer(
                input_dim=X.shape[1], 
                d_model=128, 
                nhead=8, 
                num_layers=3,
                dropout=0.1
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 训练模型
        train_losses, val_f1_scores = train_transformer_model(
            model, train_loader, val_loader, 
            epochs=epochs, class_weights=class_weights
        )
        
        # 获取早停轮数和最佳F1分数
        actual_epochs = len(train_losses)
        fold_best_f1 = max(val_f1_scores) if val_f1_scores else 0
        
        # 获取最终预测
        model.eval()
        with torch.no_grad():
            outputs = model(X_val_tensor.to(device))
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.cpu().numpy()
        
        # 计算最终评估F1
        final_eval_f1 = f1_score(y_val, predictions, average='weighted')
        
        print(f"🎯 最佳验证F1: {fold_best_f1:.4f}")
        print(f"最终评估F1: {final_eval_f1:.4f}")
        
        # 存储结果
        fold_results.append({
            'fold_idx': fold_idx,
            'f1_score': fold_best_f1,
            'epochs': actual_epochs,
            'predictions': predictions,
            'val_indices': val_idx,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'final_eval_f1': final_eval_f1
        })
        
        print(f"Fold {fold_idx+1} F1 Score: {fold_best_f1:.4f}, Early stopped at epoch: {actual_epochs}")
    
    # 选择最佳fold
    best_fold = max(fold_results, key=lambda x: x['f1_score'])
    best_fold_idx = best_fold['fold_idx']
    
    print(f"\n🏆 最佳表现: Fold {best_fold_idx+1}")
    print(f"   F1分数: {best_fold['f1_score']:.4f}")
    print(f"   早停轮数: {best_fold['epochs']}")
    print(f"   训练集大小: {best_fold['train_size']}")
    print(f"   验证集大小: {best_fold['val_size']}")
    
    # 收集所有结果
    fold_f1_scores = [r['f1_score'] for r in fold_results]
    fold_predictions = []
    fold_indices = []
    
    for result in fold_results:
        fold_predictions.extend(result['predictions'])
        fold_indices.extend(result['val_indices'])
    
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    optimal_epochs = best_fold['epochs']
    
    print(f"\n=== 交叉验证结果摘要 ===")
    print(f"全部{cv_folds}折F1分数: {fold_f1_scores}")
    print(f"各折早停轮数: {[r['epochs'] for r in fold_results]}")
    print(f"平均F1分数: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"🎯 选用最优训练轮数: {optimal_epochs} (来自最佳Fold {best_fold_idx+1})")
    
    return mean_f1, fold_f1_scores, fold_predictions, fold_indices, optimal_epochs, fold_results, best_fold

def train_final_model(X_train, y_train, optimal_epochs, model_type="basic"):
    """训练最终Transformer模型"""
    
    print(f"\n🚀 训练最终{model_type} Transformer模型 (使用全部{len(y_train)}个训练样本)...")
    print(f"🎯 最优训练轮数: {optimal_epochs}")
    
    # 计算类别权重
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(len(y_train) / (len(class_counts) * class_counts))
    print(f"类别权重: {class_weights}")
    
    # 转换为tensor
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.LongTensor(y_train)
    
    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 初始化模型
    if model_type == "basic":
        model = FinancialTransformer(
            input_dim=X_train.shape[1], 
            d_model=128, 
            nhead=8, 
            num_layers=3,
            dropout=0.1
        ).to(device)
    elif model_type == "advanced":
        model = AdvancedFinancialTransformer(
            input_dim=X_train.shape[1], 
            d_model=128, 
            nhead=8, 
            num_layers=3,
            dropout=0.1
        ).to(device)
    
    # 训练设置
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    print(f"开始训练最终模型 ({optimal_epochs} epochs, 无验证集)...")
    model.train()
    
    for epoch in range(optimal_epochs):
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        if epoch % 5 == 0:
            print(f'Epoch [{epoch+1}/{optimal_epochs}], Training Loss: {avg_loss:.4f}')
    
    print("✅ 最终Transformer模型训练完成!")
    return model

def predict_test_set(model, X_test, test_accounts):
    """对测试集进行预测"""
    
    print("\n🔮 开始预测测试集...")
    
    model.eval()
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
    
    predictions = predictions.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    # 创建预测结果DataFrame
    test_results = pd.DataFrame({
        'account': test_accounts,
        'predicted_label': predictions,
        'probability_good': probabilities[:, 0],
        'probability_bad': probabilities[:, 1]
    })
    
    print(f"测试集预测完成: {len(test_results)} 个账户")
    print(f"预测为Bad的账户: {(predictions == 1).sum()} ({(predictions == 1).mean()*100:.2f}%)")
    
    return test_results

def analyze_attention_patterns(model, X_sample, feature_names):
    """分析注意力模式，了解模型关注哪些特征"""
    
    model.eval()
    
    if hasattr(model, 'get_attention_weights'):
        try:
            attention_weights = model.get_attention_weights(X_sample.to(device))
            
            # 分析第一层的注意力权重
            first_layer_attention = attention_weights[0]  # [batch_size, num_heads, seq_len, seq_len]
            
            # 平均所有head和batch
            avg_attention = first_layer_attention.mean(dim=(0, 1)).cpu().numpy()  # [seq_len, seq_len]
            
            # 可视化注意力矩阵
            plt.figure(figsize=(12, 10))
            sns.heatmap(avg_attention, 
                        xticklabels=feature_names[:len(avg_attention)], 
                        yticklabels=feature_names[:len(avg_attention)],
                        annot=False, cmap='Blues')
            plt.title('Feature-to-Feature Attention Weights (Transformer)')
            plt.tight_layout()
            plt.show()
            
            return avg_attention
        except Exception as e:
            print(f"注意力分析出错: {e}")
            return None
    else:
        print("模型不支持注意力权重提取")
        return None

def main():
    """主函数 - Transformer版本"""
    
    # 1. 加载所有数据
    print("🔄 加载数据...")
    data_path = "/Users/mannormal/4011/Qi Zihan/feature_extraction/generated_features/all_features_complete.csv"
    df = pd.read_csv(data_path)
    
    # 加载训练标签
    train_path = "/Users/mannormal/4011/Qi Zihan/original_data/train_acc.csv"
    train_df = pd.read_csv(train_path)
    
    # 加载测试账户
    test_path = "/Users/mannormal/4011/Qi Zihan/original_data/test_acc_predict.csv"
    test_df = pd.read_csv(test_path)
    
    print(f"特征数据: {df.shape}")
    print(f"训练标签: {train_df.shape}")
    print(f"测试账户: {test_df.shape}")
    
    # 2. 分离训练集和测试集
    train_accounts = set(train_df['account'])
    test_accounts = set(test_df['account'])
    
    # 训练集：有标签的账户
    df_train = df[df['account'].isin(train_accounts)].copy()
    df_train = df_train.merge(train_df[['account', 'flag']], on='account', how='inner')
    df_train['label'] = df_train['flag']
    
    # 测试集：需要预测的账户
    df_test = df[df['account'].isin(test_accounts)].copy()
    
    print(f"训练集大小: {df_train.shape}")
    print(f"测试集大小: {df_test.shape}")
    print(f"训练集标签分布: {df_train['label'].value_counts()}")
    
    # 3. 特征工程
    feature_cols = [col for col in df.columns if col != 'account']
    print(f"\n📊 开始处理训练集特征...")
    X_train, scaler = preprocess_features(df_train, feature_cols)
    y_train = df_train['label'].values
    
    print(f"训练集处理后特征数: {X_train.shape[1]}")
    
    # 4. 选择模型类型
    model_type = "advanced"  # 可以改为 "advanced"
    print(f"\n🏗️ 选择模型类型: {model_type}")
    
    # 5. 交叉验证评估
    print("\n" + "="*60)
    print("📊 开始Transformer交叉验证评估...")
    mean_f1, fold_f1_scores, cv_predictions, cv_indices, optimal_epochs, fold_results, best_fold = cross_validation_training(
        X_train, y_train, model_type=model_type, cv_folds=10, epochs=100
    )
    
    # 6. 处理测试集特征
    print(f"\n📊 开始处理测试集特征...")
    X_test = df_test[feature_cols].copy()
    X_test = X_test[X_train.columns]  # 保持特征一致性
    
    # 应用相同的预处理步骤
    for col in X_test.columns:
        if 'profit' in col:
            X_test[col] = np.sign(X_test[col]) * np.log1p(np.abs(X_test[col]))
            Q01 = X_train[col].quantile(0.01) if col in X_train.columns else X_test[col].min()
            Q99 = X_train[col].quantile(0.99) if col in X_train.columns else X_test[col].max()
            X_test[col] = np.clip(X_test[col], Q01, Q99)
        elif 'ratio' in col:
            X_test[col] = np.clip(X_test[col], 0, 50)
    
    # 应用训练好的scaler
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"测试集处理后形状: {X_test_scaled.shape}")
    
    # 7. 训练最终模型
    final_model = train_final_model(X_train, y_train, optimal_epochs, model_type=model_type)
    
    # 8. 预测测试集
    test_results = predict_test_set(final_model, X_test_scaled, df_test['account'].values)
    
    # 9. 注意力分析（如果支持）
    if hasattr(final_model, 'get_attention_weights'):
        print("\n🔍 进行注意力分析...")
        sample_data = torch.FloatTensor(X_train.iloc[:10].values)
        attention_matrix = analyze_attention_patterns(final_model, sample_data, X_train.columns.tolist())
    
    # 10. 保存结果
    result_dir = "/Users/mannormal/4011/Qi Zihan/result_analysis/prediction_results/"
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存交叉验证结果
    fold_f1_mapping = []
    for result in fold_results:
        fold_size = len(result['predictions'])
        fold_f1_mapping.extend([result['f1_score']] * fold_size)
    
    cv_results_df = pd.DataFrame({
        'account': df_train.iloc[cv_indices]['account'].values,
        'true_label': df_train.iloc[cv_indices]['label'].values,
        'predicted_label': cv_predictions,
        'fold_f1_score': fold_f1_mapping
    })
    
    cv_filename = f"Transformer_{model_type}_cv_analysis_f1_score_{mean_f1:.4f}.csv"
    cv_results_df.to_csv(os.path.join(result_dir, cv_filename), index=False)
    
    # 保存全部数据训练的提交文件
    submission_df_full = pd.DataFrame({
        'account': test_results['account'],
        'Predict': test_results['predicted_label']
    })
    
    test_filename_full = f"Transformer_{model_type}_submission_FULL_DATA_f1_{mean_f1:.4f}_epochs_{optimal_epochs}.csv"
    submission_df_full.to_csv(os.path.join(result_dir, test_filename_full), index=False)
    
    # 使用最佳Fold重新训练和预测
    print(f"\n🏆 使用最佳Fold {best_fold['fold_idx']+1} 的设置重新预测测试集...")
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_folds = list(skf.split(X_train, y_train))
    best_train_idx, best_val_idx = all_folds[best_fold['fold_idx']]
    
    X_best_train = X_train.iloc[best_train_idx]
    y_best_train = y_train[best_train_idx]
    
    best_fold_model = train_final_model(X_best_train, y_best_train, best_fold['epochs'], model_type=model_type)
    best_fold_test_results = predict_test_set(best_fold_model, X_test_scaled, df_test['account'].values)
    
    submission_df_best = pd.DataFrame({
        'account': best_fold_test_results['account'],
        'Predict': best_fold_test_results['predicted_label']
    })
    
    test_filename_best = f"Transformer_{model_type}_submission_BEST_FOLD_{best_fold['fold_idx']+1}_f1_{best_fold['f1_score']:.4f}_epochs_{best_fold['epochs']}.csv"
    submission_df_best.to_csv(os.path.join(result_dir, test_filename_best), index=False)
    
    print(f"\n🎉 Transformer训练完成！生成了两个提交文件：")
    print(f"📊 交叉验证平均F1: {mean_f1:.4f}")
    print(f"🏆 最佳Fold F1: {best_fold['f1_score']:.4f}")
    print(f"🤖 模型类型: {model_type} Transformer")
    print(f"📁 交叉验证分析文件: {cv_filename}")
    print(f"📄 全部数据提交文件: {test_filename_full}")
    print(f"🥇 最佳Fold提交文件: {test_filename_best}")
    
    # 对比两个预测结果
    print(f"\n📊 两个提交文件对比:")
    print(f"全部数据模型预测分布: {submission_df_full['Predict'].value_counts().to_dict()}")
    print(f"最佳Fold模型预测分布: {submission_df_best['Predict'].value_counts().to_dict()}")
    
    # 计算两个预测的一致性
    agreement = (submission_df_full['Predict'] == submission_df_best['Predict']).mean()
    print(f"两个模型预测一致性: {agreement:.4f} ({agreement*100:.2f}%)")
    
    return mean_f1, cv_results_df, submission_df_full, submission_df_best, best_fold

if __name__ == "__main__":
    print("🚀 启动Transformer金融账户分类系统...")
    mean_f1_score, cv_results, submission_full, submission_best, best_fold_info = main()