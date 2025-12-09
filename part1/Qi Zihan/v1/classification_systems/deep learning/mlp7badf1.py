import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
import os
from datetime import datetime

# 设置设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class RefinedMLP(nn.Module):
    """精简版MLP"""
    
    def __init__(self, input_dim=53, dropout_rates=[0.3, 0.2, 0.2, 0.1, 0.1]):
        super(RefinedMLP, self).__init__()
        
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

class LightLabelSmoothingCE(nn.Module):
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
                if self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}

def train_model(model, train_loader, val_loader, config, class_weights, epochs=100):
    """训练模型 - 使用Bad客户F1分数作为主要指标"""
    
    criterion = LightLabelSmoothingCE(
        smoothing=config['smoothing'], 
        class_weights=class_weights.to(device)
    )
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=6, min_lr=config['lr']*0.01
    )
    early_stopping = SimpleEarlyStopping(patience=12, min_delta=0.001)
    
    # 存储详细指标
    epoch_metrics = []
    
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
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        # 计算全面的指标 - 重点关注Bad客户F1
        val_f1_macro = f1_score(val_targets, val_predictions, average='macro')
        val_f1_weighted = f1_score(val_targets, val_predictions, average='weighted')
        val_accuracy = accuracy_score(val_targets, val_predictions)
        val_precision_weighted = precision_score(val_targets, val_predictions, average='weighted', zero_division=0)
        val_recall_weighted = recall_score(val_targets, val_predictions, average='weighted', zero_division=0)
        
        # 计算各类别的详细指标
        val_precision_per_class = precision_score(val_targets, val_predictions, average=None, zero_division=0)
        val_recall_per_class = recall_score(val_targets, val_predictions, average=None, zero_division=0)
        val_f1_per_class = f1_score(val_targets, val_predictions, average=None, zero_division=0)
        
        # 特别关注坏客户(类别1)的指标 - 作为主要指标
        bad_precision = val_precision_per_class[1] if len(val_precision_per_class) > 1 else 0
        bad_recall = val_recall_per_class[1] if len(val_recall_per_class) > 1 else 0
        bad_f1 = val_f1_per_class[1] if len(val_f1_per_class) > 1 else 0  # 主要指标
        
        # 好客户(类别0)的指标
        good_precision = val_precision_per_class[0]
        good_recall = val_recall_per_class[0]
        good_f1 = val_f1_per_class[0]
        
        # 存储本轮指标 - Bad F1作为主要指标
        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'bad_f1': bad_f1,  # 主要指标 - Bad客户F1
            'f1_macro': val_f1_macro,
            'f1_weighted': val_f1_weighted,
            'accuracy': val_accuracy,
            'precision_weighted': val_precision_weighted,
            'recall_weighted': val_recall_weighted,
            'good_precision': good_precision,
            'good_recall': good_recall,
            'good_f1': good_f1,
            'bad_precision': bad_precision,
            'bad_recall': bad_recall
        })
        
        # 使用Bad F1进行学习率调度和早停
        scheduler.step(bad_f1)
        
        # 每10轮或最后几轮打印详细指标 - 突出显示Bad F1
        if epoch % 10 == 0 or epoch >= epochs - 3:
            print(f'Epoch [{epoch+1:3d}]: Bad_F1={bad_f1:.4f}, mF1={val_f1_macro:.4f}, F1_w={val_f1_weighted:.4f}, Acc={val_accuracy:.4f}')
            print(f'              Good: Prec={good_precision:.4f}, Recall={good_recall:.4f}, F1={good_f1:.4f}')
            print(f'              Bad:  Prec={bad_precision:.4f}, Recall={bad_recall:.4f}, F1={bad_f1:.4f}')
        
        if early_stopping(bad_f1, model):  # 使用Bad F1进行早停
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return epoch_metrics

def predict_with_model(model, data_loader):
    """使用模型进行预测"""
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)

def evaluate_detailed_metrics(y_true, y_pred, title=""):
    """详细的指标评估"""
    print(f"\n📊 {title}详细指标:")
    
    # 基础指标
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # 各类别详细指标
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Bad客户F1作为主要指标
    bad_f1 = f1_per_class[1] if len(f1_per_class) > 1 else 0
    
    print(f"   🎯 Bad客户F1(主要指标): {bad_f1:.4f}")
    print(f"   准确率(Accuracy): {accuracy:.4f}")
    print(f"   F1分数(加权): {f1_weighted:.4f}")
    print(f"   F1分数(宏平均): {f1_macro:.4f}")
    print(f"   精确率(加权): {precision_weighted:.4f}")
    print(f"   召回率(加权): {recall_weighted:.4f}")
    
    print(f"\n   📈 各类别指标:")
    class_names = ['Good客户', 'Bad客户']
    for i, name in enumerate(class_names):
        if i < len(precision_per_class):
            marker = "🎯" if i == 1 else "  "  # Bad客户特殊标记
            print(f" {marker} {name}: Precision={precision_per_class[i]:.4f}, "
                  f"Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n   📋 混淆矩阵:")
    print(f"              预测Good  预测Bad")
    print(f"   实际Good    {cm[0,0]:6d}   {cm[0,1]:6d}")
    if cm.shape[0] > 1:
        print(f"   实际Bad     {cm[1,0]:6d}   {cm[1,1]:6d}")
    
    # 风控特别关注的指标
    if cm.shape[0] > 1:
        # 坏客户检出率 (Bad Recall)
        bad_recall = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
        # 坏客户预测精确率 (Bad Precision)  
        bad_precision = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0
        # 误杀率 (Good客户被错误预测为Bad的比例)
        false_positive_rate = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        
        print(f"\n   🎯 风控关键指标:")
        print(f"   坏客户检出率: {bad_recall:.4f} ({bad_recall*100:.1f}%)")
        print(f"   坏客户预测准确率: {bad_precision:.4f} ({bad_precision*100:.1f}%)")
        print(f"   好客户误杀率: {false_positive_rate:.4f} ({false_positive_rate*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'bad_f1': bad_f1,  # 新增Bad F1
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm
    }

def main_best_config_cv():
    """主函数 - 只使用最佳配置进行CV分析，使用Bad客户F1"""
    
    print("=" * 80)
    print("🎯 MLP最佳配置CV预测分析 - Bad客户F1指标版")
    print("=" * 80)
    
    # 最佳配置
    best_config = {
        'weight_decay': 0.0005, 
        'lr': 0.001, 
        'smoothing': 0.03, 
        'name': '保守改进_BadF1'
    }
    
    print(f"使用配置: {best_config}")
    print("主要评估指标: Bad客户F1分数 (Class 1 F1)")
    
    # 1. 数据加载
    print("\n📂 数据加载...")
    data_path = "/Users/mannormal/4011/Qi Zihan/feature_extraction/generated_features/all_features_super_optimized.csv"
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
    
    # 处理测试数据
    test_accounts = set(test_df['account'])
    df_test = df[df['account'].isin(test_accounts)].copy()
    
    print(f"训练数据: {df_train.shape}")
    print(f"测试数据: {df_test.shape}")
    print(f"标签分布: {np.bincount(df_train['label'])}")
    
    # 2. 特征预处理
    time_cols = ['first_transaction_time', 'last_transaction_time']
    feature_cols = [col for col in df.columns if col not in ['account'] + time_cols]
    
    print(f"📊 使用 {len(feature_cols)} 个特征")
    
    # 训练集特征处理
    X_train = df_train[feature_cols].copy()
    for col in X_train.columns:
        if 'profit' in col.lower():
            X_train[col] = np.sign(X_train[col]) * np.log1p(np.abs(X_train[col]))
            Q01, Q99 = X_train[col].quantile([0.01, 0.99])
            X_train[col] = np.clip(X_train[col], Q01, Q99)
        elif 'ratio' in col.lower():
            X_train[col] = np.clip(X_train[col], 0, 50)
    
    # 测试集特征处理
    X_test = df_test[feature_cols].copy()
    for col in X_test.columns:
        if 'profit' in col.lower():
            X_test[col] = np.sign(X_test[col]) * np.log1p(np.abs(X_test[col]))
            Q01 = X_train[col].quantile(0.01)
            Q99 = X_train[col].quantile(0.99)
            X_test[col] = np.clip(X_test[col], Q01, Q99)
        elif 'ratio' in col.lower():
            X_test[col] = np.clip(X_test[col], 0, 50)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    y_train = df_train['label'].values
    
    # 3. 10折交叉验证
    print(f"\n🔄 10折交叉验证 (主要指标: Bad客户F1)...")
    
    # 计算类别权重
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(len(y_train) / (len(class_counts) * class_counts))
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # 存储结果
    fold_results = []
    test_predictions_all = []
    oof_predictions = np.zeros(len(y_train))
    oof_probabilities = np.zeros((len(y_train), 2))
    
    # 存储详细指标
    all_fold_metrics = []
    
    # 创建结果目录
    result_dir = "/Users/mannormal/4011/Qi Zihan/result_analysis/prediction_results"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        print(f"\n--- Fold {fold_idx+1}/10 ---")
        
        X_fold_train, X_fold_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        print(f"训练集大小: {len(y_fold_train)}, 验证集大小: {len(y_fold_val)}")
        print(f"训练集标签分布: {np.bincount(y_fold_train)}")
        print(f"验证集标签分布: {np.bincount(y_fold_val)}")
        
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
        model = RefinedMLP(input_dim=X_train_scaled.shape[1]).to(device)
        epoch_metrics = train_model(
            model, train_loader, val_loader, best_config, class_weights, epochs=100
        )
        
        # 获取最佳轮次的指标 - 使用Bad F1
        best_epoch_idx = np.argmax([m['bad_f1'] for m in epoch_metrics])
        best_metrics = epoch_metrics[best_epoch_idx]
        
        # 验证集预测(OOF)
        val_preds, val_probs = predict_with_model(model, val_loader)
        oof_predictions[val_idx] = val_preds
        oof_probabilities[val_idx] = val_probs
        
        # 详细评估验证集
        fold_detailed_metrics = evaluate_detailed_metrics(
            y_fold_val, val_preds, f"Fold {fold_idx+1} 验证集"
        )
        
        # 测试集预测
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled.values),
            torch.zeros(len(X_test_scaled))
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        test_preds, test_probs = predict_with_model(model, test_loader)
        
        test_predictions_all.append(test_preds)
        
        # 记录fold结果
        fold_results.append({
            'fold': fold_idx + 1,
            'best_metrics': best_metrics,
            'detailed_metrics': fold_detailed_metrics,
            'epochs': len(epoch_metrics),
            'val_indices': val_idx,
            'test_predictions': test_preds,
            'test_probabilities': test_probs
        })
        
        all_fold_metrics.append(epoch_metrics)
        
        print(f"Fold {fold_idx+1} 最佳: Bad_F1={best_metrics['bad_f1']:.4f}, "
              f"mF1={best_metrics['f1_macro']:.4f}, Acc={best_metrics['accuracy']:.4f}, "
              f"训练轮数: {len(epoch_metrics)}")
    
    # 4. 整体OOF评估
    print(f"\n" + "="*60)
    print("📊 整体Out-of-Fold(OOF)评估结果 (主要关注Bad客户F1)")
    print("="*60)
    
    oof_detailed_metrics = evaluate_detailed_metrics(
        y_train, oof_predictions, "OOF整体"
    )
    
    # 5. CV统计分析 - 将Bad F1作为第一指标
    cv_metrics = {
        'bad_f1': [r['best_metrics']['bad_f1'] for r in fold_results],  # 主要指标放在第一位
        'f1_macro': [r['best_metrics']['f1_macro'] for r in fold_results],
        'f1_weighted': [r['best_metrics']['f1_weighted'] for r in fold_results],
        'accuracy': [r['best_metrics']['accuracy'] for r in fold_results],
        'precision_weighted': [r['best_metrics']['precision_weighted'] for r in fold_results],
        'recall_weighted': [r['best_metrics']['recall_weighted'] for r in fold_results],
        'bad_precision': [r['best_metrics']['bad_precision'] for r in fold_results],
        'bad_recall': [r['best_metrics']['bad_recall'] for r in fold_results]
    }
    
    print(f"\n📈 10折交叉验证统计 (主要指标: Bad客户F1):")
    for metric_name, values in cv_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        marker = "🎯" if metric_name == 'bad_f1' else "  "
        print(f"{marker} {metric_name}: {mean_val:.4f} ± {std_val:.4f} "
              f"(范围: {min(values):.4f} - {max(values):.4f})")
    
    # 6. 保存增强的结果文件
    print(f"\n💾 保存增强结果到: {result_dir}")
    
    # 保存每个fold的详细训练过程
    all_training_history = []
    for fold_idx, epoch_metrics in enumerate(all_fold_metrics):
        for epoch_data in epoch_metrics:
            epoch_data['fold'] = fold_idx + 1
            all_training_history.append(epoch_data)
    
    training_history_df = pd.DataFrame(all_training_history)
    training_history_filename = f"MLP_training_history_BadF1_{timestamp}.csv"
    training_history_df.to_csv(os.path.join(result_dir, training_history_filename), index=False)
    print(f"   训练历史: {training_history_filename}")
    
    # 保存CV详细指标统计
    cv_detailed_stats = []
    for metric_name, values in cv_metrics.items():
        cv_detailed_stats.append({
            'metric': metric_name,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'fold_1': values[0] if len(values) > 0 else None,
            'fold_2': values[1] if len(values) > 1 else None,
            'fold_3': values[2] if len(values) > 2 else None,
            'fold_4': values[3] if len(values) > 3 else None,
            'fold_5': values[4] if len(values) > 4 else None,
            'fold_6': values[5] if len(values) > 5 else None,
            'fold_7': values[6] if len(values) > 6 else None,
            'fold_8': values[7] if len(values) > 7 else None,
            'fold_9': values[8] if len(values) > 8 else None,
            'fold_10': values[9] if len(values) > 9 else None,
        })
    
    cv_detailed_df = pd.DataFrame(cv_detailed_stats)
    cv_detailed_filename = f"MLP_cv_detailed_metrics_BadF1_{timestamp}.csv"
    cv_detailed_df.to_csv(os.path.join(result_dir, cv_detailed_filename), index=False)
    print(f"   CV详细指标: {cv_detailed_filename}")
    
    # 最终总结增强版 - 突出Bad F1
    print(f"\n" + "=" * 80)
    print("🎉 增强分析完成总结 (Bad客户F1指标)")
    print("=" * 80)
    print(f"📊 最佳配置: {best_config['name']}")
    print(f"🎯 CV Bad客户F1: {np.mean(cv_metrics['bad_f1']):.4f} ± {np.std(cv_metrics['bad_f1']):.4f}")
    print(f"⭐ CV宏平均F1: {np.mean(cv_metrics['f1_macro']):.4f} ± {np.std(cv_metrics['f1_macro']):.4f}")
    print(f"📈 CV加权F1: {np.mean(cv_metrics['f1_weighted']):.4f} ± {np.std(cv_metrics['f1_weighted']):.4f}")
    print(f"🔍 CV准确率: {np.mean(cv_metrics['accuracy']):.4f} ± {np.std(cv_metrics['accuracy']):.4f}")
    print(f"🎯 OOF Bad客户F1: {oof_detailed_metrics['bad_f1']:.4f}")
    print(f"⭐ OOF宏平均F1: {oof_detailed_metrics['f1_macro']:.4f}")
    print(f"📋 OOF加权F1: {oof_detailed_metrics['f1_weighted']:.4f}")
    print(f"🔍 OOF准确率: {oof_detailed_metrics['accuracy']:.4f}")
    print(f"⚡ 坏客户检出率: {np.mean(cv_metrics['bad_recall']):.4f} ± {np.std(cv_metrics['bad_recall']):.4f}")
    print(f"🎯 坏客户预测准确率: {np.mean(cv_metrics['bad_precision']):.4f} ± {np.std(cv_metrics['bad_precision']):.4f}")
    
    return {
        'config': best_config,
        'cv_metrics': cv_metrics,
        'oof_metrics': oof_detailed_metrics,
        'filenames': {
            'training_history': training_history_filename,
            'cv_detailed': cv_detailed_filename,
        }
    }

if __name__ == "__main__":
    results = main_best_config_cv()