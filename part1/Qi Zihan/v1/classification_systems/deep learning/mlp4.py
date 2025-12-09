import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
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
    
    def __init__(self, input_dim=53, dropout_rates=[0.3, 0.2, 0.2, 0.1, 0.1]):
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
            break
    
    return train_losses, val_f1_scores

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

# ========== CV预测和分析 ==========
def cv_prediction_analysis(X, y, test_data, train_data, config, cv_folds=10):
    """进行CV预测分析并保存每个fold的结果"""
    
    print(f"\n🔄 {cv_folds}折交叉验证预测分析")
    print("=" * 60)
    print(f"使用配置: {config['name']}")
    print(f"配置详情: {config}")
    
    # 计算类别权重
    class_counts = np.bincount(y)
    class_weights = torch.FloatTensor(len(y) / (len(class_counts) * class_counts))
    
    # 准备测试数据
    df_test = test_data['data'].copy()
    X_test = test_data['features'].copy()
    
    # 准备训练数据用于OOF预测
    df_train_full = train_data['data'].copy()
    X_train_full = train_data['features'].copy()
    
    # CV分割
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # 存储结果
    fold_results = []
    test_predictions_all = []  # 存储每个fold对测试集的预测
    oof_predictions = np.zeros(len(y))  # Out-of-fold预测
    oof_probabilities = np.zeros((len(y), 2))
    
    # 创建结果目录
    result_dir = "/Users/mannormal/4011/Qi Zihan/result_analysis/prediction_results"
    os.makedirs(result_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_idx+1}/{cv_folds} ---")
        
        X_train_fold, X_val_fold = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # 数据准备
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_fold.values),
            torch.LongTensor(y_train_fold)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_fold.values),
            torch.LongTensor(y_val_fold)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # 模型训练
        model = RefinedMLP(input_dim=X.shape[1]).to(device)
        train_losses, val_f1_scores = train_refined_model(
            model, train_loader, val_loader, config, class_weights, epochs=100
        )
        
        best_f1 = max(val_f1_scores)
        
        # 验证集预测(用于OOF)
        val_preds, val_probs = predict_with_model(model, val_loader)
        oof_predictions[val_idx] = val_preds
        oof_probabilities[val_idx] = val_probs
        
        # 测试集预测
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test.values),
            torch.zeros(len(X_test))  # dummy labels
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        test_preds, test_probs = predict_with_model(model, test_loader)
        
        test_predictions_all.append(test_preds)
        
        # 保存单个fold结果
        fold_result = {
            'fold': fold_idx + 1,
            'f1': best_f1,
            'epochs': len(train_losses),
            'val_indices': val_idx.tolist(),
            'test_predictions': test_preds,
            'test_probabilities': test_probs
        }
        fold_results.append(fold_result)
        
        print(f"Fold {fold_idx+1} F1: {best_f1:.4f}, Epochs: {len(train_losses)}")
    
    # 计算整体统计
    cv_f1_scores = [r['f1'] for r in fold_results]
    mean_f1 = np.mean(cv_f1_scores)
    std_f1 = np.std(cv_f1_scores)
    
    # OOF F1分数
    oof_f1 = f1_score(y, oof_predictions, average='weighted')
    
    print(f"\n🎉 CV结果统计:")
    print(f"   平均F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"   OOF F1: {oof_f1:.4f}")
    print(f"   各折F1: {[f'{f:.4f}' for f in cv_f1_scores]}")
    
    # 测试集集成预测
    test_predictions_ensemble = np.array(test_predictions_all)
    test_pred_final = np.round(np.mean(test_predictions_ensemble, axis=0)).astype(int)
    
    # 分析预测一致性
    prediction_consistency = []
    for i in range(len(test_pred_final)):
        unique_preds = np.unique(test_predictions_ensemble[:, i])
        consistency = np.max(np.bincount(test_predictions_ensemble[:, i].astype(int))) / cv_folds
        prediction_consistency.append(consistency)
    
    avg_consistency = np.mean(prediction_consistency)
    print(f"   测试集预测一致性: {avg_consistency:.4f}")
    
    # 保存详细结果
    # 1. 训练集OOF预测
    oof_df = pd.DataFrame({
        'account': df_train_full['account'].values,
        'true_label': y,
        'oof_prediction': oof_predictions,
        'oof_prob_0': oof_probabilities[:, 0],
        'oof_prob_1': oof_probabilities[:, 1]
    })
    oof_filename = f"MLP_OOF_predictions_{timestamp}_f1_{oof_f1:.4f}.csv"
    oof_df.to_csv(os.path.join(result_dir, oof_filename), index=False)
    
    # 2. 测试集预测
    test_df = pd.DataFrame({
        'account': df_test['account'].values,
        'prediction': test_pred_final
    })
    test_filename = f"MLP_test_predictions_{timestamp}_cv_f1_{mean_f1:.4f}.csv"
    test_df.to_csv(os.path.join(result_dir, test_filename), index=False)
    
    # 3. 每个fold的测试集预测
    fold_predictions_df = pd.DataFrame(
        test_predictions_ensemble.T,
        columns=[f'fold_{i+1}' for i in range(cv_folds)]
    )
    fold_predictions_df['account'] = df_test['account'].values
    fold_predictions_df['ensemble_pred'] = test_pred_final
    fold_predictions_df['consistency'] = prediction_consistency
    
    fold_filename = f"MLP_fold_predictions_{timestamp}.csv"
    fold_predictions_df.to_csv(os.path.join(result_dir, fold_filename), index=False)
    
    # 4. CV详细分析
    analysis_results = {
        'config': config,
        'cv_mean_f1': mean_f1,
        'cv_std_f1': std_f1,
        'oof_f1': oof_f1,
        'fold_f1_scores': cv_f1_scores,
        'prediction_consistency': avg_consistency,
        'test_pred_distribution': np.bincount(test_pred_final).tolist(),
        'timestamp': timestamp
    }
    
    return analysis_results, fold_results, {
        'oof_filename': oof_filename,
        'test_filename': test_filename,
        'fold_filename': fold_filename
    }

def analyze_cv_differences(fold_results, X_test):
    """分析CV折之间的预测差异"""
    
    print(f"\n📊 CV预测差异分析")
    print("=" * 40)
    
    # 提取所有fold的测试集预测
    all_preds = np.array([fold['test_predictions'] for fold in fold_results])
    n_folds, n_samples = all_preds.shape
    
    # 计算每个样本的预测方差
    pred_variance = np.var(all_preds, axis=0)
    
    # 统计预测一致性
    unanimous_count = 0  # 所有fold预测一致
    majority_count = 0   # 大多数fold预测一致
    split_count = 0      # 预测分歧严重
    
    for i in range(n_samples):
        unique_preds, counts = np.unique(all_preds[:, i], return_counts=True)
        max_count = np.max(counts)
        
        if max_count == n_folds:
            unanimous_count += 1
        elif max_count >= n_folds * 0.7:
            majority_count += 1
        else:
            split_count += 1
    
    print(f"📈 预测一致性统计:")
    print(f"   完全一致: {unanimous_count} ({unanimous_count/n_samples*100:.1f}%)")
    print(f"   大多一致: {majority_count} ({majority_count/n_samples*100:.1f}%)")
    print(f"   预测分歧: {split_count} ({split_count/n_samples*100:.1f}%)")
    print(f"   平均预测方差: {np.mean(pred_variance):.4f}")
    
    # F1分数分析
    f1_scores = [fold['f1'] for fold in fold_results]
    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)
    
    print(f"\n📊 F1分数分析:")
    print(f"   F1范围: {np.min(f1_scores):.4f} - {np.max(f1_scores):.4f}")
    print(f"   F1标准差: {f1_std:.4f}")
    print(f"   F1变异系数: {f1_std/f1_mean*100:.2f}%")
    
    # 识别高分歧样本
    high_variance_idx = np.where(pred_variance > np.percentile(pred_variance, 90))[0]
    print(f"\n🎯 高分歧样本分析:")
    print(f"   高分歧样本数: {len(high_variance_idx)} (top 10%)")
    
    return {
        'unanimous_ratio': unanimous_count / n_samples,
        'majority_ratio': majority_count / n_samples,
        'split_ratio': split_count / n_samples,
        'avg_pred_variance': np.mean(pred_variance),
        'f1_stability': f1_std / f1_mean,
        'high_variance_samples': len(high_variance_idx)
    }

def get_top5_results():
    """获取top5结果"""
    
    # 最优配置
    best_config = {'weight_decay': 0.0005, 'lr': 0.001, 'smoothing': 0.03, 'name': '保守改进'}

    return [best_config]

# ========== 主函数 ==========
def main_cv_analysis():
    """主要的CV分析函数"""
    
    print("="*80)
    print("🎯 MLP CV预测分析 - 保存每个fold结果")
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
    
    # 处理测试数据
    test_accounts = set(test_df['account'])
    df_test = df[df['account'].isin(test_accounts)].copy()
    
    print(f"训练数据: {df_train.shape}")
    print(f"测试数据: {df_test.shape}")
    print(f"标签分布: {np.bincount(df_train['label'])}")
    
    # 特征预处理
    time_cols = ['first_transaction_time', 'last_transaction_time'] 
    feature_cols = [col for col in df.columns if col not in ['account'] + time_cols]
    
    print(f"📊 使用 {len(feature_cols)} 个数值特征")
    
    # 训练集特征处理
    X_train = df_train[feature_cols].copy()
    
    # 基础预处理
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
            Q01 = X_train[col].quantile(0.01)  # 使用训练集的分位数
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
    
    # 准备数据字典
    train_data = {
        'data': df_train,
        'features': X_train_scaled
    }
    test_data = {
        'data': df_test,
        'features': X_test_scaled
    }
    
    # 2. 获取top5配置
    top_configs = get_top5_results()
    
    print(f"\n🏆 运行Top5配置:")
    for i, config in enumerate(top_configs, 1):
        print(f"   {i}. {config['name']}: {config}")
    
    # 3. 运行每个配置的CV分析
    all_results = []
    
    for i, config in enumerate(top_configs, 1):
        print(f"\n" + "="*60)
        print(f"🚀 配置 {i}/{len(top_configs)}: {config['name']}")
        print("="*60)
        
        try:
            analysis_result, fold_results, filenames = cv_prediction_analysis(
                X_train_scaled, y_train, test_data, train_data, config, cv_folds=10
            )
            
            # CV差异分析
            cv_diff_analysis = analyze_cv_differences(fold_results, X_test_scaled)
            
            # 合并结果
            analysis_result.update(cv_diff_analysis)
            analysis_result['filenames'] = filenames
            all_results.append(analysis_result)
            
            print(f"✅ 配置 {config['name']} 完成")
            print(f"   CV F1: {analysis_result['cv_mean_f1']:.4f} ± {analysis_result['cv_std_f1']:.4f}")
            print(f"   OOF F1: {analysis_result['oof_f1']:.4f}")
            print(f"   预测一致性: {analysis_result['unanimous_ratio']:.3f}")
            
        except Exception as e:
            print(f"❌ 配置 {config['name']} 失败: {str(e)}")
            continue
    
    # 4. 总结最佳结果
    if all_results:
        # 按OOF F1排序
        all_results.sort(key=lambda x: x['oof_f1'], reverse=True)
        
        print(f"\n" + "="*80)
        print("🎉 Top5配置最终排名 (按OOF F1排序)")
        print("="*80)
        
        for i, result in enumerate(all_results, 1):
            config = result['config']
            print(f"\n🏆 第{i}名: {config['name']}")
            print(f"   CV F1: {result['cv_mean_f1']:.4f} ± {result['cv_std_f1']:.4f}")
            print(f"   OOF F1: {result['oof_f1']:.4f}")
            print(f"   预测一致性: {result['unanimous_ratio']:.3f}")
            print(f"   F1稳定性: {result['f1_stability']:.4f}")
            print(f"   文件: {result['filenames']['test_filename']}")
        
        # 保存汇总结果
        result_dir = "/Users/mannormal/4011/Qi Zihan/result_analysis/prediction_results"
        summary_df = pd.DataFrame([
            {
                'rank': i+1,
                'config_name': r['config']['name'],
                'weight_decay': r['config']['weight_decay'],
                'lr': r['config']['lr'],
                'smoothing': r['config']['smoothing'],
                'cv_mean_f1': r['cv_mean_f1'],
                'cv_std_f1': r['cv_std_f1'],
                'oof_f1': r['oof_f1'],
                'unanimous_ratio': r['unanimous_ratio'],
                'f1_stability': r['f1_stability'],
                'test_filename': r['filenames']['test_filename']
            }
            for i, r in enumerate(all_results)
        ])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"MLP_top5_summary_{timestamp}.csv"
        summary_df.to_csv(os.path.join(result_dir, summary_filename), index=False)
        
        print(f"\n📁 汇总文件已保存: {summary_filename}")
        
        return all_results
    
    else:
        print("❌ 没有成功的配置结果")
        return None

if __name__ == "__main__":
    results = main_cv_analysis()


