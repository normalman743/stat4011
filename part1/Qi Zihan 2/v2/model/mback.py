import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import warnings
import random
import os
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

seed_num = 13
set_seed(seed_num)

print("=== 过拟合实验：最大化 Bad F1 ===")

# =====================================================
# 模型保存函数
# =====================================================
def save_model_and_artifacts(model, scaler, results, save_dir='saved_models'):
    """
    保存模型、预处理器和训练结果
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型权重
    model_path = os.path.join(save_dir, f'model_weights_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型权重已保存: {model_path}")
    
    # 保存完整模型(包含结构)
    full_model_path = os.path.join(save_dir, f'full_model_{timestamp}.pth')
    torch.save(model, full_model_path)
    print(f"完整模型已保存: {full_model_path}")
    
    # 保存预处理器
    scaler_path = os.path.join(save_dir, f'scaler_{timestamp}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"预处理器已保存: {scaler_path}")
    
    # 保存训练结果和超参数
    results_path = os.path.join(save_dir, f'training_results_{timestamp}.pkl')
    save_results = {
        'final_f1_bad': results['final_f1_bad'],
        'best_bad_f1': results['best_bad_f1'],
        'best_epoch': results['best_epoch'],
        'final_acc': results['final_acc'],
        'model_architecture': {
            'input_dim': model.layer1[0].in_features,
            'hidden_dims': [64, 32, 16],
            'output_dim': 1
        },
        'training_config': {
            'n_epochs': 2000,
            'learning_rate': 1e-3,
            'optimizer': 'Adam',
            'regularization': 'None',
            'early_stopping': False,
            'seed': seed_num
        },
        'timestamp': timestamp
    }
    
    with open(results_path, 'wb') as f:
        pickle.dump(save_results, f)
    print(f"训练结果已保存: {results_path}")
    
    # 创建模型信息文件
    info_path = os.path.join(save_dir, f'model_info_{timestamp}.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"模型训练信息\n")
        f.write(f"==================\n")
        f.write(f"训练时间: {timestamp}\n")
        f.write(f"随机种子: {seed_num}\n")
        f.write(f"模型架构: {model.layer1[0].in_features} → 64 → 32 → 16 → 1\n")
        f.write(f"训练轮数: 2000\n")
        f.write(f"学习率: 1e-3\n")
        f.write(f"优化器: Adam\n")
        f.write(f"正则化: 无\n")
        f.write(f"早停: 无\n\n")
        f.write(f"性能指标\n")
        f.write(f"==================\n")
        f.write(f"最终Bad F1: {results['final_f1_bad']:.4f}\n")
        f.write(f"历史最佳Bad F1: {results['best_bad_f1']:.4f} (第{results['best_epoch']}轮)\n")
        f.write(f"最终准确率: {results['final_acc']:.4f}\n\n")
        f.write(f"文件说明\n")
        f.write(f"==================\n")
        f.write(f"model_weights_{timestamp}.pth - 模型权重文件\n")
        f.write(f"full_model_{timestamp}.pth - 完整模型文件\n")
        f.write(f"scaler_{timestamp}.pkl - 数据预处理器\n")
        f.write(f"training_results_{timestamp}.pkl - 训练结果和配置\n")
        f.write(f"model_info_{timestamp}.txt - 此信息文件\n")
    
    print(f"模型信息已保存: {info_path}")
    
    return {
        'model_path': model_path,
        'full_model_path': full_model_path,
        'scaler_path': scaler_path,
        'results_path': results_path,
        'info_path': info_path,
        'timestamp': timestamp
    }

def load_model_and_artifacts(timestamp, save_dir='saved_models'):
    """
    加载保存的模型和相关组件
    """
    # 加载完整模型
    full_model_path = os.path.join(save_dir, f'full_model_{timestamp}.pth')
    model = torch.load(full_model_path)
    print(f"模型已加载: {full_model_path}")
    
    # 加载预处理器
    scaler_path = os.path.join(save_dir, f'scaler_{timestamp}.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"预处理器已加载: {scaler_path}")
    
    # 加载训练结果
    results_path = os.path.join(save_dir, f'training_results_{timestamp}.pkl')
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    print(f"训练结果已加载: {results_path}")
    
    return model, scaler, results

# =====================================================
# 简化的Meta-ANN模型（去除正则化）
# =====================================================
class SimplifiedMetaANN(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        
        # 64 -> 32 -> 16 -> 1 架构，无正则化
        self.layer1 = nn.Sequential(
            nn.Linear(n_feat, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
            # 去除dropout
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
            # 去除dropout
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
            # 去除dropout
        )
        
        self.out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.sigmoid(self.out(x))

def train_overfitting_model(features, y_true, n_epochs=2000):
    """
    完全过拟合训练，最大化bad F1
    """
    print(f"\n训练过拟合模型")
    print(f"特征维度: {features.shape}")
    print(f"训练轮数: {n_epochs}")
    print(f"正则化: 无")
    print(f"早停: 无")
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 转换为张量
    X_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_true.reshape(-1,1), dtype=torch.float32).to(device)
    
    # 创建模型
    model = SimplifiedMetaANN(n_feat=features_scaled.shape[1]).to(device)
    
    # 优化器（去除权重衰减）
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 去除weight_decay
    criterion = nn.BCELoss()
    
    print(f"\n开始训练...")
    print("Epoch | Train F1 | Good F1  | Bad F1   | Train Acc| Loss     | Status")
    print("-" * 75)
    
    best_bad_f1 = 0
    best_epoch = 0
    
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        
        # 评估阶段
        model.eval()
        with torch.no_grad():
            y_pred_prob = model(X_tensor).cpu().numpy().flatten()
            y_pred_label = (y_pred_prob > 0.5).astype(int)
            
            # 计算各种F1分数
            train_f1_overall = metrics.f1_score(y_true, y_pred_label, average='binary', zero_division=0)
            train_f1_good = metrics.f1_score(y_true, y_pred_label, pos_label=0, zero_division=0)  # good=0
            train_f1_bad = metrics.f1_score(y_true, y_pred_label, pos_label=1, zero_division=0)   # bad=1
            train_acc = metrics.accuracy_score(y_true, y_pred_label)
            
            # 记录最佳bad F1
            if train_f1_bad > best_bad_f1:
                best_bad_f1 = train_f1_bad
                best_epoch = epoch
                best_model_state = model.state_dict().copy()
                status = "🏆 Best Bad F1"
            else:
                status = ""
        
        # 打印进度
        if epoch % 100 == 0 or status:
            print(f"{epoch:5d} | {train_f1_overall:8.4f} | {train_f1_good:8.4f} | {train_f1_bad:8.4f} | {train_acc:8.4f} | {loss.item():8.4f} | {status}")
    
    # 加载最佳模型
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        final_pred_prob = model(X_tensor).cpu().numpy().flatten()
        final_pred_label = (final_pred_prob > 0.5).astype(int)
        
        final_f1_overall = metrics.f1_score(y_true, final_pred_label, average='binary', zero_division=0)
        final_f1_good = metrics.f1_score(y_true, final_pred_label, pos_label=0, zero_division=0)  # good=0
        final_f1_bad = metrics.f1_score(y_true, final_pred_label, pos_label=1, zero_division=0)   # bad=1
        final_f1_macro = metrics.f1_score(y_true, final_pred_label, average='macro', zero_division=0)
        final_f1_weighted = metrics.f1_score(y_true, final_pred_label, average='weighted', zero_division=0)
        final_acc = metrics.accuracy_score(y_true, final_pred_label)
        
        # 分类报告
        print(f"\n" + "="*75)
        print("最终过拟合结果:")
        print(f"   训练准确率: {final_acc:.4f}")
        print(f"   整体F1: {final_f1_overall:.4f}")
        print(f"   Good类F1 (pos_label=0): {final_f1_good:.4f}")
        print(f"   Bad类F1 (pos_label=1): {final_f1_bad:.4f}")
        print(f"   宏平均F1: {final_f1_macro:.4f}")
        print(f"   加权平均F1: {final_f1_weighted:.4f}")
        print(f"   最佳Bad F1出现在第{best_epoch}轮: {best_bad_f1:.4f}")
        
        # 混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, final_pred_label)
        print(f"\n混淆矩阵:")
        print(f"   预测\\实际  Good(0)  Bad(1)")
        print(f"   Good(0)    {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"   Bad(1)     {cm[1,0]:6d}  {cm[1,1]:6d}")
        
        # 预测分布
        pred_counts = np.bincount(final_pred_label)
        total_samples = len(final_pred_label)
        print(f"\n预测分布:")
        print(f"   预测为Good(0): {pred_counts[0]} ({pred_counts[0]/total_samples*100:.1f}%)")
        print(f"   预测为Bad(1): {pred_counts[1]} ({pred_counts[1]/total_samples*100:.1f}%)")
        
        # 真实分布
        true_counts = np.bincount(y_true)
        print(f"\n真实分布:")
        print(f"   真实Good(0): {true_counts[0]} ({true_counts[0]/total_samples*100:.1f}%)")
        print(f"   真实Bad(1): {true_counts[1]} ({true_counts[1]/total_samples*100:.1f}%)")
        
    return {
        'model': model,
        'scaler': scaler,
        'final_f1_bad': final_f1_bad,
        'best_bad_f1': best_bad_f1,
        'best_epoch': best_epoch,
        'final_acc': final_acc,
        'final_predictions': final_pred_label,
        'final_probabilities': final_pred_prob
    }

def main():
    # =====================================================
    # 数据加载
    # =====================================================
    print("\n=== 加载数据 ===")
    
    # 特征数据
    features_path = '/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v2/feature_extraction/result/features_cleaned_no_leakage1.csv'
    all_features_df = pd.read_csv(features_path)
    print(f"特征数据: {all_features_df.shape}")
    
    # 完整标签数据
    labels_path = '/Users/mannormal/Desktop/课程/y4t1/stat 4011/融合二分模型_最终版 copy.csv'
    labels_df = pd.read_csv(labels_path)
    print(f"标签数据: {labels_df.shape}")
    
    # 重命名列以匹配
    labels_df = labels_df.rename(columns={'ID': 'account', 'Predict': 'flag'})
    
    # 去除特征数据中可能存在的flag列
    cols_to_drop = []
    if 'flag' in all_features_df.columns:
        cols_to_drop.append('flag')
    if 'data_type' in all_features_df.columns:
        cols_to_drop.append('data_type')
    
    if cols_to_drop:
        print(f"从特征数据中删除列: {cols_to_drop}")
        all_features_df = all_features_df.drop(cols_to_drop, axis=1)
    
    # 合并数据
    full_df = pd.merge(all_features_df, labels_df[['account', 'flag']], on='account', how='inner')
    print(f"合并后数据: {full_df.shape}")
    
    # 检查标签分布
    flag_counts = full_df['flag'].value_counts()
    print(f"标签分布: {dict(flag_counts)}")
    
    # 准备特征和标签
    feature_cols = [col for col in full_df.columns if col not in ['account', 'flag']]
    features = full_df[feature_cols].values
    y_true = full_df['flag'].values
    
    print(f"最终特征矩阵: {features.shape}")
    print(f"标签分布: Good(0): {np.sum(y_true==0)}, Bad(1): {np.sum(y_true==1)}")
    print(f"类别不平衡比例: 1:{np.sum(y_true==0)/max(np.sum(y_true==1), 1):.2f}")
    
    # =====================================================
    # 过拟合训练
    # =====================================================
    print(f"\n{'='*75}")
    print("开始过拟合实验")
    print(f"{'='*75}")
    
    results = train_overfitting_model(features, y_true, n_epochs=2000)
    
    # =====================================================
    # 结果总结
    # =====================================================
    print(f"\n{'='*75}")
    print("过拟合实验完成")
    print(f"{'='*75}")
    
    print(f"\n🎯 关键指标:")
    print(f"   最终Bad F1: {results['final_f1_bad']:.4f}")
    print(f"   历史最佳Bad F1: {results['best_bad_f1']:.4f} (第{results['best_epoch']}轮)")
    print(f"   最终准确率: {results['final_acc']:.4f}")
    
    print(f"\n📊 实验设置:")
    print(f"   数据集: 完整训练+测试集 ({full_df.shape[0]} 样本)")
    print(f"   模型: 简化Meta-ANN (无正则化)")
    print(f"   架构: {features.shape[1]} → 64 → 32 → 16 → 1")
    print(f"   训练轮数: 2000")
    print(f"   优化器: Adam (lr=1e-3, 无权重衰减)")
    print(f"   早停: 无")
    print(f"   数据增强: 无")
    print(f"   验证集: 无 (完全过拟合)")
    
    print(f"\n✅ 过拟合实验结束!")
    print(f"   理论上限Bad F1: {results['best_bad_f1']:.4f}")
    
    # =====================================================
    # 保存模型和相关组件
    # =====================================================
    print(f"\n{'='*75}")
    print("保存模型和相关组件")
    print(f"{'='*75}")
    
    save_paths = save_model_and_artifacts(results['model'], results['scaler'], results)
    
    print(f"\n💾 模型保存完成!")
    print(f"   时间戳: {save_paths['timestamp']}")
    print(f"   保存目录: saved_models/")
    print(f"   模型权重: {os.path.basename(save_paths['model_path'])}")
    print(f"   完整模型: {os.path.basename(save_paths['full_model_path'])}")
    print(f"   预处理器: {os.path.basename(save_paths['scaler_path'])}")
    print(f"   训练结果: {os.path.basename(save_paths['results_path'])}")
    print(f"   模型信息: {os.path.basename(save_paths['info_path'])}")
    
    return results

if __name__ == "__main__":
    results = main()
    
    # 示例：如何加载保存的模型进行预测
    # 注释掉下面的代码，需要时取消注释
    """
    # 加载保存的模型示例
    print(f"\n{'='*75}")
    print("模型加载示例")
    print(f"{'='*75}")
    
    # 使用保存的时间戳加载模型
    saved_timestamp = "20241216_143052"  # 替换为实际的时间戳
    
    try:
        loaded_model, loaded_scaler, loaded_results = load_model_and_artifacts(saved_timestamp)
        
        print(f"\n🔄 模型加载成功!")
        print(f"   模型架构: {loaded_results['model_architecture']}")
        print(f"   训练配置: {loaded_results['training_config']}")
        print(f"   最佳性能: Bad F1 = {loaded_results['best_bad_f1']:.4f}")
        
        # 使用加载的模型进行预测示例
        # 假设有新的特征数据 new_features
        # new_features_scaled = loaded_scaler.transform(new_features)
        # loaded_model.eval()
        # with torch.no_grad():
        #     predictions = loaded_model(torch.tensor(new_features_scaled, dtype=torch.float32))
        #     pred_labels = (predictions.numpy() > 0.5).astype(int)
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
    """