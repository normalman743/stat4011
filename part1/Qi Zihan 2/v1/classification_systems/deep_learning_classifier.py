import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class FinancialDataset(Dataset):
    """金融数据集类"""
    def __init__(self, X, y, augment=False, noise_level=0.01):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.noise_level = noise_level
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment:
            x = self._augment_features(x)
            
        return x, y
    
    def _augment_features(self, x):
        """特征增强"""
        # 添加高斯噪声
        noise = torch.randn_like(x) * self.noise_level
        
        # 特征dropout (随机置零10%的特征)
        dropout_mask = torch.rand_like(x) > 0.1
        
        # 轻微缩放
        scale = torch.normal(1.0, 0.02, x.shape)
        
        augmented_x = x * dropout_mask * scale + noise
        return augmented_x

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AttentionMLP(nn.Module):
    """带注意力机制的多层感知机"""
    def __init__(self, input_dim, hidden_dims=[256, 512, 256, 128], num_classes=2, dropout_rate=0.3):
        super(AttentionMLP, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
            nn.Tanh(),
            nn.Linear(hidden_dims[0] // 2, hidden_dims[0]),
            nn.Sigmoid()  # 使用Sigmoid获得0-1权重
        )
        
        # 深层网络
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate * (0.8 ** i))  # 逐层减少dropout
            ])
        
        self.deep_layers = nn.Sequential(*layers)
        
        # 分类器
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征编码
        encoded = self.feature_encoder(x)
        
        # 注意力权重
        attention_weights = self.attention(encoded)
        
        # 加权特征
        attended_features = encoded * attention_weights
        
        # 深层处理
        deep_features = self.deep_layers(attended_features)
        
        # 分类
        output = self.classifier(deep_features)
        
        return output, attention_weights

class GroupedFeatureNet(nn.Module):
    """分组特征网络"""
    def __init__(self, feature_groups, num_classes=2):
        super(GroupedFeatureNet, self).__init__()
        
        # 为每个特征组创建子网络
        self.group_nets = nn.ModuleDict()
        self.group_outputs = []
        
        for group_name, group_size in feature_groups.items():
            self.group_nets[group_name] = nn.Sequential(
                nn.Linear(group_size, max(group_size * 2, 32)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(max(group_size * 2, 32), 64),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.group_outputs.append(64)
        
        # 特征融合网络
        fusion_input_dim = sum(self.group_outputs)
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x, feature_indices):
        """
        x: 输入特征
        feature_indices: 每个组的特征索引字典
        """
        group_outputs = []
        
        for group_name, net in self.group_nets.items():
            if group_name in feature_indices:
                indices = feature_indices[group_name]
                group_features = x[:, indices]
                group_output = net(group_features)
                group_outputs.append(group_output)
        
        # 拼接所有组的输出
        fused_features = torch.cat(group_outputs, dim=1)
        
        # 最终分类
        output = self.fusion_net(fused_features)
        return output

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_dim, out_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        
        # 如果输入输出维度不同，需要投影
        self.projection = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
    def forward(self, x):
        residual = self.projection(x)
        out = self.block(x)
        return F.relu(out + residual)

class ResidualFinancialNet(nn.Module):
    """残差金融网络"""
    def __init__(self, input_dim, num_classes=2):
        super(ResidualFinancialNet, self).__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, 256)
        
        # 残差块
        self.res_blocks = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 512),
            ResidualBlock(512, 256),
            ResidualBlock(256, 128)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = F.relu(self.input_proj(x))
        x = self.res_blocks(x)
        output = self.classifier(x)
        return output

class DeepLearningClassifier:
    """深度学习分类器主类"""
    
    def __init__(self, model_type='attention', device=None):
        self.model_type = model_type
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.models = []
        self.scalers = []
        self.label_encoders = {}
        self.feature_names = None
        
        print(f"使用设备: {self.device}")
    
    def _prepare_data(self, df):
        """数据预处理"""
        df = df.copy()
        
        # 处理分类特征
        categorical_features = ['traditional_category', 'volume_category', 'profit_category', 
                              'interaction_category', 'behavior_category']
        
        for col in categorical_features:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # 处理布尔特征
        boolean_features = ['has_forward_cnt', 'has_backward_cnt', 'has_A_forward', 
                           'has_B_forward', 'has_A_backward', 'has_B_backward']
        
        for col in boolean_features:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        return df
    
    def _create_model(self, input_dim):
        """创建模型"""
        if self.model_type == 'attention':
            return AttentionMLP(input_dim)
        elif self.model_type == 'residual':
            return ResidualFinancialNet(input_dim)
        elif self.model_type == 'grouped':
            # 定义特征组
            feature_groups = {
                'profit_features': 15,  # 各种profit特征
                'size_features': 15,    # 各种size特征
                'categorical_features': 5,  # 分类特征
                'boolean_features': 6,   # 布尔特征
                'summary_features': 3    # 总体统计特征
            }
            return GroupedFeatureNet(feature_groups)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _get_class_weights(self, y):
        """计算类别权重"""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return torch.FloatTensor(weights).to(self.device)
    
    def _create_weighted_sampler(self, y):
        """创建加权采样器"""
        class_counts = Counter(y)
        total_samples = len(y)
        
        # 计算每个类别的权重
        weights = []
        for label in y:
            weights.append(1.0 / class_counts[label])
        
        return WeightedRandomSampler(weights, total_samples, replacement=True)
    
    def train_single_model(self, X_train, y_train, X_val=None, y_val=None, 
                          epochs=200, batch_size=256, lr=0.001):
        """训练单个模型"""
        
        # 创建模型
        model = self._create_model(X_train.shape[1]).to(self.device)
        
        # 创建数据加载器
        train_dataset = FinancialDataset(X_train, y_train, augment=True)
        sampler = self._create_weighted_sampler(y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        
        if X_val is not None:
            val_dataset = FinancialDataset(X_val, y_val, augment=False)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 损失函数和优化器
        class_weights = self._get_class_weights(y_train)
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        # criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 训练历史
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                if self.model_type == 'attention':
                    outputs, attention = model(batch_x)
                else:
                    outputs = model(batch_x)
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            scheduler.step()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            if X_val is not None:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        
                        if self.model_type == 'attention':
                            outputs, _ = model(batch_x)
                        else:
                            outputs = model(batch_x)
                        
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # 早停
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # 保存最佳模型状态
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"早停在第 {epoch+1} 轮")
                    # 恢复最佳模型状态
                    model.load_state_dict(best_model_state)
                    break
                
                if (epoch + 1) % 20 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            else:
                if (epoch + 1) % 20 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}')
        
        return model, train_losses, val_losses
    
    def train_ensemble(self, df, target_column, n_models=5, test_size=0.2, cv_folds=5):
        """训练集成模型"""
        
        # 数据预处理
        df_processed = self._prepare_data(df)
        
        # 分离特征和标签
        feature_columns = [col for col in df_processed.columns if col not in [target_column, 'account']]
        X = df_processed[feature_columns].values
        y = df_processed[target_column].values
        
        self.feature_names = feature_columns
        
        print(f"特征维度: {X.shape}")
        print(f"类别分布: {Counter(y)}")
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.test_data = (X_test_scaled, y_test)
        
        # 交叉验证训练多个模型
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        all_val_predictions = []
        all_val_labels = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
            print(f"\n训练第 {fold+1}/{cv_folds} 折...")
            
            X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # 训练多个模型
            fold_models = []
            for model_idx in range(n_models):
                print(f"  训练模型 {model_idx+1}/{n_models}")
                
                model, train_losses, val_losses = self.train_single_model(
                    X_fold_train, y_fold_train, X_fold_val, y_fold_val,
                    epochs=200, batch_size=256, lr=0.001
                )
                
                fold_models.append(model)
            
            self.models.extend(fold_models)
            self.scalers.extend([scaler] * n_models)
            
            # 验证集预测
            fold_predictions = self._predict_ensemble_fold(fold_models, X_fold_val)
            all_val_predictions.extend(fold_predictions)
            all_val_labels.extend(y_fold_val)
        
        # 验证集性能
        val_accuracy = np.mean(np.array(all_val_predictions) == np.array(all_val_labels))
        print(f"\n交叉验证平均准确率: {val_accuracy:.4f}")
        
        # 测试集评估
        test_predictions = self.predict(X_test_scaled)
        test_accuracy = np.mean(test_predictions == y_test)
        
        print(f"测试集准确率: {test_accuracy:.4f}")
        print("\n测试集分类报告:")
        print(classification_report(y_test, test_predictions))
        
        return {
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'test_predictions': test_predictions,
            'test_labels': y_test
        }
    
    def _predict_ensemble_fold(self, models, X):
        """单折集成预测"""
        predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                
                if self.model_type == 'attention':
                    outputs, _ = model(X_tensor)
                else:
                    outputs = model(X_tensor)
                
                probs = F.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).cpu().numpy()
                predictions.append(pred)
        
        # 投票决策
        predictions = np.array(predictions)
        ensemble_pred = []
        for i in range(predictions.shape[1]):
            votes = predictions[:, i]
            ensemble_pred.append(np.bincount(votes).argmax())
        
        return ensemble_pred
    
    def predict(self, X):
        """集成预测"""
        if not self.models:
            raise ValueError("模型尚未训练")
        
        all_predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                
                if self.model_type == 'attention':
                    outputs, _ = model(X_tensor)
                else:
                    outputs = model(X_tensor)
                
                probs = F.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).cpu().numpy()
                all_predictions.append(pred)
        
        # 集成投票
        all_predictions = np.array(all_predictions)
        ensemble_predictions = []
        
        for i in range(all_predictions.shape[1]):
            votes = all_predictions[:, i]
            ensemble_predictions.append(np.bincount(votes).argmax())
        
        return np.array(ensemble_predictions)
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.models:
            raise ValueError("模型尚未训练")
        
        all_probabilities = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                
                if self.model_type == 'attention':
                    outputs, _ = model(X_tensor)
                else:
                    outputs = model(X_tensor)
                
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                all_probabilities.append(probs)
        
        # 平均概率
        mean_probabilities = np.mean(all_probabilities, axis=0)
        return mean_probabilities
    
    def get_feature_importance(self, X_sample):
        """获取特征重要性（仅适用于attention模型）"""
        if self.model_type != 'attention':
            print("特征重要性分析仅适用于attention模型")
            return None
        
        if not self.models:
            raise ValueError("模型尚未训练")
        
        all_attentions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_sample).to(self.device)
                outputs, attention = model(X_tensor)
                all_attentions.append(attention.cpu().numpy())
        
        # 平均注意力权重
        mean_attention = np.mean(all_attentions, axis=0)
        mean_attention = np.mean(mean_attention, axis=0)  # 对样本维度平均
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_attention
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_training_history(self, train_losses, val_losses):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='训练损失')
        if val_losses:
            plt.plot(val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('训练过程')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.show()
        
        return cm

def main():
    """主函数示例"""
    
    # 读取数据
    print("加载数据...")
    df = pd.read_csv('/Users/mannormal/4011/Qi Zihan/feature_extraction/generated_features/all_features_with_categories.csv')
    
    # 读取标签数据
    train_acc = pd.read_csv('/Users/mannormal/4011/Qi Zihan/original_data/train_acc.csv')
    test_acc = pd.read_csv('/Users/mannormal/4011/Qi Zihan/original_data/test_acc_predict.csv')
    
    # 重命名标签列 - 这是关键修复！
    train_acc = train_acc.rename(columns={'flag': 'label'})
    test_acc = test_acc.rename(columns={'Predict': 'label'})
    
    # 合并标签
    all_labels = pd.concat([train_acc, test_acc], ignore_index=True)
    
    # 合并特征和标签
    df_merged = df.merge(all_labels, on='account', how='inner')
    
    print(f"合并后数据形状: {df_merged.shape}")
    print(f"标签分布: \n{df_merged['label'].value_counts()}")
    
    # 测试不同模型类型
    model_types = ['attention', 'residual']  # 'grouped'需要特殊处理
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"训练 {model_type.upper()} 模型")
        print('='*50)
        
        # 创建分类器
        classifier = DeepLearningClassifier(model_type=model_type)
        
        # 训练集成模型
        result = classifier.train_ensemble(
            df_merged, 
            target_column='label',
            n_models=3,  # 每折3个模型
            cv_folds=3   # 3折交叉验证，总共9个模型
        )
        
        results[model_type] = result
        
        # 绘制混淆矩阵
        classifier.plot_confusion_matrix(result['test_labels'], result['test_predictions'])
        
        # 如果是attention模型，显示特征重要性
        if model_type == 'attention':
            X_test, y_test = classifier.test_data
            importance = classifier.get_feature_importance(X_test[:100])  # 使用前100个样本
            
            print("\n前20个重要特征:")
            print(importance.head(20))
            
            # 可视化特征重要性
            plt.figure(figsize=(12, 8))
            top_features = importance.head(20)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('注意力权重')
            plt.title('Top 20 特征重要性')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
    
    # 结果对比
    print(f"\n{'='*50}")
    print("模型性能对比")
    print('='*50)
    
    for model_type, result in results.items():
        print(f"{model_type.upper()}:")
        print(f"  验证准确率: {result['val_accuracy']:.4f}")
        print(f"  测试准确率: {result['test_accuracy']:.4f}")
        
        # 计算其他指标
        y_true, y_pred = result['test_labels'], result['test_predictions']
        report = classification_report(y_true, y_pred, output_dict=True)
        
        print(f"  F1-Score (类别1): {report['1']['f1-score']:.4f}")
        print(f"  精确率 (类别1): {report['1']['precision']:.4f}")
        print(f"  召回率 (类别1): {report['1']['recall']:.4f}")
        print()

if __name__ == "__main__":
    main()