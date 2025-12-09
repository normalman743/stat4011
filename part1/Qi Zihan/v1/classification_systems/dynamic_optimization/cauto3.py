import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedEnsembleOptimizer:
    def __init__(self):
        """基于您当前最佳Enhanced Ensemble方法的优化器"""
        self.best_params = None
        self.best_score = 0
        self.optimization_history = []
        self.label_encoders = {}  # 存储标签编码器
        
    def load_enhanced_features(self):
        """加载增强特征数据 - 使用您当前最佳方法的特征"""
        print("=== 加载增强特征数据 ===")
        
        # 方案1: 使用all_features_with_categories.csv (您的增强特征)
        try:
            features_df = pd.read_csv('../../feature_extraction/generated_features/all_features_with_categories.csv')
            print(f"使用增强特征集: {features_df.shape[1]-1}个特征")
        except:
            # 方案2: 使用基础特征
            features_df = pd.read_csv('../../feature_extraction/generated_features/all_features.csv')
            print(f"使用基础特征集: {features_df.shape[1]-1}个特征")
        
        # 加载训练标签
        train_labels = pd.read_csv('../../original_data/train_acc.csv')
        
        # 合并数据
        data = features_df.merge(train_labels, on='account', how='inner')
        
        # 处理分类特征
        data = self.encode_categorical_features(data)
        
        print(f"最终数据形状: {data.shape}")
        print(f"特征列数: {len([col for col in data.columns if col not in ['account', 'flag']])}")
        print(f"标签分布: {data['flag'].value_counts().to_dict()}")
        
        return data
    
    def encode_categorical_features(self, data):
        """编码分类特征"""
        print("=== 处理分类特征 ===")
        
        # 获取特征列
        feature_cols = [col for col in data.columns if col not in ['account', 'flag']]
        
        # 检查每列的数据类型
        categorical_cols = []
        for col in feature_cols:
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                categorical_cols.append(col)
                print(f"发现分类特征: {col}, 唯一值: {data[col].unique()}")
        
        # 编码分类特征
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            self.label_encoders[col] = le
            print(f"编码 {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # 确保所有特征都是数值型并处理缺失值
        for col in feature_cols:
            if data[col].dtype == 'object':
                print(f"强制转换 {col} 为数值型")
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # 统一处理所有缺失值
            if data[col].isnull().any():
                print(f"填充 {col} 的 {data[col].isnull().sum()} 个缺失值")
                data[col] = data[col].fillna(0)
            
            # 确保数据类型为float64
            data[col] = data[col].astype(np.float64)
        
        print(f"分类特征编码完成，共处理 {len(categorical_cols)} 个分类特征")
        
        # 最终检查数据类型
        print("=== 最终数据类型检查 ===")
        for col in feature_cols[:5]:  # 只打印前5列作为示例
            print(f"{col}: {data[col].dtype}, 包含NaN: {data[col].isnull().any()}")
        
        return data
    
    def enhanced_ensemble_predict(self, data, n_models=100, sample_size=532, 
                                voting_threshold=93, random_state=42,
                                n_estimators=100, max_depth=None, 
                                min_samples_split=2, min_samples_leaf=1,
                                trained_models=None, predict_data=None):
        """
        增强集成预测 - 支持训练和预测分离
        """
        np.random.seed(random_state)
        
        # 准备特征
        feature_cols = [col for col in data.columns if col not in ['account', 'flag']]
        
        # 再次确保特征是数值型
        data_clean = data.copy()
        for col in feature_cols:
            if data_clean[col].dtype == 'object':
                print(f"警告: {col} 仍为object类型，强制转换")
                data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')
            
            # 统一填充缺失值
            data_clean[col] = data_clean[col].fillna(0).astype(np.float64)
        
        # 如果提供了训练好的模型，直接用于预测
        if trained_models is not None and predict_data is not None:
            # 处理预测数据
            predict_data_clean = predict_data.copy()
            for col in feature_cols:
                if predict_data_clean[col].dtype == 'object':
                    predict_data_clean[col] = pd.to_numeric(predict_data_clean[col], errors='coerce')
                predict_data_clean[col] = predict_data_clean[col].fillna(0).astype(np.float64)
            
            print(f"使用 {len(trained_models)} 个训练好的模型进行预测...")
            y_preds1 = []
            
            x_pred = predict_data_clean[feature_cols].values.astype(np.float64)
            x_pred = np.where(np.isfinite(x_pred), x_pred, 0)
            
            for i, clf in enumerate(trained_models):
                if (i + 1) % 20 == 0:
                    print(f"  完成预测 {i+1}/{len(trained_models)} 个模型")
                pred = clf.predict(x_pred)
                y_preds1.append(pred)
            
            # 集成投票
            y_pred1 = np.where(
                pd.DataFrame(y_preds1).sum(axis='rows').values > voting_threshold, 
                1, 0
            )
            return y_pred1
        
        # 原始训练+预测逻辑
        data1 = data_clean.copy()
        y_preds1 = []
        trained_models = []
        
        print(f"开始训练 {n_models} 个模型...")
        for i in range(n_models):
            if (i + 1) % 20 == 0:
                print(f"  完成 {i+1}/{n_models} 个模型")
            
            # 平衡采样
            positive_samples = len(data1[data1.flag == 1])
            negative_samples = len(data1[data1.flag == 0])
            
            train11 = data1[data1.flag == 1].sample(
                min(sample_size, positive_samples), 
                random_state=random_state + i
            )
            train10 = data1[data1.flag == 0].sample(
                min(sample_size, negative_samples), 
                random_state=random_state + i + 1000
            )
            
            # 合并训练数据
            train1 = pd.concat([train10, train11], axis='rows')
            
            # 准备训练特征和标签
            x_train1 = train1[feature_cols].values.astype(np.float64)
            y_train1 = train1.flag.values.astype(np.int64)
            
            # 检查数据完整性
            try:
                if np.any(pd.isnull(x_train1)) or not np.isfinite(x_train1).all():
                    print(f"警告: 模型 {i} 的训练数据包含NaN或无穷值，填充为0")
                    x_train1 = np.where(np.isfinite(x_train1), x_train1, 0)
            except Exception as e:
                print(f"数据检查异常: {e}，使用默认填充")
                x_train1 = np.where(np.isfinite(x_train1), x_train1, 0)
            
            # 创建RandomForest
            clf1 = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state + i,
                n_jobs=-1
            )
            
            # 训练模型
            clf1.fit(x_train1, y_train1)
            trained_models.append(clf1)
            
            # 预测数据
            x_pred = data1[feature_cols].values.astype(np.float64)
            try:
                if np.any(pd.isnull(x_pred)) or not np.isfinite(x_pred).all():
                    x_pred = np.where(np.isfinite(x_pred), x_pred, 0)
            except Exception as e:
                x_pred = np.where(np.isfinite(x_pred), x_pred, 0)
            
            pred = clf1.predict(x_pred)
            y_preds1.append(pred)
        
        # 集成投票
        y_pred1 = np.where(
            pd.DataFrame(y_preds1).sum(axis='rows').values > voting_threshold, 
            1, 0
        )
        
        return y_pred1, trained_models
    
    def optimize_enhanced_ensemble(self, data, target_metric='f1_binary', max_iterations=30):
        """优化增强集成模型参数"""
        print(f"\n=== 基于Enhanced Ensemble方法进行参数优化 ===")
        print(f"目标指标: {target_metric}")
        
        # 重要修复: 复制原始代码的标签处理
        data_copy = data.copy()
        data_copy.loc[data_copy['flag'] == -1, 'flag'] = 0
        
        # 创建验证集
        train_data, val_data = train_test_split(
            data_copy, test_size=0.2, random_state=42, stratify=data_copy['flag']
        )
        
        print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")
        print(f"验证集标签分布: {val_data['flag'].value_counts().to_dict()}")
        
        # 大幅扩展参数搜索空间
        param_combinations = [
            # 原始enhanced_natxis_classification.py的参数
            {'n_models': 100, 'sample_size': 532, 'voting_threshold': 93, 'n_estimators': 100},
            
            # 探索不同的投票阈值
            {'n_models': 100, 'sample_size': 532, 'voting_threshold': 90, 'n_estimators': 100},
            {'n_models': 100, 'sample_size': 532, 'voting_threshold': 95, 'n_estimators': 100},
            {'n_models': 100, 'sample_size': 532, 'voting_threshold': 85, 'n_estimators': 100},
            {'n_models': 100, 'sample_size': 532, 'voting_threshold': 80, 'n_estimators': 100},
            {'n_models': 100, 'sample_size': 532, 'voting_threshold': 75, 'n_estimators': 100},
            
            # 探索不同的样本大小
            {'n_models': 100, 'sample_size': 400, 'voting_threshold': 93, 'n_estimators': 100},
            {'n_models': 100, 'sample_size': 600, 'voting_threshold': 93, 'n_estimators': 100},
            {'n_models': 100, 'sample_size': 700, 'voting_threshold': 93, 'n_estimators': 100},
            {'n_models': 100, 'sample_size': 300, 'voting_threshold': 93, 'n_estimators': 100},
            
            # 探索不同的模型数量
            {'n_models': 80, 'sample_size': 532, 'voting_threshold': 75, 'n_estimators': 100},
            {'n_models': 120, 'sample_size': 532, 'voting_threshold': 110, 'n_estimators': 100},
            {'n_models': 150, 'sample_size': 532, 'voting_threshold': 140, 'n_estimators': 100},
            
            # 探索不同的随机森林参数
            {'n_models': 100, 'sample_size': 532, 'voting_threshold': 93, 'n_estimators': 80},
            {'n_models': 100, 'sample_size': 532, 'voting_threshold': 93, 'n_estimators': 120},
            {'n_models': 100, 'sample_size': 532, 'voting_threshold': 93, 'n_estimators': 150},
            
            # 组合优化
            {'n_models': 80, 'sample_size': 400, 'voting_threshold': 75, 'n_estimators': 120},
            {'n_models': 120, 'sample_size': 600, 'voting_threshold': 110, 'n_estimators': 80},
            {'n_models': 150, 'sample_size': 700, 'voting_threshold': 140, 'n_estimators': 100},
        ]
        
        best_score = 0
        best_params = None
        
        for i, params in enumerate(param_combinations):
            if i >= max_iterations:
                break
                
            print(f"\n--- 测试配置 {i+1}/{min(len(param_combinations), max_iterations)} ---")
            print(f"参数: {params}")
            
            start_time = time.time()
            
            try:
                # 在训练集上训练模型
                train_predictions, trained_models = self.enhanced_ensemble_predict(train_data, **params)
                
                # 使用训练好的模型在验证集上预测
                val_predictions = self.enhanced_ensemble_predict(
                    train_data, trained_models=trained_models, predict_data=val_data, **params
                )
                val_metrics = self.evaluate_model(val_data['flag'].values, val_predictions)
                
                runtime = time.time() - start_time
                score = val_metrics[target_metric]
                
                print(f"验证集性能:")
                print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  F1-Binary: {val_metrics['f1_binary']:.4f}")
                print(f"  F1-Weighted: {val_metrics['f1_weighted']:.4f}")
                print(f"  Precision: {val_metrics['precision']:.4f}")
                print(f"  Recall: {val_metrics['recall']:.4f}")
                print(f"  运行时间: {runtime:.1f}秒")
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"  ★ 新最佳! {target_metric}: {score:.4f}")
                
                self.optimization_history.append({
                    'iteration': i + 1,
                    'params': params.copy(),
                    'val_metrics': val_metrics,
                    'runtime': runtime
                })
                
            except Exception as e:
                print(f"  错误: {e}")
                continue
        
        self.best_params = best_params
        self.best_score = best_score
        
        print(f"\n=== 优化完成 ===")
        print(f"最佳 {target_metric}: {best_score:.4f}")
        print(f"最佳参数: {best_params}")
        
        return best_params, best_score
    
    def evaluate_model(self, y_true, y_pred):
        """评估模型性能"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_binary': f1_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0)
        }
        return metrics
    
    def train_and_test_final_model(self, data):
        """在验证集上训练和测试最终模型"""
        print(f"\n=== 使用最佳参数训练并在验证集上测试最终模型 ===")
        
        if self.best_params is None:
            print("错误: 请先运行 optimize_enhanced_ensemble()!")
            return None, None
        
        print(f"使用最佳参数: {self.best_params}")
        
        # 创建验证集（保持与优化阶段一致）
        train_data, val_data = train_test_split(
            data, test_size=0.2, random_state=42, stratify=data['flag']
        )
        
        start_time = time.time()
        
        # 在训练集上训练，验证集上预测
        train_predictions, trained_models = self.enhanced_ensemble_predict(train_data, **self.best_params)
        val_predictions = self.enhanced_ensemble_predict(
            train_data, trained_models=trained_models, predict_data=val_data, **self.best_params
        )
        val_metrics = self.evaluate_model(val_data['flag'].values, val_predictions)
        
        training_time = time.time() - start_time
        
        print(f"最终模型在验证集上的性能:")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"运行时间: {training_time:.1f}秒")
        
        return val_predictions, val_metrics
    
    def generate_optimized_code(self, timestamp):
        """生成优化后的可复现代码"""
        if self.best_params is None:
            print("错误: 没有找到最佳参数!")
            return

        # 构建标签编码映射
        encoder_mappings = {}
        for col, le in self.label_encoders.items():
            encoder_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

        code_template = f'''
# 优化后的Enhanced Ensemble模型 - {timestamp}
# 基于原始enhanced_natxis_classification.py的改进版本
# 最佳参数: {self.best_params}
# 标签编码映射 (classes -> integer): {encoder_mappings}

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

def optimized_enhanced_ensemble():
    """使用优化参数的Enhanced Ensemble模型"""

    # 加载数据
    try:
        features_df = pd.read_csv('../../feature_extraction/generated_features/all_features_with_categories.csv')
    except:
        features_df = pd.read_csv('../../feature_extraction/generated_features/all_features.csv')

    train_labels = pd.read_csv('../../original_data/train_acc.csv')
    data1 = features_df.merge(train_labels, on='account', how='inner')

    # 处理分类特征
    label_encoders = {encoder_mappings}
    feature_cols = [col for col in data1.columns if col not in ['account', 'flag']]

    for col in feature_cols:
        if col in label_encoders:
            mapping = label_encoders[col]
            data1[col] = data1[col].astype(str).map(mapping).fillna(-1).astype(int)
        else:
            # 非分类特征，尝试转为数值型
            if data1[col].dtype == 'object':
                data1[col] = pd.to_numeric(data1[col], errors='coerce').fillna(0)
    
    # 确保所有特征都是数值型
    for col in feature_cols:
        if data1[col].dtype == 'object':
            data1[col] = pd.to_numeric(data1[col], errors='coerce')
        data1[col] = data1[col].fillna(0).astype(np.float64)

    # 创建训练/验证分割 - 与优化过程保持一致
    train_data, val_data = train_test_split(
        data1, test_size=0.2, random_state=42, stratify=data1['flag']
    )
    
    print(f"训练集大小: {{len(train_data)}}, 验证集大小: {{len(val_data)}}")
    print(f"验证集标签分布: {{val_data['flag'].value_counts().to_dict()}}")

    # 最优参数
    BEST_PARAMS = {self.best_params}

    # 在训练集上进行集成训练，然后在验证集上预测
    np.random.seed(42)
    y_preds1 = []

    print(f"在训练集上训练 {{BEST_PARAMS['n_models']}} 个模型...")

    for i in range(BEST_PARAMS['n_models']):
        if (i + 1) % 20 == 0:
            print(f"  完成 {{i+1}}/{{BEST_PARAMS['n_models']}} 个模型")

        # 平衡采样 - 从训练集中采样
        positive_samples = len(train_data[train_data.flag == 1])
        negative_samples = len(train_data[train_data.flag == 0])
        
        train11 = train_data[train_data.flag == 1].sample(
            min(BEST_PARAMS['sample_size'], positive_samples),
            random_state=42 + i
        )
        train10 = train_data[train_data.flag == 0].sample(
            min(BEST_PARAMS['sample_size'], negative_samples),
            random_state=42 + i + 1000
        )

        train1 = pd.concat([train10, train11], axis='rows')
        x_train1 = train1[feature_cols].values
        y_train1 = train1.flag.values

        # 处理NaN值
        x_train1 = np.nan_to_num(x_train1)

        # RandomForest
        clf1 = RandomForestClassifier(
            n_estimators=BEST_PARAMS.get('n_estimators', 100),
            max_depth=BEST_PARAMS.get('max_depth', None),
            min_samples_split=BEST_PARAMS.get('min_samples_split', 2),
            min_samples_leaf=BEST_PARAMS.get('min_samples_leaf', 1),
            random_state=42 + i,
            n_jobs=-1
        )

        clf1.fit(x_train1, y_train1)

        # 在验证集上预测
        x_pred = val_data[feature_cols].values
        x_pred = np.nan_to_num(x_pred)
        pred = clf1.predict(x_pred)
        y_preds1.append(pred)

    # 集成投票
    y_pred1 = np.where(
        pd.DataFrame(y_preds1).sum(axis='rows').values > BEST_PARAMS['voting_threshold'],
        1, 0
    )

    # 评估（在验证集上）
    accuracy = accuracy_score(val_data['flag'].values, y_pred1)
    f1_binary = f1_score(val_data['flag'].values, y_pred1, average='binary')
    f1_weighted = f1_score(val_data['flag'].values, y_pred1, average='weighted')

    print(f"\\n=== 优化后模型在验证集上的性能 ===")
    print(f"Accuracy: {{accuracy:.4f}}")
    print(f"F1-Binary: {{f1_binary:.4f}}")
    print(f"F1-Weighted: {{f1_weighted:.4f}}")

    return y_pred1, accuracy, f1_binary

if __name__ == "__main__":
    predictions, acc, f1 = optimized_enhanced_ensemble()
    print(f"\\n最终结果: Accuracy={{acc:.4f}}, F1-Binary={{f1:.4f}}")
'''
        filename = f'optimized_enhanced_ensemble_{timestamp}.py'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(code_template)

        print(f"优化后的Enhanced Ensemble代码已保存: {filename}")

def main():
    """主函数"""
    optimizer = EnhancedEnsembleOptimizer()
    
    # 加载数据
    data = optimizer.load_enhanced_features()
    
    # 优化参数
    best_params, best_score = optimizer.optimize_enhanced_ensemble(
        data, 
        target_metric='f1_binary',
        max_iterations=13
    )
    
    # 训练最终模型
    train_predictions, train_metrics = optimizer.train_and_test_final_model(data)
    
    # 生成优化代码
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer.generate_optimized_code(timestamp)
    
    print(f"\n=== 优化完成 ===")
    print(f"相比原始Enhanced Ensemble的改进:")
    print(f"- 使用了相同的核心算法")
    print(f"- 处理了分类特征编码问题")
    print(f"- 优化了参数配置")
    print(f"- 提供了可复现的代码")

if __name__ == "__main__":
    main()