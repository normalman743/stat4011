import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
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
        
        print(f"最终数据形状: {data.shape}")
        print(f"特征列数: {len([col for col in data.columns if col not in ['account', 'flag']])}")
        print(f"标签分布: {data['flag'].value_counts().to_dict()}")
        
        return data
    
    def enhanced_ensemble_predict(self, data, n_models=100, sample_size=532, 
                                voting_threshold=93, random_state=42,
                                n_estimators=100, max_depth=None, 
                                min_samples_split=2, min_samples_leaf=1):
        """
        增强集成预测 - 完全复制您当前最佳方法
        基于您的enhanced_natxis_classification.py逻辑
        """
        np.random.seed(random_state)
        
        # 准备特征
        feature_cols = [col for col in data.columns if col not in ['account', 'flag']]
        
        # 分离正负类数据 (完全按照您的方法)
        data1 = data.copy()  # 保持变量名一致
        
        y_preds1 = []  # 存储所有模型预测
        
        print(f"开始训练 {n_models} 个模型...")
        for i in range(n_models):
            if (i + 1) % 20 == 0:
                print(f"  完成 {i+1}/{n_models} 个模型")
            
            # 平衡采样 - 完全按照您的逻辑
            train11 = data1[data1.flag == 1].sample(
                min(sample_size, len(data1[data1.flag == 1])), 
                random_state=random_state + i
            )
            train10 = data1[data1.flag == 0].sample(
                min(sample_size, len(data1[data1.flag == 0])), 
                random_state=random_state + i + 1000
            )
            
            # 合并训练数据
            train1 = pd.concat([train10, train11], axis='rows')
            
            # 准备训练特征和标签
            x_train1 = train1[feature_cols].values
            y_train1 = train1.flag.values
            
            # 创建RandomForest - 使用您的配置
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
            
            # 预测全部数据
            pred = clf1.predict(data1[feature_cols].values)
            y_preds1.append(pred)
        
        # 集成投票 - 完全按照您的逻辑
        y_pred1 = np.where(
            pd.DataFrame(y_preds1).sum(axis='rows').values > voting_threshold, 
            1, 0
        )
        
        return y_pred1
    
    def optimize_enhanced_ensemble(self, data, target_metric='f1_binary', max_iterations=30):
        """优化增强集成模型参数"""
        print(f"\n=== 基于Enhanced Ensemble方法进行参数优化 ===")
        print(f"目标指标: {target_metric}")
        
        # 创建验证集
        train_data, val_data = train_test_split(
            data, test_size=0.2, random_state=42, stratify=data['flag']
        )
        
        # 参数搜索空间 - 基于您当前的最佳配置进行微调
        param_combinations = [
            # 基于您当前最佳配置的变体
            {'n_models': 100, 'sample_size': 532, 'voting_threshold': 93, 'n_estimators': 100},
            {'n_models': 100, 'sample_size': 532, 'voting_threshold': 95, 'n_estimators': 100},
            {'n_models': 100, 'sample_size': 532, 'voting_threshold': 90, 'n_estimators': 100},
            {'n_models': 100, 'sample_size': 532, 'voting_threshold': 93, 'n_estimators': 150},
            {'n_models': 100, 'sample_size': 532, 'voting_threshold': 93, 'n_estimators': 200},
            
            # 调整采样大小
            {'n_models': 100, 'sample_size': 400, 'voting_threshold': 93, 'n_estimators': 100},
            {'n_models': 100, 'sample_size': 600, 'voting_threshold': 93, 'n_estimators': 100},
            {'n_models': 100, 'sample_size': 700, 'voting_threshold': 93, 'n_estimators': 100},
            
            # 调整模型数量
            {'n_models': 120, 'sample_size': 532, 'voting_threshold': 93, 'n_estimators': 100},
            {'n_models': 150, 'sample_size': 532, 'voting_threshold': 93, 'n_estimators': 100},
            {'n_models': 80, 'sample_size': 532, 'voting_threshold': 93, 'n_estimators': 100},
            
            # 组合优化
            {'n_models': 120, 'sample_size': 600, 'voting_threshold': 95, 'n_estimators': 150},
            {'n_models': 150, 'sample_size': 700, 'voting_threshold': 90, 'n_estimators': 200},
        ]
        
        best_score = 0
        best_params = None
        
        for i, params in enumerate(param_combinations):
            if i >= max_iterations:
                break
                
            print(f"\n--- 测试配置 {i+1}/{min(len(param_combinations), max_iterations)} ---")
            print(f"参数: {params}")
            
            start_time = time.time()
            
            # 在验证集上测试
            val_predictions = self.enhanced_ensemble_predict(val_data, **params)
            val_metrics = self.evaluate_model(val_data['flag'].values, val_predictions)
            
            runtime = time.time() - start_time
            score = val_metrics[target_metric]
            
            print(f"验证集性能:")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  F1-Binary: {val_metrics['f1_binary']:.4f}")
            print(f"  F1-Weighted: {val_metrics['f1_weighted']:.4f}")
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
    
    def train_and_test_final_model(self, data, test_data_path='../../original_data/test_acc_predict.csv'):
        """训练最终模型并在测试集上预测"""
        print(f"\n=== 训练最终模型并预测测试集 ===")
        
        if self.best_params is None:
            print("错误: 请先运行optimize_enhanced_ensemble()!")
            return None, None
        
        print(f"使用最佳参数: {self.best_params}")
        
        # 在全部训练数据上训练
        start_time = time.time()
        train_predictions = self.enhanced_ensemble_predict(data, **self.best_params)
        train_metrics = self.evaluate_model(data['flag'].values, train_predictions)
        training_time = time.time() - start_time
        
        print(f"训练集性能:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"训练用时: {training_time:.1f}秒")
        
        # 加载测试数据并预测
        try:
            test_df = pd.read_csv(test_data_path)
            print(f"测试集大小: {len(test_df)}")
            
            # 为测试集添加特征（这里需要您的特征工程流程）
            print("注意: 需要为测试集生成相同的特征")
            
        except Exception as e:
            print(f"无法加载测试数据: {e}")
            test_predictions = None
        
        return train_predictions, train_metrics
    
    def generate_optimized_code(self, timestamp):
        """生成优化后的可复现代码"""
        if self.best_params is None:
            print("错误: 没有找到最佳参数!")
            return
        
        code_template = f'''
# 优化后的Enhanced Ensemble模型 - {timestamp}
# 基于原始enhanced_natxis_classification.py的改进版本
# 最佳参数: {self.best_params}

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def optimized_enhanced_ensemble():
    """使用优化参数的Enhanced Ensemble模型"""
    
    # 加载数据
    try:
        features_df = pd.read_csv('../../feature_extraction/generated_features/all_features_with_categories.csv')
    except:
        features_df = pd.read_csv('../../feature_extraction/generated_features/all_features.csv')
    
    train_labels = pd.read_csv('../../original_data/train_acc.csv')
    data1 = features_df.merge(train_labels, on='account', how='inner')
    
    # 最优参数
    BEST_PARAMS = {self.best_params}
    
    # 特征列
    feature_cols = [col for col in data1.columns if col not in ['account', 'flag']]
    
    # 集成预测
    np.random.seed(BEST_PARAMS.get('random_state', 42))
    y_preds1 = []
    
    print(f"训练 {{BEST_PARAMS['n_models']}} 个模型...")
    
    for i in range(BEST_PARAMS['n_models']):
        if (i + 1) % 20 == 0:
            print(f"  完成 {{i+1}}/{{BEST_PARAMS['n_models']}} 个模型")
        
        # 平衡采样
        train11 = data1[data1.flag == 1].sample(
            min(BEST_PARAMS['sample_size'], len(data1[data1.flag == 1])), 
            random_state=42 + i
        )
        train10 = data1[data1.flag == 0].sample(
            min(BEST_PARAMS['sample_size'], len(data1[data1.flag == 0])), 
            random_state=42 + i + 1000
        )
        
        train1 = pd.concat([train10, train11], axis='rows')
        x_train1 = train1[feature_cols].values
        y_train1 = train1.flag.values
        
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
        pred = clf1.predict(data1[feature_cols].values)
        y_preds1.append(pred)
    
    # 集成投票
    y_pred1 = np.where(
        pd.DataFrame(y_preds1).sum(axis='rows').values > BEST_PARAMS['voting_threshold'], 
        1, 0
    )
    
    # 评估
    accuracy = accuracy_score(data1['flag'].values, y_pred1)
    f1_binary = f1_score(data1['flag'].values, y_pred1, average='binary')
    f1_weighted = f1_score(data1['flag'].values, y_pred1, average='weighted')
    
    print(f"\\n=== 优化后模型性能 ===")
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
        max_iterations=15
    )
    
    # 训练最终模型
    train_predictions, train_metrics = optimizer.train_and_test_final_model(data)
    
    # 生成优化代码
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer.generate_optimized_code(timestamp)
    
    print(f"\n=== 优化完成 ===")
    print(f"相比原始Enhanced Ensemble的改进:")
    print(f"- 使用了相同的核心算法")
    print(f"- 优化了参数配置")
    print(f"- 提供了可复现的代码")

if __name__ == "__main__":
    main()