import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import itertools
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DynamicEnsembleOptimizer:
    def __init__(self, data_path='../../feature_extraction/generated_features/'):
        """初始化动态集成优化器"""
        self.data_path = data_path
        self.best_params = None
        self.best_score = 0
        self.optimization_history = []
        
    def load_data(self):
        """加载数据"""
        print("=== 加载数据 ===")
        # 加载特征数据
        features_df = pd.read_csv(f'{self.data_path}all_features_with_categories.csv')
        
        # 加载训练标签
        train_labels = pd.read_csv('../../original_data/train_acc.csv')
        
        # 合并数据
        data = features_df.merge(train_labels, on='account', how='inner')
        
        print(f"数据形状: {data.shape}")
        print(f"标签分布: {data['flag'].value_counts().to_dict()}")
        
        return data
    
    def create_ensemble_predictions(self, data, n_models=100, sample_size=532, 
                                  voting_threshold=93, random_state=42,
                                  max_depth=None, n_estimators=100, 
                                  min_samples_split=2, min_samples_leaf=1):
        """创建集成预测"""
        np.random.seed(random_state)
        
        # 分离特征和标签
        feature_cols = [col for col in data.columns if col not in ['account', 'flag']]
        X = data[feature_cols].values
        y = data['flag'].values
        
        # 分离正负类数据
        positive_data = data[data.flag == 1]
        negative_data = data[data.flag == 0]
        
        predictions = []
        
        for i in range(n_models):
            # 平衡采样
            pos_sample = positive_data.sample(min(sample_size, len(positive_data)), 
                                            random_state=random_state + i)
            neg_sample = negative_data.sample(min(sample_size, len(negative_data)), 
                                            random_state=random_state + i + 1000)
            
            train_data = pd.concat([pos_sample, neg_sample])
            
            X_train = train_data[feature_cols].values
            y_train = train_data['flag'].values
            
            # 创建随机森林
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state + i,
                n_jobs=-1
            )
            
            rf.fit(X_train, y_train)
            pred = rf.predict(X)
            predictions.append(pred)
        
        # 集成投票
        predictions_array = np.array(predictions)
        vote_counts = np.sum(predictions_array, axis=0)
        final_predictions = np.where(vote_counts > voting_threshold, 1, 0)
        
        return final_predictions
    
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
    
    def optimize_parameters(self, data, target_metric='f1_binary', max_iterations=50):
        """动态优化参数"""
        print(f"\n=== 开始参数优化 (目标: {target_metric}) ===")
        
        # 创建验证集
        train_data, val_data = train_test_split(data, test_size=0.2, 
                                               random_state=42, stratify=data['flag'])
        
        print(f"训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")
        
        # 参数搜索空间
        param_grid = {
            'n_models': [50, 80, 100, 120, 150],
            'sample_size': [300, 400, 500, 532, 600, 700],
            'voting_threshold': [85, 88, 90, 93, 95, 97, 99],
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [None, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # 智能搜索策略
        best_score = 0
        best_params = None
        iteration = 0
        
        # 第一阶段: 粗搜索关键参数
        print("\n--- 第一阶段: 粗搜索 ---")
        for voting_threshold in param_grid['voting_threshold']:
            for n_models in [50, 100, 150]:
                for sample_size in [400, 532, 600]:
                    iteration += 1
                    if iteration > max_iterations // 2:
                        break
                    
                    params = {
                        'n_models': n_models,
                        'sample_size': sample_size,
                        'voting_threshold': voting_threshold,
                        'n_estimators': 100,
                        'max_depth': None,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'random_state': 42
                    }
                    
                    start_time = time.time()
                    predictions = self.create_ensemble_predictions(train_data, **params)
                    metrics = self.evaluate_model(train_data['flag'].values, predictions)
                    
                    # 在验证集上测试
                    val_predictions = self.create_ensemble_predictions(val_data, **params)
                    val_metrics = self.evaluate_model(val_data['flag'].values, val_predictions)
                    
                    score = val_metrics[target_metric]
                    runtime = time.time() - start_time
                    
                    print(f"迭代 {iteration:2d} | 参数: {params['n_models']:3d}模型, "
                          f"阈值{params['voting_threshold']:2d}, 采样{params['sample_size']:3d} | "
                          f"验证{target_metric}: {score:.4f} | 用时: {runtime:.1f}s")
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        print(f"  ★ 新最佳! {target_metric}: {score:.4f}")
                    
                    self.optimization_history.append({
                        'iteration': iteration,
                        'params': params.copy(),
                        'train_metrics': metrics,
                        'val_metrics': val_metrics,
                        'runtime': runtime
                    })
        
        # 第二阶段: 基于最佳参数精细调优
        print(f"\n--- 第二阶段: 精细调优 (基于最佳参数) ---")
        if best_params:
            print(f"最佳基础参数: {best_params}")
            
            # 精细调优RandomForest参数
            for n_estimators in [150, 200, 250]:
                for max_depth in [None, 15, 20]:
                    for min_samples_split in [2, 5]:
                        iteration += 1
                        if iteration > max_iterations:
                            break
                        
                        params = best_params.copy()
                        params.update({
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split
                        })
                        
                        start_time = time.time()
                        val_predictions = self.create_ensemble_predictions(val_data, **params)
                        val_metrics = self.evaluate_model(val_data['flag'].values, val_predictions)
                        
                        score = val_metrics[target_metric]
                        runtime = time.time() - start_time
                        
                        print(f"迭代 {iteration:2d} | RF参数: est={n_estimators}, "
                              f"depth={max_depth}, split={min_samples_split} | "
                              f"验证{target_metric}: {score:.4f} | 用时: {runtime:.1f}s")
                        
                        if score > best_score:
                            best_score = score
                            best_params = params.copy()
                            print(f"  ★ 新最佳! {target_metric}: {score:.4f}")
        
        self.best_params = best_params
        self.best_score = best_score
        
        print(f"\n=== 优化完成 ===")
        print(f"最佳 {target_metric}: {best_score:.4f}")
        print(f"最佳参数: {best_params}")
        
        return best_params, best_score
    
    def train_final_model(self, data, params):
        """训练最终的可复现模型"""
        print(f"\n=== 训练最终模型 ===")
        print(f"使用参数: {params}")
        
        # 使用全部数据训练
        start_time = time.time()
        final_predictions = self.create_ensemble_predictions(data, **params)
        final_metrics = self.evaluate_model(data['flag'].values, final_predictions)
        training_time = time.time() - start_time
        
        print(f"最终模型性能:")
        for metric, value in final_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print(f"训练用时: {training_time:.1f}秒")
        
        return final_predictions, final_metrics
    
    def save_results(self, params, metrics, predictions=None):
        """保存优化结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存最佳参数
        params_df = pd.DataFrame([params])
        params_df.to_csv(f'optimized_params_{timestamp}.csv', index=False)
        
        # 保存性能指标
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f'optimized_metrics_{timestamp}.csv', index=False)
        
        # 保存优化历史
        history_df = pd.DataFrame(self.optimization_history)
        history_df.to_csv(f'optimization_history_{timestamp}.csv', index=False)
        
        # 保存预测结果
        if predictions is not None:
            pred_df = pd.DataFrame({'prediction': predictions})
            pred_df.to_csv(f'optimized_predictions_{timestamp}.csv', index=False)
        
        print(f"\n结果已保存 (时间戳: {timestamp})")
        
        # 生成可复现代码
        self.generate_reproducible_code(params, timestamp)
    
    def generate_reproducible_code(self, params, timestamp):
        """生成可复现的模型代码"""
        code_template = f'''
# 自动生成的最优模型代码 - {timestamp}
# 最佳参数: {params}

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def load_and_predict():
    """使用优化后的参数进行预测"""
    # 加载数据
    features_df = pd.read_csv('../../feature_extraction/generated_features/all_features_with_categories.csv')
    train_labels = pd.read_csv('../../original_data/train_acc.csv')
    data = features_df.merge(train_labels, on='account', how='inner')
    
    # 最优参数
    BEST_PARAMS = {params}
    
    # 特征列
    feature_cols = [col for col in data.columns if col not in ['account', 'flag']]
    X = data[feature_cols].values
    y = data['flag'].values
    
    # 分离正负类
    positive_data = data[data.flag == 1]
    negative_data = data[data.flag == 0]
    
    # 集成预测
    np.random.seed(BEST_PARAMS['random_state'])
    predictions = []
    
    for i in range(BEST_PARAMS['n_models']):
        # 平衡采样
        pos_sample = positive_data.sample(
            min(BEST_PARAMS['sample_size'], len(positive_data)), 
            random_state=BEST_PARAMS['random_state'] + i
        )
        neg_sample = negative_data.sample(
            min(BEST_PARAMS['sample_size'], len(negative_data)), 
            random_state=BEST_PARAMS['random_state'] + i + 1000
        )
        
        train_data = pd.concat([pos_sample, neg_sample])
        X_train = train_data[feature_cols].values
        y_train = train_data['flag'].values
        
        # 训练随机森林
        rf = RandomForestClassifier(
            n_estimators=BEST_PARAMS['n_estimators'],
            max_depth=BEST_PARAMS['max_depth'],
            min_samples_split=BEST_PARAMS['min_samples_split'],
            min_samples_leaf=BEST_PARAMS['min_samples_leaf'],
            random_state=BEST_PARAMS['random_state'] + i,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        pred = rf.predict(X)
        predictions.append(pred)
    
    # 集成投票
    predictions_array = np.array(predictions)
    vote_counts = np.sum(predictions_array, axis=0)
    final_predictions = np.where(vote_counts > BEST_PARAMS['voting_threshold'], 1, 0)
    
    # 评估
    accuracy = accuracy_score(y, final_predictions)
    f1_binary = f1_score(y, final_predictions, average='binary')
    
    print(f"最优模型性能:")
    print(f"Accuracy: {{accuracy:.4f}}")
    print(f"F1-Binary: {{f1_binary:.4f}}")
    
    return final_predictions, accuracy, f1_binary

if __name__ == "__main__":
    predictions, acc, f1 = load_and_predict()
'''
        
        with open(f'optimized_model_{timestamp}.py', 'w', encoding='utf-8') as f:
            f.write(code_template)
        
        print(f"可复现代码已保存: optimized_model_{timestamp}.py")

def main():
    """主函数"""
    optimizer = DynamicEnsembleOptimizer()
    
    # 加载数据
    data = optimizer.load_data()
    
    # 优化参数
    best_params, best_score = optimizer.optimize_parameters(
        data, 
        target_metric='f1_binary',  # 可选: 'accuracy', 'f1_weighted', 'f1_macro'
        max_iterations=30
    )
    
    # 训练最终模型
    final_predictions, final_metrics = optimizer.train_final_model(data, best_params)
    
    # 保存结果
    optimizer.save_results(best_params, final_metrics, final_predictions)
    
    print(f"\n=== 最终结果总结 ===")
    print(f"最佳F1-Binary: {final_metrics['f1_binary']:.4f}")
    print(f"最佳Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"最佳参数已保存，可复现代码已生成")

if __name__ == "__main__":
    main()