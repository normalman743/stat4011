#!/usr/bin/env python3
"""
软投票权重优化系统
使用贪心逐步优化算法找到12个预测文件的最优权重
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from simulator import simulate_f1
import os

class SoftVotingOptimizer:
    def __init__(self):
        # 12个预测文件路径
        self.prediction_files = [
            "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/AGGRESSIVE_AGGRESSIVE_VOTING_REAL_F1_0.7521489971346705.csv",
            "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/GRADIENT_TUNE_10PCT_REAL_F1_0.7611336032388665.csv",
            "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/GRADIENT_TUNE_7PCT_REAL_F1_0.7531847133757962.csv",
            "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/GRADIENT_TUNE_8PCT_REAL_F1_0.7528174305033809.csv",
            "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/GRADIENT_TUNE_9PCT_REAL_F1_0.7533759772565743.csv",
            "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/result.csv",
            "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/submit.csv",
            "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold1_bad_f1_0.7778_good_0.9765_bad_0.7778_macro_0.8771_weighted_0.9570_seed_13_REAL_F1_0.7549378200438918.csv",
            "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold1_bad_f1_0.8083_good_0.9803_bad_0.8083_macro_0.8943_weighted_0.9634_seed_13_REAL_F1_0.7628549501151188.csv",
            "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold4_bad_f1_0.8250_good_0.9814_bad_0.8250_macro_0.9032_weighted_0.9661_seed_13_REAL_F1_0.7525325615050651_REAL_F1_0.7525325615050651.csv",
            "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold5_bad_f1_0.8401_good_0.9838_bad_0.8401_macro_0.9119_weighted_0.9697_seed_13_REAL_F1_0.7579273008507347.csv",
        ]
        
        # 加载所有预测
        self.predictions = []
        self.account_ids = None
        self.load_predictions()
        
        # 最优权重
        self.best_weights = None
        self.best_f1 = 0.0
        
    def load_predictions(self):
        """加载所有预测文件"""
        print("📂 加载预测文件...")
        
        for i, filepath in enumerate(self.prediction_files):
            if not os.path.exists(filepath):
                print(f"⚠️  文件不存在: {filepath}")
                continue
                
            df = pd.read_csv(filepath)
            
            # 第一个文件，保存账户ID
            if self.account_ids is None:
                self.account_ids = df['ID'].values
            
            # 保存预测值
            self.predictions.append(df['Predict'].values)
            
            # 计算单个模型的F1
            f1 = simulate_f1(filepath)
            filename = os.path.basename(filepath)
            print(f"  [{i+1:2d}] F1={f1:.6f} - {filename[:60]}...")
        
        self.predictions = np.array(self.predictions)  # shape: (12, 7558)
        print(f"\n✅ 加载完成: {len(self.predictions)}个模型, {len(self.account_ids)}个账户\n")
    
    def weighted_predict(self, weights, threshold=0.5):
        """根据权重进行软投票预测"""
        # 加权平均分数
        weighted_scores = np.dot(weights, self.predictions)  # (7558,)
        
        # 根据阈值决策
        predictions = (weighted_scores >= threshold).astype(int)
        
        return predictions
    
    def save_prediction_csv(self, predictions, filename):
        """保存预测结果到临时文件"""
        df = pd.DataFrame({
            'ID': self.account_ids,
            'Predict': predictions
        })
        df.to_csv(filename, index=False)
        return filename
    
    def objective_function(self, weights, threshold=0.5):
        """优化目标函数: 最大化F1 score (最小化负F1)"""
        # 归一化权重
        weights = weights / weights.sum()
        
        # 软投票预测
        predictions = self.weighted_predict(weights, threshold)
        
        # 保存临时文件
        temp_file = "/tmp/soft_voting_temp.csv"
        self.save_prediction_csv(predictions, temp_file)
        
        # 计算F1
        f1_score = simulate_f1(temp_file)
        
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # 返回负F1 (因为minimize函数是最小化)
        return -f1_score
    
    def optimize_weights_scipy(self, initial_weights=None, threshold=0.5):
        """使用scipy优化权重"""
        n_models = len(self.predictions)
        
        # 初始权重 (均匀分布)
        if initial_weights is None:
            initial_weights = np.ones(n_models) / n_models
        
        # 约束: 权重和为1, 权重非负
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_models)]
        
        print(f"🔧 开始优化权重 (阈值={threshold})...")
        
        result = minimize(
            self.objective_function,
            initial_weights,
            args=(threshold,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'disp': True}
        )
        
        # 归一化最优权重
        optimal_weights = result.x / result.x.sum()
        optimal_f1 = -result.fun
        
        return optimal_weights, optimal_f1
    
    def grid_search_optimization(self, threshold=0.5):
        """网格搜索优化 - 简单且稳定"""
        n_models = len(self.predictions)
        
        print(f"🎯 网格搜索权重优化 (阈值={threshold})")
        print("="*80)
        
        # 评估单个模型
        print("\n📊 评估单个模型...")
        model_f1s = []
        for i in range(n_models):
            f1 = simulate_f1(self.prediction_files[i])
            model_f1s.append(f1)
            print(f"  模型 {i+1:2d}: F1={f1:.6f}")
        
        # 找出最好的3个模型
        top3_indices = np.argsort(model_f1s)[-3:][::-1]
        print(f"\n✨ Top 3 模型: #{top3_indices[0]+1}, #{top3_indices[1]+1}, #{top3_indices[2]+1}")
        
        # 网格搜索权重组合
        print(f"\n🔍 网格搜索权重组合...")
        
        best_weights = np.zeros(n_models)
        best_f1 = 0
        
        # 生成权重网格 (步长0.1)
        weight_grid = np.arange(0, 1.1, 0.1)
        total_combinations = 0
        
        for w1 in weight_grid:
            for w2 in weight_grid:
                for w3 in weight_grid:
                    # 归一化
                    total = w1 + w2 + w3
                    if total == 0:
                        continue
                    
                    w1_norm = w1 / total
                    w2_norm = w2 / total
                    w3_norm = w3 / total
                    
                    # 构建权重向量
                    weights = np.zeros(n_models)
                    weights[top3_indices[0]] = w1_norm
                    weights[top3_indices[1]] = w2_norm
                    weights[top3_indices[2]] = w3_norm
                    
                    # 预测
                    predictions = self.weighted_predict(weights, threshold)
                    temp_file = f"/tmp/grid_search_temp.csv"
                    self.save_prediction_csv(predictions, temp_file)
                    f1 = simulate_f1(temp_file)
                    os.remove(temp_file)
                    
                    total_combinations += 1
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_weights = weights.copy()
                        print(f"  🆕 新最优: w=[{w1_norm:.2f}, {w2_norm:.2f}, {w3_norm:.2f}], F1={f1:.6f}")
        
        print(f"\n✅ 搜索完成! 总组合数={total_combinations}, 最优F1={best_f1:.6f}")
        
        return best_weights, best_f1
    
    def greedy_stepwise_optimization(self, threshold=0.5):
        """贪心逐步优化算法 - 修复版"""
        n_models = len(self.predictions)
        
        print(f"🎯 贪心逐步优化 (阈值={threshold})")
        print("="*80)
        
        # Step 1: 找到最好的单个模型
        best_single_f1 = 0
        best_single_idx = 0
        
        print("\n📊 Step 1: 评估单个模型...")
        for i in range(n_models):
            f1 = simulate_f1(self.prediction_files[i])
            if f1 > best_single_f1:
                best_single_f1 = f1
                best_single_idx = i
            print(f"  模型 {i+1:2d}: F1={f1:.6f}")
        
        print(f"\n✨ 最佳单模型: #{best_single_idx+1}, F1={best_single_f1:.6f}")
        
        # 初始化权重
        selected_models = [best_single_idx]
        current_weights = np.zeros(n_models)
        current_weights[best_single_idx] = 1.0
        current_f1 = best_single_f1
        
        # Step 2: 逐步添加其他模型
        print(f"\n📈 Step 2: 逐步添加模型...")
        
        remaining_models = [i for i in range(n_models) if i != best_single_idx]
        
        for step, candidate_idx in enumerate(remaining_models):
            print(f"\n  尝试添加模型 #{candidate_idx+1}...")
            
            best_new_f1 = current_f1
            best_candidate_weight = 0
            
            # 网格搜索候选模型的权重
            for candidate_w in np.arange(0.05, 0.96, 0.05):
                w = np.zeros(n_models)
                w[candidate_idx] = candidate_w
                
                # 已选模型平均分配剩余权重
                remaining_w = 1 - candidate_w
                for idx in selected_models:
                    w[idx] = remaining_w / len(selected_models)
                
                # 计算F1
                predictions = self.weighted_predict(w, threshold)
                temp_file = f"/tmp/greedy_temp_{step}_{candidate_w:.2f}.csv"
                self.save_prediction_csv(predictions, temp_file)
                f1 = simulate_f1(temp_file)
                os.remove(temp_file)
                
                if f1 > best_new_f1:
                    best_new_f1 = f1
                    best_candidate_weight = candidate_w
            
            # 如果F1提升，则添加这个模型
            if best_new_f1 > current_f1:
                selected_models.append(candidate_idx)
                
                # 更新权重
                current_weights = np.zeros(n_models)
                current_weights[candidate_idx] = best_candidate_weight
                remaining_w = 1 - best_candidate_weight
                for idx in selected_models[:-1]:
                    current_weights[idx] = remaining_w / len(selected_models[:-1])
                
                current_f1 = best_new_f1
                print(f"    ✅ 添加成功! 权重={best_candidate_weight:.4f}, F1={current_f1:.6f}")
            else:
                print(f"    ❌ F1未提升 ({best_new_f1:.6f} <= {current_f1:.6f}), 跳过")
        
        print(f"\n✅ 优化完成! 最终F1={current_f1:.6f}")
        
        return current_weights, current_f1
    
    def optimize_threshold(self, weights, thresholds=None):
        """优化决策阈值"""
        if thresholds is None:
            thresholds = np.linspace(0.3, 0.7, 21)
        
        print(f"\n🎯 优化决策阈值...")
        
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in thresholds:
            predictions = self.weighted_predict(weights, threshold)
            temp_file = f"/tmp/threshold_temp.csv"
            self.save_prediction_csv(predictions, temp_file)
            f1 = simulate_f1(temp_file)
            os.remove(temp_file)
            
            print(f"  阈值={threshold:.2f}, F1={f1:.6f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"\n✨ 最优阈值={best_threshold:.2f}, F1={best_f1:.6f}")
        
        return best_threshold, best_f1
    
    def print_weights(self, weights, f1_score):
        """打印权重信息"""
        print("\n" + "="*80)
        print("🏆 最优权重配置")
        print("="*80)
        
        # 按权重排序
        sorted_indices = np.argsort(weights)[::-1]
        
        for rank, idx in enumerate(sorted_indices):
            if weights[idx] > 0.001:  # 只显示权重>0.1%的模型
                filename = os.path.basename(self.prediction_files[idx])
                print(f"  #{rank+1:2d} [模型{idx+1:2d}] 权重={weights[idx]:.6f} ({weights[idx]*100:5.2f}%) - {filename[:50]}...")
        
        print(f"\n📊 最终F1 Score: {f1_score:.6f}")
        print(f"🎯 非零权重模型数: {np.sum(weights > 0.001)}/{len(weights)}")
        print("="*80 + "\n")
    
    def save_final_prediction(self, weights, threshold, output_file):
        """保存最终预测结果"""
        predictions = self.weighted_predict(weights, threshold)
        self.save_prediction_csv(predictions, output_file)
        
        print(f"💾 保存最终预测到: {output_file}")
        
        # 统计
        n_bad = np.sum(predictions == 1)
        n_good = np.sum(predictions == 0)
        print(f"   预测: good={n_good}, bad={n_bad}")
        
        # 验证F1
        final_f1 = simulate_f1(output_file)
        print(f"   验证F1: {final_f1:.6f}")

def main():
    optimizer = SoftVotingOptimizer()
    
    # 方法1: 网格搜索 (简单稳定)
    print("\n" + "🚀 " + "="*76)
    print("🚀 方法1: 网格搜索优化")
    print("🚀 " + "="*76)
    
    weights_grid, f1_grid = optimizer.grid_search_optimization(threshold=0.5)
    optimizer.print_weights(weights_grid, f1_grid)
    
    # 方法2: 贪心优化 (备选)
    print("\n" + "� " + "="*76)
    print("� 方法2: 贪心逐步优化")
    print("� " + "="*76)
    
    weights_greedy, f1_greedy = optimizer.greedy_stepwise_optimization(threshold=0.5)
    optimizer.print_weights(weights_greedy, f1_greedy)
    
    # 选择更好的方法
    if f1_grid >= f1_greedy:
        print(f"\n✨ 使用网格搜索结果 (F1={f1_grid:.6f})")
        best_weights = weights_grid
        best_f1 = f1_grid
    else:
        print(f"\n✨ 使用贪心优化结果 (F1={f1_greedy:.6f})")
        best_weights = weights_greedy
        best_f1 = f1_greedy
    
    # 优化阈值
    best_threshold, threshold_f1 = optimizer.optimize_threshold(best_weights)
    
    # 保存最终结果
    output_file = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/soft_voting_ensemble.csv"
    optimizer.save_final_prediction(best_weights, best_threshold, output_file)
    
    print("\n" + "🎉 " + "="*76)
    print("🎉 优化完成!")
    print("🎉 " + "="*76)

if __name__ == "__main__":
    main()
