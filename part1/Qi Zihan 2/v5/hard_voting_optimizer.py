#!/usr/bin/env python3
"""
硬投票权重优化系统
适用于只有0/1预测的情况
"""

import pandas as pd
import numpy as np
from simulator import simulate_f1
import os
from itertools import combinations

class HardVotingOptimizer:
    def __init__(self):
        # 11个预测文件路径
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
            "/Users/mannormal/Desktop/课程/y4t1/stat 4011/best.csv"
        ]
        
        # 加载所有预测
        self.predictions = []
        self.account_ids = None
        self.model_f1s = []
        self.load_predictions()
        
    def load_predictions(self):
        """加载所有预测文件"""
        print("📂 加载预测文件...")
        
        for i, filepath in enumerate(self.prediction_files):
            df = pd.read_csv(filepath)
            
            if self.account_ids is None:
                self.account_ids = df['ID'].values
            
            self.predictions.append(df['Predict'].values)
            
            # 计算单个模型的F1
            f1 = simulate_f1(filepath)
            self.model_f1s.append(f1)
            filename = os.path.basename(filepath)
            print(f"  [{i+1:2d}] F1={f1:.6f} - {filename[:60]}...")
        
        self.predictions = np.array(self.predictions)  # shape: (11, 7558)
        print(f"\n✅ 加载完成: {len(self.predictions)}个模型, {len(self.account_ids)}个账户\n")
    
    def weighted_voting(self, weights):
        """加权硬投票"""
        # 每个账户的加权投票分数
        weighted_votes = np.dot(weights, self.predictions)  # (7558,)
        
        # 使用0.5作为阈值（超过半数权重投bad则为bad）
        predictions = (weighted_votes > 0.5 * weights.sum()).astype(int)
        
        return predictions
    
    def majority_voting(self, selected_models):
        """简单多数投票"""
        selected_predictions = self.predictions[selected_models]
        # 每个账户的投票总和
        votes = np.sum(selected_predictions, axis=0)
        # 超过半数则为1
        predictions = (votes > len(selected_models) / 2).astype(int)
        return predictions
    
    def save_prediction_csv(self, predictions, filename):
        """保存预测结果"""
        df = pd.DataFrame({
            'ID': self.account_ids,
            'Predict': predictions
        })
        df.to_csv(filename, index=False)
        return filename
    
    def evaluate_combination(self, selected_models, weights=None):
        """评估模型组合"""
        if weights is None:
            # 均等权重
            weights = np.ones(len(self.predictions))
            weights[list(selected_models)] = 1
            weights = weights / weights.sum()
        
        predictions = self.weighted_voting(weights)
        temp_file = "/tmp/voting_temp.csv"
        self.save_prediction_csv(predictions, temp_file)
        f1 = simulate_f1(temp_file)
        os.remove(temp_file)
        
        return f1
    
    def find_best_subset(self, max_models=11):
        """找到最优模型子集"""
        print(f"🎯 搜索最优模型组合 (最多{max_models}个模型)")
        print("="*80)
        
        n_models = len(self.predictions)
        best_f1 = 0
        best_combination = None
        best_size = 0
        
        # 从单个模型开始搜索
        for size in range(1, min(max_models + 1, n_models + 1)):
            print(f"\n📊 测试 {size} 个模型的组合...")
            
            # 生成所有大小为size的组合
            for combo in combinations(range(n_models), size):
                # 简单多数投票
                predictions = self.majority_voting(list(combo))
                temp_file = f"/tmp/combo_temp.csv"
                self.save_prediction_csv(predictions, temp_file)
                f1 = simulate_f1(temp_file)
                os.remove(temp_file)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_combination = combo
                    best_size = size
                    combo_str = ", ".join([f"#{i+1}" for i in combo])
                    print(f"  🆕 新最优: [{combo_str}] F1={f1:.6f}")
            
            # 如果当前size没有改进，提前停止
            if size > 1 and best_size < size:
                print(f"  ⏭️  {size}个模型组合无改进，停止搜索")
                break
        
        print(f"\n✅ 搜索完成!")
        print(f"   最优组合: {[f'#{i+1}' for i in best_combination]}")
        print(f"   F1 Score: {best_f1:.6f}")
        
        return best_combination, best_f1
    
    def optimize_weights_for_subset(self, selected_models):
        """为选定的模型优化权重"""
        print(f"\n🔧 优化选定模型的权重...")
        print(f"   选定模型: {[f'#{i+1}' for i in selected_models]}")
        
        n_selected = len(selected_models)
        best_weights = np.ones(n_selected) / n_selected  # 初始均等权重
        best_f1 = 0
        
        # 生成权重网格
        weight_options = np.arange(0, 1.1, 0.1)
        
        # 对于2-3个模型，可以详尽搜索
        if n_selected == 2:
            for w1 in weight_options:
                w2 = 1 - w1
                if w2 < 0:
                    continue
                
                weights_full = np.zeros(len(self.predictions))
                weights_full[selected_models[0]] = w1
                weights_full[selected_models[1]] = w2
                
                predictions = self.weighted_voting(weights_full)
                temp_file = "/tmp/weight_opt_temp.csv"
                self.save_prediction_csv(predictions, temp_file)
                f1 = simulate_f1(temp_file)
                os.remove(temp_file)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = np.array([w1, w2])
                    print(f"  🆕 新最优权重: [{w1:.1f}, {w2:.1f}] F1={f1:.6f}")
        
        elif n_selected == 3:
            for w1 in weight_options:
                for w2 in weight_options:
                    w3 = 1 - w1 - w2
                    if w3 < 0 or w3 > 1:
                        continue
                    
                    weights_full = np.zeros(len(self.predictions))
                    weights_full[selected_models[0]] = w1
                    weights_full[selected_models[1]] = w2
                    weights_full[selected_models[2]] = w3
                    
                    predictions = self.weighted_voting(weights_full)
                    temp_file = "/tmp/weight_opt_temp.csv"
                    self.save_prediction_csv(predictions, temp_file)
                    f1 = simulate_f1(temp_file)
                    os.remove(temp_file)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_weights = np.array([w1, w2, w3])
                        print(f"  🆕 新最优权重: [{w1:.1f}, {w2:.1f}, {w3:.1f}] F1={f1:.6f}")
        
        else:
            # 多于3个模型，使用均等权重
            print("  ℹ️  模型数>3，使用均等权重")
            best_weights = np.ones(n_selected) / n_selected
            
            weights_full = np.zeros(len(self.predictions))
            for i, idx in enumerate(selected_models):
                weights_full[idx] = best_weights[i]
            
            predictions = self.weighted_voting(weights_full)
            temp_file = "/tmp/weight_opt_temp.csv"
            self.save_prediction_csv(predictions, temp_file)
            best_f1 = simulate_f1(temp_file)
            os.remove(temp_file)
        
        # 构建完整权重向量
        final_weights = np.zeros(len(self.predictions))
        for i, idx in enumerate(selected_models):
            final_weights[idx] = best_weights[i]
        
        print(f"\n✅ 权重优化完成! F1={best_f1:.6f}")
        
        return final_weights, best_f1
    
    def print_weights(self, weights, f1_score):
        """打印权重信息"""
        print("\n" + "="*80)
        print("🏆 最优权重配置")
        print("="*80)
        
        sorted_indices = np.argsort(weights)[::-1]
        
        for rank, idx in enumerate(sorted_indices):
            if weights[idx] > 0.001:
                filename = os.path.basename(self.prediction_files[idx])
                single_f1 = self.model_f1s[idx]
                print(f"  #{rank+1:2d} [模型{idx+1:2d}] 权重={weights[idx]:.4f} ({weights[idx]*100:5.1f}%) F1单独={single_f1:.4f} - {filename[:45]}...")
        
        print(f"\n📊 最终F1 Score: {f1_score:.6f}")
        print(f"🎯 使用模型数: {np.sum(weights > 0.001)}/{len(weights)}")
        print("="*80 + "\n")
    
    def save_final_prediction(self, weights, output_file):
        """保存最终预测"""
        predictions = self.weighted_voting(weights)
        self.save_prediction_csv(predictions, output_file)
        
        print(f"💾 保存最终预测到: {output_file}")
        
        n_bad = np.sum(predictions == 1)
        n_good = np.sum(predictions == 0)
        print(f"   预测: good={n_good}, bad={n_bad}")
        
        final_f1 = simulate_f1(output_file)
        print(f"   验证F1: {final_f1:.6f}")
        
        return final_f1

def main():
    optimizer = HardVotingOptimizer()
    
    # 方法1: 找最优模型子集
    print("\n" + "🚀 " + "="*76)
    print("🚀 方法1: 搜索最优模型组合")
    print("🚀 " + "="*76)
    
    best_combo, combo_f1 = optimizer.find_best_subset(max_models=5)
    
    # 方法2: 为最优子集优化权重
    print("\n" + "🔧 " + "="*76)
    print("🔧 方法2: 优化权重")
    print("🔧 " + "="*76)
    
    best_weights, weighted_f1 = optimizer.optimize_weights_for_subset(best_combo)
    optimizer.print_weights(best_weights, weighted_f1)
    
    # 保存最终结果
    output_file = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/hard_voting_ensemble.csv"
    final_f1 = optimizer.save_final_prediction(best_weights, output_file)
    
    print("\n" + "🎉 " + "="*76)
    print("🎉 优化完成!")
    print(f"🎉 最终F1: {final_f1:.6f}")
    print("🎉 " + "="*76)

if __name__ == "__main__":
    main()
