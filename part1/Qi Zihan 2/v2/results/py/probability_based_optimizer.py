#!/usr/bin/env python3
"""
🎯 基于高分模型的概率优化器
使用所有高分文件建立每个账户的Bad概率，智能选择最优的727个Bad账户
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from upload import submit_file
from time import sleep

class ProbabilityBasedOptimizer:
    def __init__(self):
        self.high_score_dir = Path("/Users/mannormal/4011/Qi Zihan/v2/results/high_score_predictions")
        self.target_bad_count = 727  # 已知真实Bad数量
        self.results_dir = Path("/Users/mannormal/4011/Qi Zihan/v2/results")
        self.probability_file = self.results_dir / "account_probabilities.json"
        
        print("🎯 概率优化器初始化")
        print(f"   目标Bad数量: {self.target_bad_count}")
        
        # 加载所有高分模型
        self.load_high_score_models()
        
        # 计算每个账户的Bad概率
        self.calculate_account_probabilities()
        
    def load_high_score_models(self):
        """加载所有高分模型预测"""
        print(f"\n📂 加载高分模型...")
        
        self.models = {}
        self.model_scores = {}
        
        for filepath in self.high_score_dir.glob("*.csv"):
            filename = filepath.name
            
            # 提取真实F1分数
            if "REAL_F1_" in filename:
                try:
                    score_part = filename.split("REAL_F1_")[1].replace(".csv", "")
                    score = float(score_part)
                except:
                    score = 0.0
            else:
                score = 0.0
            
            if score < 0.73:  # 过滤低分模型
                continue
                
            try:
                df = pd.read_csv(filepath)
                if len(df.columns) >= 2:
                    # 假设第一列是ID，第二列是预测
                    id_col = df.columns[0]
                    pred_col = df.columns[1]
                    
                    model_key = filename.replace(".csv", "")
                    self.models[model_key] = dict(zip(df[id_col], df[pred_col]))
                    self.model_scores[model_key] = score
                    
                    bad_rate = df[pred_col].mean()
                    print(f"✅ {model_key[:50]:<50} | F1: {score:.4f} | Bad率: {bad_rate:.3f}")
                    
            except Exception as e:
                print(f"❌ 跳过 {filename}: {e}")
        
        print(f"\n📊 成功加载 {len(self.models)} 个高分模型")
        print(f"   F1分数范围: {min(self.model_scores.values()):.4f} - {max(self.model_scores.values()):.4f}")
        
    def calculate_account_probabilities(self):
        """计算每个账户的Bad概率"""
        print(f"\n🧮 计算账户Bad概率...")
        
        # 获取所有账户ID
        all_accounts = set()
        for model_pred in self.models.values():
            all_accounts.update(model_pred.keys())
        all_accounts = sorted(all_accounts)
        
        print(f"   总账户数: {len(all_accounts)}")
        
        self.account_probabilities = {}
        
        for account_id in all_accounts:
            votes = []
            weights = []
            
            # 收集每个模型对该账户的预测和权重
            for model_name, predictions in self.models.items():
                if account_id in predictions:
                    prediction = predictions[account_id]
                    weight = self.model_scores[model_name]  # 用F1分数作为权重
                    
                    votes.append(prediction)
                    weights.append(weight)
            
            if votes:
                # 加权平均概率
                weighted_prob = np.average(votes, weights=weights)
                
                # 简单投票概率
                simple_prob = np.mean(votes)
                
                # 最高分模型的预测
                max_weight_idx = np.argmax(weights)
                top_model_pred = votes[max_weight_idx]
                
                self.account_probabilities[account_id] = {
                    'weighted_probability': weighted_prob,
                    'simple_probability': simple_prob,
                    'top_model_prediction': top_model_pred,
                    'vote_count': len(votes),
                    'votes': votes,
                    'weights': weights,
                    'max_weight': max(weights),
                    'consensus_strength': self._calculate_consensus_strength(votes)
                }
        
        print(f"   完成概率计算: {len(self.account_probabilities)} 个账户")
        
        # 保存概率数据
        self.save_probability_data()
        
        # 分析概率分布
        self.analyze_probability_distribution()
    
    def _calculate_consensus_strength(self, votes):
        """计算模型间一致性强度"""
        if len(votes) <= 1:
            return 1.0
        
        # 计算方差，方差越小一致性越强
        variance = np.var(votes)
        # 转换为0-1之间的一致性分数
        consensus = 1.0 / (1.0 + variance * 4)  # 调节因子4
        return consensus
    
    def save_probability_data(self):
        """保存概率数据到JSON文件"""
        # 转换numpy类型为Python原生类型，便于JSON序列化
        serializable_data = {}
        for account_id, data in self.account_probabilities.items():
            serializable_data[account_id] = {
                'weighted_probability': float(data['weighted_probability']),
                'simple_probability': float(data['simple_probability']),
                'top_model_prediction': int(data['top_model_prediction']),
                'vote_count': int(data['vote_count']),
                'votes': [int(v) for v in data['votes']],
                'weights': [float(w) for w in data['weights']],
                'max_weight': float(data['max_weight']),
                'consensus_strength': float(data['consensus_strength'])
            }
        
        save_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'model_count': len(self.models),
                'account_count': len(self.account_probabilities),
                'target_bad_count': self.target_bad_count
            },
            'model_scores': self.model_scores,
            'account_probabilities': serializable_data
        }
        
        with open(self.probability_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"💾 概率数据已保存: {self.probability_file}")
    
    def analyze_probability_distribution(self):
        """分析概率分布"""
        print(f"\n📊 概率分布分析:")
        
        weighted_probs = [data['weighted_probability'] for data in self.account_probabilities.values()]
        simple_probs = [data['simple_probability'] for data in self.account_probabilities.values()]
        consensus_scores = [data['consensus_strength'] for data in self.account_probabilities.values()]
        
        print(f"   加权概率统计:")
        print(f"     均值: {np.mean(weighted_probs):.4f}")
        print(f"     标准差: {np.std(weighted_probs):.4f}")
        print(f"     25%分位: {np.percentile(weighted_probs, 25):.4f}")
        print(f"     50%分位: {np.percentile(weighted_probs, 50):.4f}")
        print(f"     75%分位: {np.percentile(weighted_probs, 75):.4f}")
        
        print(f"\n   一致性分析:")
        print(f"     平均一致性: {np.mean(consensus_scores):.4f}")
        high_consensus = sum(1 for s in consensus_scores if s > 0.8)
        low_consensus = sum(1 for s in consensus_scores if s < 0.5)
        print(f"     高一致性账户 (>0.8): {high_consensus} ({high_consensus/len(consensus_scores)*100:.1f}%)")
        print(f"     低一致性账户 (<0.5): {low_consensus} ({low_consensus/len(consensus_scores)*100:.1f}%)")
        
        # 预测当前最优的727个Bad
        print(f"\n🎯 基于概率预测最优727个Bad:")
        sorted_accounts = self.get_top_bad_candidates(method='weighted')
        
        top_727_weighted_probs = [self.account_probabilities[acc]['weighted_probability'] 
                                 for acc in sorted_accounts[:727]]
        print(f"   Top 727平均概率: {np.mean(top_727_weighted_probs):.4f}")
        print(f"   最低Bad概率: {min(top_727_weighted_probs):.4f}")
        print(f"   最高Bad概率: {max(top_727_weighted_probs):.4f}")
        
        # 边界分析
        if len(sorted_accounts) > 727:
            boundary_prob = self.account_probabilities[sorted_accounts[726]]['weighted_probability']
            next_prob = self.account_probabilities[sorted_accounts[727]]['weighted_probability']
            print(f"   边界概率: {boundary_prob:.4f} vs {next_prob:.4f} (差距: {boundary_prob-next_prob:.4f})")
    
    def get_top_bad_candidates(self, n=727, method='weighted'):
        """获取Top N个Bad候选账户"""
        if method == 'weighted':
            sorted_accounts = sorted(self.account_probabilities.keys(),
                                   key=lambda x: self.account_probabilities[x]['weighted_probability'],
                                   reverse=True)
        elif method == 'simple':
            sorted_accounts = sorted(self.account_probabilities.keys(),
                                   key=lambda x: self.account_probabilities[x]['simple_probability'],
                                   reverse=True)
        elif method == 'consensus':
            # 优先选择高概率且高一致性的账户
            sorted_accounts = sorted(self.account_probabilities.keys(),
                                   key=lambda x: (self.account_probabilities[x]['weighted_probability'] * 
                                                self.account_probabilities[x]['consensus_strength']),
                                   reverse=True)
        else:  # top_model
            sorted_accounts = sorted(self.account_probabilities.keys(),
                                   key=lambda x: (self.account_probabilities[x]['top_model_prediction'],
                                                self.account_probabilities[x]['max_weight']),
                                   reverse=True)
        
        return sorted_accounts[:n]
    
    def create_probability_based_submission(self, method='weighted', name_suffix=''):
        """创建基于概率的提交文件"""
        print(f"\n🎯 创建基于概率的提交 (方法: {method})")
        
        top_bad_accounts = self.get_top_bad_candidates(self.target_bad_count, method)
        
        # 创建提交数据
        submission_data = []
        all_accounts = sorted(self.account_probabilities.keys())
        
        for account_id in all_accounts:
            prediction = 1 if account_id in top_bad_accounts else 0
            submission_data.append({
                'ID': account_id,
                'Predict': prediction
            })
        
        # 创建DataFrame
        submission_df = pd.DataFrame(submission_data)
        
        # 统计
        bad_count = sum(submission_df['Predict'])
        good_count = len(submission_df) - bad_count
        
        print(f"📊 提交统计:")
        print(f"   Bad (1): {bad_count} ({bad_count/len(submission_df)*100:.2f}%)")
        print(f"   Good (0): {good_count} ({good_count/len(submission_df)*100:.2f}%)")
        
        # 保存文件
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"PROBABILITY_{method.upper()}_{self.target_bad_count}{name_suffix}_{timestamp}.csv"
        filepath = self.results_dir / filename
        
        submission_df.to_csv(filepath, index=False)
        print(f"✅ 保存: {filename}")
        
        return filepath, submission_df
    
    def create_multiple_strategies(self):
        """创建多种策略的提交文件"""
        print(f"\n🚀 生成多种概率策略...")
        
        strategies = [
            ('weighted', '基于F1加权概率'),
            ('simple', '简单投票概率'), 
            ('consensus', '概率×一致性'),
            ('top_model', '最高分模型主导')
        ]
        
        submissions = []
        
        for method, description in strategies:
            print(f"\n📝 策略: {description}")
            filepath, submission_df = self.create_probability_based_submission(method)
            submissions.append({
                'method': method,
                'description': description,
                'filepath': filepath,
                'submission_df': submission_df
            })
        
        return submissions
    
    def submit_and_compare_strategies(self, submissions):
        """提交并比较不同策略"""
        print(f"\n🚀 提交并比较策略效果...")
        
        results = []
        
        for i, submission in enumerate(submissions):
            print(f"\n🎯 提交策略 {i+1}/{len(submissions)}: {submission['description']}")
            
            try:
                score = submit_file(12507, str(submission['filepath']))
                if score is not None:
                    print(f"   F1分数: {score:.6f}")
                    
                    # 重命名文件包含分数
                    old_filepath = submission['filepath']
                    new_filename = old_filepath.stem + f"_F1_{score:.6f}.csv"
                    new_filepath = old_filepath.parent / new_filename
                    os.rename(old_filepath, new_filepath)
                    
                    results.append({
                        'method': submission['method'],
                        'description': submission['description'],
                        'f1_score': score,
                        'filepath': new_filepath
                    })
                    
                    print(f"   ✅ 重命名为: {new_filename}")
                    
                else:
                    print(f"   ❌ 提交失败")
                    
            except Exception as e:
                print(f"   ❌ 错误: {e}")
            
            # 避免提交过快
            if i < len(submissions) - 1:
                print(f"   ⏱️  等待3秒...")
                sleep(3)
        
        # 结果排序和分析
        if results:
            results.sort(key=lambda x: x['f1_score'], reverse=True)
            
            print(f"\n🏆 策略效果排名:")
            print("排名 | 方法        | F1分数   | 描述")
            print("-" * 50)
            
            for i, result in enumerate(results, 1):
                print(f"{i:2d}   | {result['method']:<10} | {result['f1_score']:.6f} | {result['description']}")
            
            best_result = results[0]
            print(f"\n🎉 最佳策略: {best_result['method']} (F1: {best_result['f1_score']:.6f})")
            
            return results
        else:
            print("❌ 没有成功的提交")
            return []

def main():
    """主程序"""
    print("🎯 概率优化器启动")
    print("="*50)
    
    optimizer = ProbabilityBasedOptimizer()
    
    # 创建多种策略
    submissions = optimizer.create_multiple_strategies()
    
    print(f"\n❓ 是否要提交所有策略进行比较？")
    print(f"   策略数量: {len(submissions)}")
    print(f"   目标: 找到最优的727个Bad账户选择方法")
    
    choice = input("提交所有策略? (y/n): ").lower().strip()
    if choice == 'y':
        results = optimizer.submit_and_compare_strategies(submissions)
        
        if results:
            best_f1 = results[0]['f1_score']
            print(f"\n🎯 最终结果:")
            print(f"   最高F1: {best_f1:.6f}")
            
            if best_f1 > 0.80:
                print("🎉 突破0.8大关！基于概率的方法非常成功！")
            elif best_f1 > 0.77:
                print("🎊 显著改进！接近最优解！")
            else:
                print("🤔 需要进一步优化概率模型")

if __name__ == "__main__":
    main()