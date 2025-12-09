#!/usr/bin/env python3
"""
🎯 梯度调节器 - 基于分歧账户的渐进优化
朝1:9方向渐进调节，生成多个不同Bad率的版本供测试
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import json

def load_consensus_analysis():
    """加载一致性分析结果"""
    high_score_dir = Path("/Users/mannormal/4011/Qi Zihan/v2/results/high_score_predictions")
    
    predictions = {}
    model_info = {}
    
    print("🔍 加载所有高分模型...")
    
    for filepath in high_score_dir.glob("*.csv"):
        filename = filepath.name
        try:
            df = pd.read_csv(filepath)
            if 'ID' in df.columns and 'Predict' in df.columns:
                model_key = filename.rsplit('.', 1)[0]
                predictions[model_key] = dict(zip(df['ID'], df['Predict']))
                model_info[model_key] = {
                    'bad_rate': df['Predict'].mean(),
                    'filename': filename
                }
                print(f"✅ {model_key:<25} | Bad率: {df['Predict'].mean():.3f}")
        except Exception as e:
            print(f"❌ {filename}: {e}")
    
    print(f"📊 加载了 {len(predictions)} 个模型")
    return predictions, model_info

def analyze_disagreement_accounts(predictions):
    """分析分歧账户"""
    print("\n🔍 重新分析分歧账户...")
    
    all_ids = set()
    for pred_dict in predictions.values():
        all_ids.update(pred_dict.keys())
    all_ids = sorted(all_ids)
    
    model_names = list(predictions.keys())
    n_models = len(model_names)
    
    # 分类账户
    unanimous_good = []      # 所有模型预测Good
    unanimous_bad = []       # 所有模型预测Bad
    disagreement_accounts = [] # 分歧账户
    
    for account_id in all_ids:
        votes = []
        for model in model_names:
            if account_id in predictions[model]:
                votes.append(predictions[model][account_id])
        
        if not votes:
            continue
            
        positive_votes = sum(votes)  # Bad票数
        total_votes = len(votes)
        
        if positive_votes == 0:
            unanimous_good.append(account_id)
        elif positive_votes == total_votes:
            unanimous_bad.append(account_id)
        else:
            # 分歧账户
            vote_pattern = ''.join(str(v) for v in votes)
            bad_probability = positive_votes / total_votes
            
            disagreement_accounts.append({
                'account_id': account_id,
                'votes': votes,
                'pattern': vote_pattern,
                'bad_votes': positive_votes,
                'total_votes': total_votes,
                'bad_probability': bad_probability,
                'current_prediction': 1 if bad_probability >= 0.5 else 0
            })
    
    print(f"📊 账户分类:")
    print(f"   🟢 一致Good: {len(unanimous_good):4d} ({len(unanimous_good)/len(all_ids)*100:5.1f}%)")
    print(f"   🔴 一致Bad:  {len(unanimous_bad):4d} ({len(unanimous_bad)/len(all_ids)*100:5.1f}%)")
    print(f"   🤔 分歧账户: {len(disagreement_accounts):4d} ({len(disagreement_accounts)/len(all_ids)*100:5.1f}%)")
    
    # 按Bad概率排序分歧账户
    disagreement_accounts.sort(key=lambda x: x['bad_probability'], reverse=True)
    
    print(f"\n🎭 分歧账户模式分布:")
    pattern_count = Counter([acc['pattern'] for acc in disagreement_accounts])
    for pattern, count in pattern_count.most_common():
        bad_votes = sum(int(x) for x in pattern)
        print(f"   {pattern}: {count:3d}个 (Bad概率: {bad_votes/len(pattern):.2f})")
    
    return unanimous_good, unanimous_bad, disagreement_accounts, all_ids

def generate_gradient_tuning_strategies(unanimous_good, unanimous_bad, disagreement_accounts, all_ids):
    """生成梯度调节策略"""
    print(f"\n🎯 生成梯度调节策略...")
    
    current_bad_count = len(unanimous_bad)
    total_accounts = len(all_ids)
    current_bad_rate = current_bad_count / total_accounts
    
    print(f"📊 当前基础状态:")
    print(f"   确定Bad: {current_bad_count} ({current_bad_rate:.1%})")
    print(f"   分歧空间: {len(disagreement_accounts)} 个账户可调节")
    
    # 目标Bad率梯度
    target_rates = [0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
    strategies = {}
    
    for rate in target_rates:
        target_bad_count = int(total_accounts * rate)
        additional_bad_needed = target_bad_count - current_bad_count
        
        print(f"\n🎲 目标Bad率 {rate:.1%}:")
        print(f"   目标Bad总数: {target_bad_count}")
        print(f"   需要额外Bad: {additional_bad_needed}")
        
        # 生成策略
        strategy_name = f"TUNE_{rate:.0%}".replace('%', 'PCT')
        
        # 基础预测：一致的账户保持不变
        strategy_pred = {}
        for account_id in unanimous_good:
            strategy_pred[account_id] = 0
        for account_id in unanimous_bad:
            strategy_pred[account_id] = 1
        
        # 处理分歧账户
        if additional_bad_needed <= 0:
            # 如果不需要额外Bad，所有分歧账户预测为Good
            for acc in disagreement_accounts:
                strategy_pred[acc['account_id']] = 0
        elif additional_bad_needed >= len(disagreement_accounts):
            # 如果需要的Bad超过分歧账户数，全部预测为Bad
            for acc in disagreement_accounts:
                strategy_pred[acc['account_id']] = 1
        else:
            # 按Bad概率选择前N个作为Bad
            for i, acc in enumerate(disagreement_accounts):
                if i < additional_bad_needed:
                    strategy_pred[acc['account_id']] = 1
                    print(f"     选择: {acc['account_id']} (模式:{acc['pattern']}, 概率:{acc['bad_probability']:.2f})")
                else:
                    strategy_pred[acc['account_id']] = 0
        
        strategies[strategy_name] = strategy_pred
        
        # 验证结果
        actual_bad = sum(strategy_pred.values())
        actual_rate = actual_bad / len(strategy_pred)
        print(f"   实际Bad: {actual_bad} ({actual_rate:.1%})")
    
    return strategies

def save_gradient_strategies(strategies):
    """保存梯度策略"""
    results_dir = Path("/Users/mannormal/4011/Qi Zihan/v2/results")
    
    print(f"\n💾 保存梯度调节策略到 {results_dir}...")
    
    strategy_summary = []
    
    for strategy_name, predictions in strategies.items():
        # 保存预测文件
        df = pd.DataFrame({
            'ID': sorted(predictions.keys()),
            'Predict': [predictions[id] for id in sorted(predictions.keys())]
        })
        
        filename = f"GRADIENT_{strategy_name}.csv"
        filepath = results_dir / filename
        df.to_csv(filepath, index=False)
        
        # 统计信息
        pred_counts = Counter(df['Predict'])
        bad_rate = pred_counts[1] / len(df)
        
        strategy_info = {
            'strategy': strategy_name,
            'filename': filename,
            'bad_count': pred_counts[1],
            'good_count': pred_counts[0],
            'bad_rate': bad_rate,
            'total_accounts': len(df)
        }
        strategy_summary.append(strategy_info)
        
        print(f"✅ {filename}")
        print(f"   Bad (1):  {pred_counts[1]:4d} ({bad_rate:6.1%}) 🎯")
        print(f"   Good (0): {pred_counts[0]:4d} ({1-bad_rate:6.1%})")
    
    # 保存策略摘要
    summary_file = results_dir / "gradient_tuning_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(strategy_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📋 策略摘要: gradient_tuning_summary.json")
    
    return strategy_summary

def main():
    print("🎯🔧 梯度调节器 - 渐进优化Bad率 🔧🎯")
    print("=" * 60)
    print("🎯 目标: 朝1:9方向渐进调节，生成多个测试版本")
    
    # 1. 加载顶级模型
    predictions, model_info = load_consensus_analysis()
    
    if len(predictions) < 3:
        print("❌ 顶级模型不足")
        return
    
    # 2. 分析分歧账户
    unanimous_good, unanimous_bad, disagreement_accounts, all_ids = analyze_disagreement_accounts(predictions)
    
    # 3. 生成梯度策略
    strategies = generate_gradient_tuning_strategies(unanimous_good, unanimous_bad, disagreement_accounts, all_ids)
    
    # 4. 保存策略
    strategy_summary = save_gradient_strategies(strategies)
    
    print(f"\n🎉 梯度调节完成！")
    print(f"\n📊 生成了 {len(strategies)} 个不同Bad率的版本:")
    
    for info in strategy_summary:
        print(f"   {info['strategy']:<15} | Bad率: {info['bad_rate']:6.1%} | 文件: {info['filename']}")
    
    print(f"\n🎯 建议测试顺序:")
    print(f"   1. GRADIENT_TUNE_7PCT.csv  (保守调节)")
    print(f"   2. GRADIENT_TUNE_8PCT.csv  (温和调节)")
    print(f"   3. GRADIENT_TUNE_9PCT.csv  (标准调节)")
    print(f"   4. GRADIENT_TUNE_10PCT.csv (目标1:9)")
    print(f"   5. GRADIENT_TUNE_11PCT.csv (激进调节)")
    print(f"   6. GRADIENT_TUNE_12PCT.csv (极限测试)")
    
    print(f"\n🚀 通过这些渐进测试找出最优Bad率!")
    print(f"💡 所有文件保存在: {Path('/Users/mannormal/4011/Qi Zihan/v2/results')}")

if __name__ == "__main__":
    main()