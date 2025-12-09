#!/usr/bin/env python3
"""
🔍 预测一致性深度分析器
分析高质量模型的预测一致性，找出：
1. 100%一致的账户（所有模型都同意）
2. 高度分歧的账户（模型意见不统一）
3. 关键分歧点的详细分析
4. 提供精准修改建议
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import json

def load_top_quality_predictions():
    """加载最高质量的预测文件"""
    high_score_dir = Path("/Users/mannormal/4011/Qi Zihan/v2/results/high_score_predictions")
    
    # 选择已验证的顶级模型（REAL F1 > 0.74）
    top_models = {
        # 单模型顶级
        'v3.2refined_fold1_REAL_F1_0.7549378200438918.csv': {
            'score': 0.7549, 'type': 'Single_V3.2', 'seed': 13
        },
        'v3.2refined_fold4_REAL_F1_0.7525325615050651_REAL_F1_0.7525325615050651.csv': {
            'score': 0.7525, 'type': 'Single_V3.2', 'seed': 13
        },
        
        # 融合成功案例  
        'AGGRESSIVE_AGGRESSIVE_VOTING_REAL_F1_0.7521489971346705.csv': {
            'score': 0.7521, 'type': 'Fusion_Aggressive', 'method': 'voting'
        },
        'FUSION_WEIGHTED_090_REAL_F1_0.7446102819237148.csv': {
            'score': 0.7446, 'type': 'Fusion_Conservative', 'method': 'weighted'
        },
        'AGGRESSIVE_DISTRIBUTION_MATCHING_REAL_F1_0.7435897435897436.csv': {
            'score': 0.7436, 'type': 'Fusion_Distribution', 'method': 'matching'
        },
        'AGGRESSIVE_TOP_MODEL_AGGRESSIVE_REAL_F1_0.7421052631578947.csv': {
            'score': 0.7421, 'type': 'Fusion_TopModel', 'method': 'aggressive'
        },
    }
    
    predictions = {}
    model_info = {}
    
    print("🔍 加载顶级质量预测文件（REAL F1 > 0.74）...")
    
    loaded_count = 0
    for filename, info in top_models.items():
        filepath = high_score_dir / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                if 'ID' in df.columns and 'Predict' in df.columns:
                    model_key = f"M{loaded_count+1}_{info['type']}"
                    predictions[model_key] = dict(zip(df['ID'], df['Predict']))
                    model_info[model_key] = {
                        'filename': filename,
                        'score': info['score'],
                        'bad_rate': df['Predict'].mean(),
                        'total_predictions': len(df),
                        **info
                    }
                    loaded_count += 1
                    print(f"✅ {model_key:<20} | Score: {info['score']:.4f} | Bad率: {df['Predict'].mean():.3f} | {info['type']}")
            except Exception as e:
                print(f"❌ {filename}: {e}")
    
    print(f"📊 成功加载 {len(predictions)} 个顶级模型")
    return predictions, model_info

def analyze_consensus_patterns(predictions, model_info):
    """分析预测一致性模式"""
    print("\n🎯 深度分析预测一致性模式...")
    
    # 获取所有账户ID
    all_ids = set()
    for pred_dict in predictions.values():
        all_ids.update(pred_dict.keys())
    all_ids = sorted(all_ids)
    
    model_names = list(predictions.keys())
    n_models = len(model_names)
    
    print(f"📊 分析范围: {len(all_ids)} 个账户 × {n_models} 个顶级模型")
    
    # 为每个账户分析预测模式
    account_analysis = []
    consensus_stats = {
        'unanimous_good': [],      # 所有模型都预测Good (100%一致)
        'unanimous_bad': [],       # 所有模型都预测Bad (100%一致)
        'near_unanimous_good': [], # 几乎一致Good (1个模型分歧)
        'near_unanimous_bad': [],  # 几乎一致Bad (1个模型分歧)
        'split_decisions': [],     # 严重分歧 (2-4个模型分歧)
        'disputed': []             # 高度争议 (接近50-50分割)
    }
    
    for account_id in all_ids:
        # 收集所有模型对此账户的预测
        account_votes = []
        missing_predictions = []
        
        for model in model_names:
            if account_id in predictions[model]:
                account_votes.append(predictions[model][account_id])
            else:
                missing_predictions.append(model)
        
        if not account_votes:  # 如果所有模型都没有此账户的预测
            continue
            
        # 统计投票结果
        positive_votes = sum(account_votes)  # Bad的票数
        total_votes = len(account_votes)
        negative_votes = total_votes - positive_votes  # Good的票数
        
        # 计算一致性
        consensus_ratio = max(positive_votes, negative_votes) / total_votes
        
        # 创建投票模式字符串
        vote_pattern = ''.join(str(v) for v in account_votes)
        
        account_data = {
            'account_id': account_id,
            'positive_votes': positive_votes,
            'negative_votes': negative_votes,
            'total_votes': total_votes,
            'consensus_ratio': consensus_ratio,
            'vote_pattern': vote_pattern,
            'missing_models': missing_predictions,
            'final_prediction': 1 if positive_votes > negative_votes else 0,
            'confidence': consensus_ratio
        }
        
        account_analysis.append(account_data)
        
        # 分类账户
        if positive_votes == 0:  # 所有模型都预测Good
            consensus_stats['unanimous_good'].append(account_data)
        elif positive_votes == total_votes:  # 所有模型都预测Bad
            consensus_stats['unanimous_bad'].append(account_data)
        elif positive_votes == 1 or negative_votes == 1:  # 只有1个模型不同意
            if positive_votes == 1:
                consensus_stats['near_unanimous_good'].append(account_data)
            else:
                consensus_stats['near_unanimous_bad'].append(account_data)
        elif abs(positive_votes - negative_votes) <= 1:  # 接近平分
            consensus_stats['disputed'].append(account_data)
        else:  # 其他分歧情况
            consensus_stats['split_decisions'].append(account_data)
    
    return account_analysis, consensus_stats, model_names

def detailed_consensus_analysis(consensus_stats, model_names, model_info):
    """详细的一致性分析"""
    print("\n📈 详细一致性分析报告:")
    print("=" * 80)
    
    total_accounts = sum(len(group) for group in consensus_stats.values())
    
    # 统计各类别
    categories = [
        ('unanimous_good', '100%一致预测Good', '🟢'),
        ('unanimous_bad', '100%一致预测Bad', '🔴'), 
        ('near_unanimous_good', '近乎一致Good(1票分歧)', '🟡'),
        ('near_unanimous_bad', '近乎一致Bad(1票分歧)', '🟠'),
        ('split_decisions', '明显分歧(2-3票分歧)', '🔵'),
        ('disputed', '高度争议(接近平分)', '🟣')
    ]
    
    for cat_key, cat_name, emoji in categories:
        count = len(consensus_stats[cat_key])
        percentage = count / total_accounts * 100 if total_accounts > 0 else 0
        print(f"{emoji} {cat_name:<25} | {count:4d} 个账户 ({percentage:5.1f}%)")
    
    print(f"\n📊 总计: {total_accounts} 个账户")
    
    # 分析100%一致的情况
    print(f"\n🎯 100%一致性分析:")
    unanimous_good = len(consensus_stats['unanimous_good'])
    unanimous_bad = len(consensus_stats['unanimous_bad'])
    total_unanimous = unanimous_good + unanimous_bad
    
    print(f"   🟢 100%一致Good: {unanimous_good:4d} 个 ({unanimous_good/total_accounts*100:5.1f}%)")
    print(f"   🔴 100%一致Bad:  {unanimous_bad:4d} 个 ({unanimous_bad/total_accounts*100:5.1f}%)")
    print(f"   ✅ 总一致率:     {total_unanimous:4d} 个 ({total_unanimous/total_accounts*100:5.1f}%)")
    
    return total_accounts, total_unanimous

def analyze_disagreement_patterns(consensus_stats, model_names, model_info):
    """分析分歧模式"""
    print(f"\n🔍 分歧模式深度分析:")
    print("=" * 80)
    
    # 分析哪些模型最容易产生分歧
    disagreement_matrix = defaultdict(int)
    
    # 统计分歧账户的投票模式
    disagreement_accounts = []
    disagreement_accounts.extend(consensus_stats['near_unanimous_good'])
    disagreement_accounts.extend(consensus_stats['near_unanimous_bad']) 
    disagreement_accounts.extend(consensus_stats['split_decisions'])
    disagreement_accounts.extend(consensus_stats['disputed'])
    
    print(f"📊 分歧账户总数: {len(disagreement_accounts)}")
    
    # 分析投票模式频率
    pattern_frequency = Counter()
    for account in disagreement_accounts:
        pattern_frequency[account['vote_pattern']] += 1
    
    print(f"\n🎭 最常见的分歧模式 (前10):")
    print("   模式    | 频次 | 含义")
    print("   -------|------|----------------------------------")
    
    for pattern, count in pattern_frequency.most_common(10):
        bad_votes = sum(int(x) for x in pattern)
        total_votes = len(pattern)
        good_votes = total_votes - bad_votes
        meaning = f"{good_votes}个Good票, {bad_votes}个Bad票"
        print(f"   {pattern:<7} | {count:4d} | {meaning}")
    
    return disagreement_accounts, pattern_frequency

def generate_modification_recommendations(consensus_stats, disagreement_accounts, pattern_frequency, model_info):
    """生成修改建议"""
    print(f"\n💡 精准修改建议:")
    print("=" * 80)
    
    recommendations = []
    
    # 建议1: 基于100%一致性
    unanimous_good = len(consensus_stats['unanimous_good'])  
    unanimous_bad = len(consensus_stats['unanimous_bad'])
    
    rec1 = {
        'strategy': 'CONSENSUS_100',
        'description': '基于100%一致性预测',
        'logic': '所有顶级模型一致的账户直接采纳',
        'expected_changes': f"确定预测 {unanimous_good + unanimous_bad} 个账户",
        'confidence': '极高'
    }
    recommendations.append(rec1)
    
    # 建议2: 处理近乎一致的情况 
    near_unanimous = len(consensus_stats['near_unanimous_good']) + len(consensus_stats['near_unanimous_bad'])
    
    rec2 = {
        'strategy': 'CONSENSUS_83',  # 5个模型中4个一致 = 83%
        'description': '处理83%一致性账户',
        'logic': '只有1个模型分歧时，跟随多数',
        'expected_changes': f"额外确定 {near_unanimous} 个账户",
        'confidence': '高'
    }
    recommendations.append(rec2)
    
    # 建议3: 基于最常见分歧模式的优化
    if pattern_frequency:
        most_common_pattern = pattern_frequency.most_common(1)[0]
        pattern, freq = most_common_pattern
        
        rec3 = {
            'strategy': 'PATTERN_OPTIMIZATION',
            'description': f'优化最常见分歧模式 {pattern}',
            'logic': f'对模式{pattern}的{freq}个账户使用特殊规则',
            'expected_changes': f"优化 {freq} 个争议账户",
            'confidence': '中等'
        }
        recommendations.append(rec3)
    
    # 建议4: 保守 vs 激进策略
    disputed_count = len(consensus_stats['disputed'])
    
    rec4 = {
        'strategy': 'DISPUTE_RESOLUTION',
        'description': '争议账户解决方案',
        'logic': f'{disputed_count}个高争议账户使用最高分模型决定',
        'expected_changes': f"解决 {disputed_count} 个争议案例", 
        'confidence': '中等'
    }
    recommendations.append(rec4)
    
    print("🎯 推荐的修改策略:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['strategy']}")
        print(f"   描述: {rec['description']}")
        print(f"   逻辑: {rec['logic']}")
        print(f"   预期: {rec['expected_changes']}")
        print(f"   置信度: {rec['confidence']}")
    
    return recommendations

def implement_consensus_strategies(predictions, consensus_stats, model_info):
    """实现基于一致性分析的策略"""
    print(f"\n🚀 实现一致性优化策略...")
    
    all_ids = set()
    for pred_dict in predictions.values():
        all_ids.update(pred_dict.keys())
    all_ids = sorted(all_ids)
    
    strategies = {}
    
    # 策略1: 100%一致性 + 83%一致性
    consensus_pred = {}
    
    # 100%一致的直接采纳
    for account in consensus_stats['unanimous_good']:
        consensus_pred[account['account_id']] = 0
    
    for account in consensus_stats['unanimous_bad']:
        consensus_pred[account['account_id']] = 1
    
    # 83%一致的跟随多数
    for account in consensus_stats['near_unanimous_good']:
        consensus_pred[account['account_id']] = 0
        
    for account in consensus_stats['near_unanimous_bad']:
        consensus_pred[account['account_id']] = 1
    
    # 其他账户使用最高分模型
    highest_score_model = max(model_info.items(), key=lambda x: x[1]['score'])[0]
    
    for account_id in all_ids:
        if account_id not in consensus_pred:
            if account_id in predictions[highest_score_model]:
                consensus_pred[account_id] = predictions[highest_score_model][account_id]
            else:
                consensus_pred[account_id] = 0  # 默认Good
    
    strategies['CONSENSUS_OPTIMIZED'] = consensus_pred
    
    # 策略2: 完全一致性（只有100%一致才决定）
    strict_consensus_pred = {}
    
    for account in consensus_stats['unanimous_good']:
        strict_consensus_pred[account['account_id']] = 0
    
    for account in consensus_stats['unanimous_bad']:
        strict_consensus_pred[account['account_id']] = 1
        
    # 其他所有争议账户都预测为Good（保守）
    for account_id in all_ids:
        if account_id not in strict_consensus_pred:
            strict_consensus_pred[account_id] = 0
    
    strategies['STRICT_CONSENSUS'] = strict_consensus_pred
    
    # 打印统计
    print(f"\n📊 一致性策略结果:")
    for name, pred in strategies.items():
        counts = Counter(pred.values())
        bad_rate = counts[1] / len(pred)
        print(f"   {name:<20} | Bad: {counts[1]:4d} ({bad_rate:6.1%}) | Good: {counts[0]:4d} ({1-bad_rate:6.1%})")
    
    return strategies

def save_consensus_analysis(consensus_stats, disagreement_accounts, recommendations, strategies):
    """保存一致性分析结果"""
    results_dir = Path("/Users/mannormal/4011/Qi Zihan/v2/results/high_score_predictions")
    
    print(f"\n💾 保存一致性分析结果...")
    
    # 保存策略预测文件
    for strategy_name, predictions in strategies.items():
        df = pd.DataFrame({
            'ID': sorted(predictions.keys()),
            'Predict': [predictions[id] for id in sorted(predictions.keys())]
        })
        
        filename = f"CONSENSUS_{strategy_name}.csv"
        filepath = results_dir / filename
        df.to_csv(filepath, index=False)
        
        pred_counts = Counter(df['Predict'])
        bad_rate = pred_counts[1] / len(df)
        print(f"✅ {filename}")
        print(f"   Bad (1):  {pred_counts[1]:4d} ({bad_rate:6.1%})")
        print(f"   Good (0): {pred_counts[0]:4d} ({1-bad_rate:6.1%})")
    
    # 保存详细分析报告
    report = {
        'analysis_summary': {
            'total_accounts': sum(len(group) for group in consensus_stats.values()),
            'unanimous_decisions': len(consensus_stats['unanimous_good']) + len(consensus_stats['unanimous_bad']),
            'disputed_accounts': len(disagreement_accounts)
        },
        'consensus_breakdown': {
            category: len(accounts) for category, accounts in consensus_stats.items()
        },
        'recommendations': recommendations,
        'generated_strategies': list(strategies.keys())
    }
    
    report_file = results_dir / "consensus_analysis_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 分析报告: consensus_analysis_report.json")

def main():
    print("🔍💎 预测一致性深度分析器 💎🔍")
    print("=" * 60)
    print("🎯 目标: 分析顶级模型的一致性，找出精准修改点")
    
    # 1. 加载顶级预测
    predictions, model_info = load_top_quality_predictions()
    
    if len(predictions) < 3:
        print("❌ 顶级预测文件不足")
        return
    
    # 2. 分析一致性模式
    account_analysis, consensus_stats, model_names = analyze_consensus_patterns(predictions, model_info)
    
    # 3. 详细分析
    total_accounts, total_unanimous = detailed_consensus_analysis(consensus_stats, model_names, model_info)
    
    # 4. 分歧模式分析
    disagreement_accounts, pattern_frequency = analyze_disagreement_patterns(consensus_stats, model_names, model_info)
    
    # 5. 生成修改建议
    recommendations = generate_modification_recommendations(consensus_stats, disagreement_accounts, pattern_frequency, model_info)
    
    # 6. 实现优化策略
    strategies = implement_consensus_strategies(predictions, consensus_stats, model_info)
    
    # 7. 保存结果
    save_consensus_analysis(consensus_stats, disagreement_accounts, recommendations, strategies)
    
    print(f"\n🎉 一致性分析完成！")
    print(f"\n🎯 关键发现:")
    print(f"   📊 总账户数: {total_accounts}")
    print(f"   ✅ 完全一致: {total_unanimous} ({total_unanimous/total_accounts*100:.1f}%)")
    print(f"   🤔 存在分歧: {len(disagreement_accounts)} ({len(disagreement_accounts)/total_accounts*100:.1f}%)")
    
    print(f"\n💡 推荐提交策略:")
    print(f"   1. CONSENSUS_CONSENSUS_OPTIMIZED.csv (平衡一致性和性能)")
    print(f"   2. CONSENSUS_STRICT_CONSENSUS.csv (极度保守，只信任100%一致)")
    
    print(f"\n🚀 这些策略基于顶级模型的深度一致性分析！")

if __name__ == "__main__":
    main()