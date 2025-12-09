#!/usr/bin/env python3
"""
🔍 预测分析器 - 深度分析高分预测文件的异同
找出最优融合策略以达到0.9分数
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import seaborn as sns

def analyze_predictions():
    """分析所有高分预测文件"""
    high_score_dir = Path("/Users/mannormal/4011/Qi Zihan/v2/results/high_score_predictions")
    
    print("🔍 加载所有高分预测文件...")
    
    # 加载所有预测
    predictions = {}
    file_info = {}
    
    for csv_file in high_score_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if 'ID' in df.columns and 'Predict' in df.columns:
                model_name = csv_file.stem[:50]  # 简化文件名
                predictions[model_name] = df.set_index('ID')['Predict'].to_dict()
                
                # 提取分数信息
                score = extract_score_from_filename(csv_file.name)
                file_info[model_name] = {
                    'filename': csv_file.name,
                    'score': score,
                    'predict_1_count': df['Predict'].sum(),
                    'predict_1_ratio': df['Predict'].mean(),
                    'total_accounts': len(df)
                }
                print(f"✅ {model_name} | Score: {score:.4f} | Bad率: {df['Predict'].mean():.3f}")
        except Exception as e:
            print(f"❌ 跳过 {csv_file.name}: {e}")
    
    print(f"\n📊 成功加载 {len(predictions)} 个预测文件")
    return predictions, file_info

def extract_score_from_filename(filename):
    """从文件名提取分数"""
    import re
    # REAL_F1分数
    if 'REAL_F1_' in filename:
        match = re.search(r'REAL_F1_([0-9.]+)', filename)
        if match:
            return float(match.group(1))
    
    # 其他分数估算
    if 'best_cv_f1_score_0.9121' in filename:
        return 0.733
    elif 'best_cv_f1_score_0.9314' in filename:
        return 0.713
    elif 'macro_f1_0.9735' in filename:
        return 0.703
    elif 'macro_f1_0.9590' in filename:
        return 0.720
    elif 'weighted_f1_0.9746' in filename:
        return 0.735
    elif 'weighted_f1_0.9733' in filename:
        return 0.736
    elif 'weighted_f1_0.9680' in filename:
        return 0.741
    elif 'badf1_0691' in filename:
        return 0.731
    elif 'voting_rf' in filename:
        return 0.736
    elif 'v2ultra_resnet' in filename:
        return 0.743
    else:
        return 0.700  # 默认估计

def analyze_agreement_patterns(predictions, file_info):
    """分析预测一致性模式"""
    print("\n🎯 分析预测一致性模式...")
    
    # 获取所有账户ID
    all_ids = set()
    for pred_dict in predictions.values():
        all_ids.update(pred_dict.keys())
    all_ids = sorted(all_ids)
    
    # 创建预测矩阵
    model_names = list(predictions.keys())
    pred_matrix = np.zeros((len(all_ids), len(model_names)), dtype=int)
    
    for j, model in enumerate(model_names):
        for i, account_id in enumerate(all_ids):
            pred_matrix[i, j] = predictions[model].get(account_id, 0)
    
    # 分析每个账户的预测模式
    account_analysis = []
    for i, account_id in enumerate(all_ids):
        account_preds = pred_matrix[i, :]
        positive_votes = account_preds.sum()
        total_votes = len(account_preds)
        agreement_ratio = positive_votes / total_votes
        
        account_analysis.append({
            'account_id': account_id,
            'positive_votes': positive_votes,
            'total_votes': total_votes,
            'agreement_ratio': agreement_ratio,
            'prediction_pattern': ''.join(map(str, account_preds))
        })
    
    # 按一致性分组
    unanimous_good = [a for a in account_analysis if a['positive_votes'] == 0]
    unanimous_bad = [a for a in account_analysis if a['positive_votes'] == a['total_votes']]
    high_consensus_bad = [a for a in account_analysis if a['agreement_ratio'] >= 0.8 and a['positive_votes'] > 0]
    high_consensus_good = [a for a in account_analysis if a['agreement_ratio'] <= 0.2 and a['positive_votes'] < a['total_votes']]
    disputed = [a for a in account_analysis if 0.3 <= a['agreement_ratio'] <= 0.7]
    
    print(f"\n📈 预测一致性统计:")
    print(f"   一致预测Good (0票): {len(unanimous_good):4d} ({len(unanimous_good)/len(all_ids)*100:5.1f}%)")
    print(f"   一致预测Bad (满票): {len(unanimous_bad):4d} ({len(unanimous_bad)/len(all_ids)*100:5.1f}%)")
    print(f"   高度一致Good (≤20%): {len(high_consensus_good):4d} ({len(high_consensus_good)/len(all_ids)*100:5.1f}%)")
    print(f"   高度一致Bad (≥80%): {len(high_consensus_bad):4d} ({len(high_consensus_bad)/len(all_ids)*100:5.1f}%)")
    print(f"   争议账户 (30%-70%): {len(disputed):4d} ({len(disputed)/len(all_ids)*100:5.1f}%)")
    
    return {
        'all_ids': all_ids,
        'pred_matrix': pred_matrix,
        'model_names': model_names,
        'account_analysis': account_analysis,
        'unanimous_good': unanimous_good,
        'unanimous_bad': unanimous_bad,
        'high_consensus_good': high_consensus_good,
        'high_consensus_bad': high_consensus_bad,
        'disputed': disputed
    }

def analyze_model_similarities(predictions, file_info):
    """分析模型间相似性"""
    print("\n🔗 分析模型间相似性...")
    
    model_names = list(predictions.keys())
    n_models = len(model_names)
    
    # 计算模型间相似性矩阵
    similarity_matrix = np.zeros((n_models, n_models))
    
    all_ids = set()
    for pred_dict in predictions.values():
        all_ids.update(pred_dict.keys())
    all_ids = sorted(all_ids)
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # 计算预测一致性
                agreements = 0
                total = 0
                for account_id in all_ids:
                    if account_id in predictions[model1] and account_id in predictions[model2]:
                        if predictions[model1][account_id] == predictions[model2][account_id]:
                            agreements += 1
                        total += 1
                
                similarity_matrix[i, j] = agreements / total if total > 0 else 0
    
    # 找出最相似和最不同的模型对
    print(f"\n📊 模型相似性分析:")
    similarities = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            sim = similarity_matrix[i, j]
            similarities.append((sim, model_names[i][:30], model_names[j][:30]))
    
    similarities.sort(reverse=True)
    print(f"   最相似的模型对:")
    for sim, m1, m2 in similarities[:3]:
        print(f"     {sim:.4f} - {m1} vs {m2}")
    
    print(f"   最不同的模型对:")
    for sim, m1, m2 in similarities[-3:]:
        print(f"     {sim:.4f} - {m1} vs {m2}")
    
    return similarity_matrix, similarities

def generate_fusion_strategies(analysis_result, file_info):
    """生成融合策略"""
    print("\n🎯 生成精确融合策略...")
    
    # 按分数排序模型
    scored_models = [(name, info['score']) for name, info in file_info.items()]
    scored_models.sort(key=lambda x: x[1], reverse=True)
    
    print(f"📊 模型质量排序 (前10):")
    for i, (name, score) in enumerate(scored_models[:10], 1):
        bad_ratio = file_info[name]['predict_1_ratio']
        print(f"   {i:2d}. {score:.4f} | Bad率:{bad_ratio:.3f} | {name[:50]}...")
    
    # 策略1: 保守融合 - 针对0.9分数目标
    conservative_strategy = {
        'name': 'CONSERVATIVE_090',
        'description': '保守策略，只有高质量模型强烈一致时才预测为Bad',
        'logic': '前6个高分模型中≥5个预测Bad才输出Bad',
        'target_bad_ratio': 0.08,  # 期望Bad率8%左右
        'models': [name for name, _ in scored_models[:6]]
    }
    
    # 策略2: 加权融合 - 基于分数加权
    weighted_strategy = {
        'name': 'WEIGHTED_090',
        'description': '基于REAL F1分数的加权融合',
        'logic': '按分数加权，阈值0.65',
        'target_bad_ratio': 0.10,
        'models': [name for name, _ in scored_models[:8]],
        'weights': [score for _, score in scored_models[:8]]
    }
    
    # 策略3: 争议账户专门处理
    dispute_strategy = {
        'name': 'DISPUTE_FOCUSED_090',
        'description': '对争议账户使用顶级模型决定',
        'logic': '一致账户直接决定，争议账户只看前3个最高分模型',
        'target_bad_ratio': 0.09,
        'models': [name for name, _ in scored_models[:3]]
    }
    
    strategies = [conservative_strategy, weighted_strategy, dispute_strategy]
    
    print(f"\n💡 推荐融合策略:")
    for i, strategy in enumerate(strategies, 1):
        print(f"   {i}. {strategy['name']}")
        print(f"      描述: {strategy['description']}")
        print(f"      逻辑: {strategy['logic']}")
        print(f"      目标Bad率: {strategy['target_bad_ratio']:.1%}")
        print()
    
    return strategies

def implement_strategies(predictions, file_info, analysis_result, strategies):
    """实现融合策略"""
    print("🚀 实现融合策略...")
    
    all_ids = analysis_result['all_ids']
    results = {}
    
    for strategy in strategies:
        print(f"\n🎲 执行策略: {strategy['name']}")
        
        fusion_pred = {}
        
        if strategy['name'] == 'CONSERVATIVE_090':
            # 保守策略：前6个模型中≥5个预测Bad
            top_models = strategy['models'][:6]
            for account_id in all_ids:
                votes = [predictions[model].get(account_id, 0) for model in top_models if account_id in predictions[model]]
                if len(votes) >= 5:  # 至少5个模型有预测
                    fusion_pred[account_id] = 1 if sum(votes) >= 5 else 0
                else:
                    fusion_pred[account_id] = 0
        
        elif strategy['name'] == 'WEIGHTED_090':
            # 加权策略
            top_models = strategy['models'][:8]
            weights = [info['score'] for name, info in file_info.items() if name in top_models]
            total_weight = sum(weights)
            normalized_weights = [w/total_weight for w in weights]
            
            for account_id in all_ids:
                weighted_sum = 0
                available_weight = 0
                for i, model in enumerate(top_models):
                    if account_id in predictions[model]:
                        weighted_sum += predictions[model][account_id] * normalized_weights[i]
                        available_weight += normalized_weights[i]
                
                if available_weight > 0:
                    fusion_pred[account_id] = 1 if weighted_sum / available_weight >= 0.65 else 0
                else:
                    fusion_pred[account_id] = 0
        
        elif strategy['name'] == 'DISPUTE_FOCUSED_090':
            # 争议处理策略
            unanimous_good_ids = {a['account_id'] for a in analysis_result['unanimous_good']}
            unanimous_bad_ids = {a['account_id'] for a in analysis_result['unanimous_bad']}
            high_consensus_good_ids = {a['account_id'] for a in analysis_result['high_consensus_good']}
            high_consensus_bad_ids = {a['account_id'] for a in analysis_result['high_consensus_bad']}
            disputed_ids = {a['account_id'] for a in analysis_result['disputed']}
            
            top3_models = strategy['models'][:3]
            
            for account_id in all_ids:
                if account_id in unanimous_good_ids or account_id in high_consensus_good_ids:
                    fusion_pred[account_id] = 0
                elif account_id in unanimous_bad_ids or account_id in high_consensus_bad_ids:
                    fusion_pred[account_id] = 1
                elif account_id in disputed_ids:
                    # 争议账户：只看前3个最高分模型
                    votes = [predictions[model].get(account_id, 0) for model in top3_models if account_id in predictions[model]]
                    fusion_pred[account_id] = 1 if sum(votes) >= 2 else 0
                else:
                    fusion_pred[account_id] = 0
        
        results[strategy['name']] = fusion_pred
        
        # 统计结果
        pred_counts = Counter(fusion_pred.values())
        bad_ratio = pred_counts[1] / len(fusion_pred)
        print(f"   结果: Bad {pred_counts[1]} ({bad_ratio:.3f}), Good {pred_counts[0]} ({1-bad_ratio:.3f})")
    
    return results

def save_fusion_results(results):
    """保存融合结果"""
    print("\n💾 保存融合结果...")
    
    results_dir = Path("/Users/mannormal/4011/Qi Zihan/v2/results/high_score_predictions")
    
    for strategy_name, predictions in results.items():
        df = pd.DataFrame({
            'ID': sorted(predictions.keys()),
            'Predict': [predictions[id] for id in sorted(predictions.keys())]
        })
        
        filename = f"FUSION_{strategy_name}.csv"
        filepath = results_dir / filename
        df.to_csv(filepath, index=False)
        
        pred_counts = Counter(df['Predict'])
        print(f"✅ {filename}")
        print(f"   Good (0): {pred_counts[0]} ({pred_counts[0]/len(df)*100:.1f}%)")
        print(f"   Bad (1):  {pred_counts[1]} ({pred_counts[1]/len(df)*100:.1f}%)")

def main():
    print("🎯🔍🎯 深度预测分析器 - 目标0.9分数! 🎯🔍🎯")
    print("=" * 60)
    
    # 1. 加载预测
    predictions, file_info = analyze_predictions()
    
    # 2. 分析一致性模式
    analysis_result = analyze_agreement_patterns(predictions, file_info)
    
    # 3. 分析模型相似性
    similarity_matrix, similarities = analyze_model_similarities(predictions, file_info)
    
    # 4. 生成融合策略
    strategies = generate_fusion_strategies(analysis_result, file_info)
    
    # 5. 实现融合策略
    fusion_results = implement_strategies(predictions, file_info, analysis_result, strategies)
    
    # 6. 保存结果
    save_fusion_results(fusion_results)
    
    print("\n🎉 分析完成！")
    print("\n💡 建议提交顺序:")
    print("   1. FUSION_CONSERVATIVE_090.csv (最保守，Bad率最低)")
    print("   2. FUSION_DISPUTE_FOCUSED_090.csv (处理争议账户)")  
    print("   3. FUSION_WEIGHTED_090.csv (如果前面效果好)")
    print("\n🎯 期待突破0.9分数！")

if __name__ == "__main__":
    main()