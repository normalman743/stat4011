#!/usr/bin/env python3
"""
🔥 激进融合策略 - 针对Bad:Good=1:9的真实分布
修正过于保守的模型，提升Bad率到合理的10-12%
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

def load_high_score_predictions():
    """加载高分预测"""
    high_score_dir = Path("/Users/mannormal/4011/Qi Zihan/v2/results/high_score_predictions")
    
    predictions = {}
    scores = {}
    
    # 手动指定最可信的文件（基于你的分析结果）
    trusted_files = {
        'v3.2refined_fold4_bad_f1_0.8250_good_0.9814_bad_0.8250_macro_0.9032_weighted_0.9661_seed_13_REAL_F1_0.7525325615050651_REAL_F1_0.7525325615050651.csv': 0.7525,
        'v3.2refined_fold1_bad_f1_0.7778_good_0.9765_bad_0.7778_macro_0.8771_weighted_0.9570_seed_13_REAL_F1_0.7549378200438918.csv': 0.7549,
        'best_rf_badf1_0691_ratio_0.731.csv': 0.731,
        'voting_rf_badf1_0669_ratio_0.736.csv': 0.736,
        'v2ultra_resnet_meta_ann_rank1_fold3_macro_f1_0.8865_good_0.9771_bad_0.7958_macro_0.8865_weighted_0.9594_seed_3650.7432239657631955.csv': 0.743
    }
    
    print("📊 加载可信预测文件...")
    
    for filename, score in trusted_files.items():
        filepath = high_score_dir / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                if 'ID' in df.columns and 'Predict' in df.columns:
                    model_name = filename.split('_')[0] + '_' + filename.split('_')[-1].replace('.csv', '')
                    predictions[model_name] = dict(zip(df['ID'], df['Predict']))
                    scores[model_name] = score
                    bad_rate = df['Predict'].mean()
                    print(f"✅ {model_name:<20} | Score: {score:.4f} | Bad率: {bad_rate:.3f}")
            except Exception as e:
                print(f"❌ {filename}: {e}")
    
    # 如果某些文件不存在，尝试加载其他高分文件
    if len(predictions) < 5:
        print("🔍 补充加载其他高分文件...")
        for csv_file in high_score_dir.glob("*.csv"):
            if len(predictions) >= 8:
                break
            if csv_file.name not in trusted_files:
                try:
                    df = pd.read_csv(csv_file)
                    if 'ID' in df.columns and 'Predict' in df.columns:
                        model_name = csv_file.stem[:15]
                        predictions[model_name] = dict(zip(df['ID'], df['Predict']))
                        scores[model_name] = 0.72  # 估计分数
                        bad_rate = df['Predict'].mean()
                        print(f"✅ {model_name:<20} | Score: ~0.720 | Bad率: {bad_rate:.3f}")
                except:
                    continue
    
    print(f"📊 总共加载 {len(predictions)} 个模型")
    return predictions, scores

def aggressive_fusion_strategies(predictions, scores):
    """激进融合策略"""
    print("\n🔥 实施激进融合策略...")
    
    # 获取所有账户ID
    all_ids = set()
    for pred_dict in predictions.values():
        all_ids.update(pred_dict.keys())
    all_ids = sorted(all_ids)
    
    # 按分数排序模型
    sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    results = {}
    
    # 策略1: 激进投票 - 降低阈值
    print("\n🎯 策略1: AGGRESSIVE_VOTING")
    print("   逻辑: 任意3个模型预测Bad就输出Bad (目标Bad率12%)")
    
    aggressive_pred = {}
    for account_id in all_ids:
        votes = [predictions[model].get(account_id, 0) for model in predictions if account_id in predictions[model]]
        # 激进策略：只需要3票就预测Bad
        aggressive_pred[account_id] = 1 if sum(votes) >= 3 else 0
    
    results['AGGRESSIVE_VOTING'] = aggressive_pred
    
    # 策略2: 概率阈值降低
    print("\n🎯 策略2: PROBABILITY_THRESHOLD")  
    print("   逻辑: 加权概率≥0.35就预测Bad (目标Bad率11%)")
    
    # 计算加权概率
    model_names = [name for name, _ in sorted_models[:6]]
    weights = [score for _, score in sorted_models[:6]]
    total_weight = sum(weights)
    normalized_weights = [w/total_weight for w in weights]
    
    threshold_pred = {}
    for account_id in all_ids:
        weighted_sum = 0
        available_weight = 0
        
        for i, model in enumerate(model_names):
            if account_id in predictions[model]:
                weighted_sum += predictions[model][account_id] * normalized_weights[i]
                available_weight += normalized_weights[i]
        
        if available_weight > 0:
            prob = weighted_sum / available_weight
            threshold_pred[account_id] = 1 if prob >= 0.35 else 0  # 降低阈值
        else:
            threshold_pred[account_id] = 0
    
    results['PROBABILITY_THRESHOLD'] = threshold_pred
    
    # 策略3: 最激进模型主导
    print("\n🎯 策略3: TOP_MODEL_AGGRESSIVE")
    print("   逻辑: 最高分模型说Bad就是Bad，其他模型2票确认 (目标Bad率10%)")
    
    best_model = sorted_models[0][0]
    aggressive_dominant = {}
    
    for account_id in all_ids:
        if account_id in predictions[best_model] and predictions[best_model][account_id] == 1:
            # 最高分模型预测Bad，直接采纳
            aggressive_dominant[account_id] = 1
        else:
            # 其他情况需要至少2票
            votes = [predictions[model].get(account_id, 0) for model in predictions if account_id in predictions[model]]
            aggressive_dominant[account_id] = 1 if sum(votes) >= 2 else 0
    
    results['TOP_MODEL_AGGRESSIVE'] = aggressive_dominant
    
    # 策略4: 数据分布匹配
    print("\n🎯 策略4: DISTRIBUTION_MATCHING")
    print("   逻辑: 强制匹配训练集分布，取概率最高的10% (目标Bad率10%)")
    
    # 计算每个账户的平均预测概率
    account_probs = {}
    for account_id in all_ids:
        votes = [predictions[model].get(account_id, 0) for model in predictions if account_id in predictions[model]]
        account_probs[account_id] = sum(votes) / len(votes) if votes else 0
    
    # 按概率排序，取前10%作为Bad
    sorted_accounts = sorted(account_probs.items(), key=lambda x: x[1], reverse=True)
    top_10_percent = int(len(sorted_accounts) * 0.10)
    
    distribution_pred = {}
    for account_id in all_ids:
        distribution_pred[account_id] = 0
    
    for i in range(top_10_percent):
        account_id = sorted_accounts[i][0]
        distribution_pred[account_id] = 1
    
    results['DISTRIBUTION_MATCHING'] = distribution_pred
    
    # 打印统计
    print("\n📊 策略结果统计:")
    for name, pred in results.items():
        counts = Counter(pred.values())
        bad_rate = counts[1] / len(pred)
        print(f"   {name:<25} | Bad: {counts[1]:4d} ({bad_rate:6.1%}) | Good: {counts[0]:4d} ({1-bad_rate:6.1%})")
    
    return results

def save_aggressive_results(results):
    """保存激进融合结果"""
    print("\n💾 保存激进融合结果...")
    
    results_dir = Path("/Users/mannormal/4011/Qi Zihan/v2/results/high_score_predictions")
    
    for strategy_name, predictions in results.items():
        df = pd.DataFrame({
            'ID': sorted(predictions.keys()),
            'Predict': [predictions[id] for id in sorted(predictions.keys())]
        })
        
        filename = f"AGGRESSIVE_{strategy_name}.csv"
        filepath = results_dir / filename
        df.to_csv(filepath, index=False)
        
        pred_counts = Counter(df['Predict'])
        bad_rate = pred_counts[1] / len(df)
        print(f"✅ {filename}")
        print(f"   Bad (1):  {pred_counts[1]:4d} ({bad_rate:6.1%}) ← 更接近真实分布")
        print(f"   Good (0): {pred_counts[0]:4d} ({1-bad_rate:6.1%})")

def main():
    print("🔥🎯 激进融合策略 - 修正保守预测，匹配真实分布 1:9 🎯🔥")
    print("=" * 70)
    
    # 1. 加载预测
    predictions, scores = load_high_score_predictions()
    
    if len(predictions) < 3:
        print("❌ 预测文件不足")
        return
    
    # 2. 执行激进策略
    results = aggressive_fusion_strategies(predictions, scores)
    
    # 3. 保存结果
    save_aggressive_results(results)
    
    print("\n🎉 激进融合完成！")
    print("\n🎯 建议提交顺序 (从保守到激进):")
    print("   1. AGGRESSIVE_TOP_MODEL_AGGRESSIVE.csv (10%Bad率)")
    print("   2. AGGRESSIVE_DISTRIBUTION_MATCHING.csv (精确10%Bad率)")  
    print("   3. AGGRESSIVE_PROBABILITY_THRESHOLD.csv (11%Bad率)")
    print("   4. AGGRESSIVE_AGGRESSIVE_VOTING.csv (最激进，12%Bad率)")
    print("\n💡 这些策略更符合训练数据的真实分布!")
    print("🚀 期待突破到真正的高分!")

if __name__ == "__main__":
    main()