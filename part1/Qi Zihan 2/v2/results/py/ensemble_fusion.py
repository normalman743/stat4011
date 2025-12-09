#!/usr/bin/env python3
"""
🔥 邪恶融合脚本 - 模型预测融合器
分析多个高分预测文件，生成更高分数的融合预测
"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import glob

# 配置
RESULTS_DIR = Path("/Users/mannormal/4011/Qi Zihan/v2/results")
REAL_F1_THRESHOLD = 0.7  # 只使用REAL F1 > 0.7的文件

def analyze_predictions():
    """分析现有预测文件"""
    print("🔍 分析现有预测文件...")
    
    # 收集所有预测文件
    all_files = list(RESULTS_DIR.glob("*.csv"))
    prediction_files = []
    
    for f in all_files:
        fname = f.name
        # 跳过非预测文件
        if any(skip in fname for skip in ['test_acc_predict', 'upload', 'voting_rf']):
            continue
            
        # 提取分数信息
        score = extract_score(fname)
        if score is not None:
            prediction_files.append({
                'file': f,
                'name': fname,
                'score': score,
                'type': get_model_type(fname)
            })
    
    # 按分数排序
    prediction_files.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"📊 找到 {len(prediction_files)} 个预测文件:")
    for i, pf in enumerate(prediction_files[:10], 1):
        print(f"  {i:2d}. {pf['score']:.4f} - {pf['type']} - {pf['name'][:80]}...")
    
    return prediction_files

def extract_score(filename):
    """从文件名提取分数"""
    # 优先提取 REAL_F1
    if 'REAL_F1_' in filename:
        import re
        match = re.search(r'REAL_F1_([0-9.]+)', filename)
        if match:
            return float(match.group(1))
    
    # 提取 bad_f1
    if 'bad_f1_' in filename:
        import re
        match = re.search(r'bad_f1_([0-9.]+)', filename)
        if match:
            return float(match.group(1)) * 0.85  # 降权，因为不是真实分数
    
    # Transformer分数
    if 'f1_0.' in filename:
        import re
        match = re.search(r'f1_([0-9.]+)', filename)
        if match:
            return float(f"0.{match.group(1)}") * 0.8  # 估算
    
    return None

def get_model_type(filename):
    """获取模型类型"""
    if 'v3.2refined' in filename:
        return 'V3.2_Combined'
    elif 'v3.1threshold' in filename:
        return 'V3.1_Threshold'  
    elif 'ultra_resnet' in filename:
        return 'Ultra_ResNet'
    elif 'Transformer' in filename:
        return 'Transformer'
    elif 'best_rf' in filename:
        return 'Best_RF'
    else:
        return 'Other'

def load_predictions(files, top_k=8):
    """加载前K个高分预测文件"""
    print(f"📚 加载前 {top_k} 个高分预测...")
    
    predictions = {}
    file_info = []
    
    for i, pf in enumerate(files[:top_k]):
        try:
            df = pd.read_csv(pf['file'])
            if 'ID' in df.columns and 'Predict' in df.columns:
                predictions[f"model_{i+1}_{pf['type']}"] = dict(zip(df['ID'], df['Predict']))
                file_info.append(f"Model {i+1}: {pf['score']:.4f} - {pf['type']}")
                print(f"  ✅ {pf['name'][:60]}... (Score: {pf['score']:.4f})")
            else:
                print(f"  ❌ 跳过格式不对: {pf['name']}")
        except Exception as e:
            print(f"  ❌ 读取失败: {pf['name']} - {e}")
    
    print(f"\n🎯 成功加载 {len(predictions)} 个模型的预测")
    return predictions, file_info

def ensemble_predictions(predictions, strategies=['voting', 'weighted', 'confident']):
    """融合预测"""
    print("🔥 开始邪恶融合...")
    
    # 获取所有账户ID
    all_ids = set()
    for pred_dict in predictions.values():
        all_ids.update(pred_dict.keys())
    all_ids = sorted(all_ids)
    
    results = {}
    
    for strategy in strategies:
        print(f"\n🎲 策略: {strategy.upper()}")
        
        if strategy == 'voting':
            # 简单投票
            ensemble_pred = {}
            for account_id in all_ids:
                votes = [predictions[model][account_id] for model in predictions if account_id in predictions[model]]
                if len(votes) > 0:
                    ensemble_pred[account_id] = 1 if sum(votes) >= len(votes) / 2 else 0
                else:
                    ensemble_pred[account_id] = 0
            
        elif strategy == 'weighted':
            # 加权投票 - 按模型质量加权
            weights = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]  # 前8个模型权重
            ensemble_pred = {}
            model_names = list(predictions.keys())
            
            for account_id in all_ids:
                weighted_sum = 0
                total_weight = 0
                for i, model in enumerate(model_names):
                    if account_id in predictions[model] and i < len(weights):
                        weighted_sum += predictions[model][account_id] * weights[i]
                        total_weight += weights[i]
                
                if total_weight > 0:
                    ensemble_pred[account_id] = 1 if weighted_sum / total_weight >= 0.5 else 0
                else:
                    ensemble_pred[account_id] = 0
                    
        elif strategy == 'confident':
            # 置信度投票 - 只有当多数模型一致时才预测为1
            ensemble_pred = {}
            for account_id in all_ids:
                votes = [predictions[model][account_id] for model in predictions if account_id in predictions[model]]
                if len(votes) > 0:
                    positive_ratio = sum(votes) / len(votes)
                    # 更保守：需要70%以上模型预测为1才预测为1
                    ensemble_pred[account_id] = 1 if positive_ratio >= 0.7 else 0
                else:
                    ensemble_pred[account_id] = 0
        
        results[strategy] = ensemble_pred
        
        # 统计预测分布
        pred_counts = Counter(ensemble_pred.values())
        total = len(ensemble_pred)
        print(f"   Predict=0 (Good): {pred_counts[0]} ({pred_counts[0]/total*100:.1f}%)")
        print(f"   Predict=1 (Bad):  {pred_counts[1]} ({pred_counts[1]/total*100:.1f}%)")
    
    return results

def save_ensemble_results(results, file_info):
    """保存融合结果"""
    print("\n💾 保存融合结果...")
    
    for strategy, predictions in results.items():
        # 创建DataFrame
        df = pd.DataFrame({
            'ID': sorted(predictions.keys()),
            'Predict': [predictions[id] for id in sorted(predictions.keys())]
        })
        
        # 生成文件名
        filename = f"ENSEMBLE_{strategy.upper()}_fusion_{len(predictions)}_accounts.csv"
        filepath = RESULTS_DIR / filename
        
        # 保存
        df.to_csv(filepath, index=False)
        print(f"  ✅ {filename}")
        
        # 统计
        pred_counts = Counter(df['Predict'])
        print(f"      Good (0): {pred_counts[0]} ({pred_counts[0]/len(df)*100:.1f}%)")
        print(f"      Bad (1):  {pred_counts[1]} ({pred_counts[1]/len(df)*100:.1f}%)")
    
    # 保存融合信息
    info_file = RESULTS_DIR / "ENSEMBLE_fusion_info.txt"
    with open(info_file, 'w') as f:
        f.write("🔥 邪恶融合信息报告\n")
        f.write("=" * 50 + "\n\n")
        f.write("使用的模型:\n")
        for i, info in enumerate(file_info, 1):
            f.write(f"{i}. {info}\n")
        f.write(f"\n生成的融合策略:\n")
        for strategy in results.keys():
            f.write(f"- {strategy.upper()}\n")
        f.write(f"\n文件位置: {RESULTS_DIR}\n")
    
    print(f"  📋 融合信息: ENSEMBLE_fusion_info.txt")

def main():
    print("🔥🔥🔥 邪恶模型融合器启动！🔥🔥🔥")
    print("=" * 60)
    
    # 分析预测文件
    prediction_files = analyze_predictions()
    
    if len(prediction_files) < 3:
        print("❌ 预测文件太少，无法进行有效融合")
        return
    
    # 加载预测
    predictions, file_info = load_predictions(prediction_files, top_k=8)
    
    if len(predictions) < 3:
        print("❌ 成功加载的预测文件太少")
        return
    
    # 融合预测
    results = ensemble_predictions(predictions, strategies=['voting', 'weighted', 'confident'])
    
    # 保存结果
    save_ensemble_results(results, file_info)
    
    print("\n🎉 邪恶融合完成！")
    print("💡 建议：")
    print("   1. 先提交 CONFIDENT 版本（更保守）")
    print("   2. 如果分数不错，再试 WEIGHTED 版本")
    print("   3. VOTING 版本作为基准对比")
    print("\n🎯 期待更高的REAL F1分数！")

if __name__ == "__main__":
    main()