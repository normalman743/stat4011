import pandas as pd
import numpy as np

def analyze_all_classification_systems():
    """分析所有6个分类系统的分布"""
    
    print("=== 分析所有分类系统分布 ===")
    
    # 1. 分析当前账户类型分布
    features_path = '/Users/mannormal/4011/Qi Zihan/v2/feature_extraction/result/features_cleaned_no_leakage1.csv'
    df = pd.read_csv(features_path)
    
    def classify_account_type_improved(row):
        forward_strength = (row['A_fprofit'] + row['B_fprofit']) / max(row['A_fsize'] + row['B_fsize'], 1)
        backward_strength = (row['A_bprofit'] + row['B_bprofit']) / max(row['A_bsize'] + row['B_bsize'], 1)
        a_dominance = (row['A_fprofit'] + row['A_bprofit']) / max(row['A_fprofit'] + row['A_bprofit'] + row['B_fprofit'] + row['B_bprofit'], 1)
        network_activity = row['out_degree'] + row['in_degree'] + row['neighbor_count_1hop']
        activity_intensity = row['activity_intensity']
        
        if network_activity > 0.528 and activity_intensity > 0.00189:
            return 'type1'
        elif a_dominance > 0.479 and forward_strength > backward_strength:
            return 'type2'
        elif a_dominance < 0.476 and backward_strength > forward_strength:
            return 'type3'
        else:
            return 'type4'
    
    df['account_type'] = df.apply(classify_account_type_improved, axis=1)
    
    print("1. 当前账户类型分布:")
    account_type_dist = df['account_type'].value_counts()
    for type_name, count in account_type_dist.items():
        percentage = count / len(df) * 100
        print(f"   {type_name}: {count} ({percentage:.1f}%)")
    
    # 2. 分析策略分类系统
    strategy_paths = {
        'traditional': '/Users/mannormal/4011/Qi Zihan/v1/classification_strategies/traditional_4types/traditional_category_mapping.csv',
        'volume': '/Users/mannormal/4011/Qi Zihan/v1/classification_strategies/volume_based/volume_category_mapping.csv',
        'profit': '/Users/mannormal/4011/Qi Zihan/v1/classification_strategies/profit_based/profit_category_mapping.csv',
        'interaction': '/Users/mannormal/4011/Qi Zihan/v1/classification_strategies/interaction_based/interaction_category_mapping.csv',
        'behavior': '/Users/mannormal/4011/Qi Zihan/v1/classification_strategies/behavior_based/behavior_category_mapping.csv'
    }
    
    all_classifications = {}
    
    for strategy_name, path in strategy_paths.items():
        try:
            strategy_df = pd.read_csv(path)
            print(f"\n2. {strategy_name.upper()} 策略分布:")
            
            # 计算分布
            strategy_dist = strategy_df.iloc[:, 1].value_counts()  # 第二列是分类
            for category, count in strategy_dist.items():
                percentage = count / len(strategy_df) * 100
                print(f"   {category}: {count} ({percentage:.1f}%)")
            
            all_classifications[strategy_name] = strategy_df
            
            # 检查是否分布合理（20-30%目标）
            percentages = [count / len(strategy_df) * 100 for count in strategy_dist.values]
            max_pct = max(percentages)
            min_pct = min(percentages)
            
            if max_pct > 50:
                print(f"   ⚠️  分布不均匀: 最大类别占{max_pct:.1f}%")
            elif min_pct < 10:
                print(f"   ⚠️  分布不均匀: 最小类别仅占{min_pct:.1f}%")
            else:
                print(f"   ✅ 分布相对均匀")
                
        except Exception as e:
            print(f"   ❌ 无法读取{strategy_name}: {e}")
    
    # 3. 分析组合后的总体分布
    print(f"\n3. 总体分析:")
    print(f"   总分类系统数: {len(all_classifications) + 1}")  # +1 for account_type
    print(f"   账户类型: 4个类型")
    
    for strategy_name, strategy_df in all_classifications.items():
        unique_categories = strategy_df.iloc[:, 1].nunique()
        print(f"   {strategy_name}: {unique_categories}个类型")
    
    # 4. 建议优化方案
    print(f"\n4. 优化建议:")
    
    # 检查账户类型分布是否需要调整
    account_percentages = [count / len(df) * 100 for count in account_type_dist.values]
    
    if max(account_percentages) > 50 or min(account_percentages) < 15:
        print("   📝 账户分类需要调整:")
        
        # 重新计算衍生特征
        df['forward_strength'] = (df['A_fprofit'] + df['B_fprofit']) / np.maximum(df['A_fsize'] + df['B_fsize'], 1)
        df['backward_strength'] = (df['A_bprofit'] + df['B_bprofit']) / np.maximum(df['A_bsize'] + df['B_bsize'], 1)
        total_profit = df['A_fprofit'] + df['A_bprofit'] + df['B_fprofit'] + df['B_bprofit']
        df['a_dominance'] = (df['A_fprofit'] + df['A_bprofit']) / np.maximum(total_profit, 1)
        df['network_activity'] = df['out_degree'] + df['in_degree'] + df['neighbor_count_1hop']
        
        # 计算更均匀的阈值
        network_q60 = df['network_activity'].quantile(0.6)
        network_q40 = df['network_activity'].quantile(0.4)
        activity_q60 = df['activity_intensity'].quantile(0.6)
        activity_q40 = df['activity_intensity'].quantile(0.4)
        dominance_q70 = df['a_dominance'].quantile(0.7)
        dominance_q30 = df['a_dominance'].quantile(0.3)
        
        print(f"      建议调整网络活跃度阈值: {network_q40:.3f} - {network_q60:.3f}")
        print(f"      建议调整活跃度强度阈值: {activity_q40:.6f} - {activity_q60:.6f}")
        print(f"      建议调整偏好阈值: {dominance_q30:.3f} - {dominance_q70:.3f}")
        
        # 测试更均匀的分类
        def classify_account_balanced(row):
            forward_strength = (row['A_fprofit'] + row['B_fprofit']) / max(row['A_fsize'] + row['B_fsize'], 1)
            backward_strength = (row['A_bprofit'] + row['B_bprofit']) / max(row['A_bsize'] + row['B_bsize'], 1)
            a_dominance = (row['A_fprofit'] + row['A_bprofit']) / max(row['A_fprofit'] + row['A_bprofit'] + row['B_fprofit'] + row['B_bprofit'], 1)
            network_activity = row['out_degree'] + row['in_degree'] + row['neighbor_count_1hop']
            activity_intensity = row['activity_intensity']
            
            if network_activity > network_q60 and activity_intensity > activity_q60:
                return 'type1'
            elif a_dominance > dominance_q70 and forward_strength > backward_strength:
                return 'type2'
            elif a_dominance < dominance_q30 and backward_strength > forward_strength:
                return 'type3'
            else:
                return 'type4'
        
        df['balanced_type'] = df.apply(classify_account_balanced, axis=1)
        
        print("\n   优化后账户类型分布:")
        balanced_dist = df['balanced_type'].value_counts()
        for type_name, count in balanced_dist.items():
            percentage = count / len(df) * 100
            print(f"      {type_name}: {count} ({percentage:.1f}%)")
    
    else:
        print("   ✅ 当前账户分类分布较为合理")
    
    # 5. 检查策略分类是否需要合并
    for strategy_name, strategy_df in all_classifications.items():
        dist = strategy_df.iloc[:, 1].value_counts()
        small_categories = [cat for cat, count in dist.items() if count / len(strategy_df) < 0.05]
        
        if small_categories:
            print(f"\n   📝 {strategy_name}策略建议合并小类别:")
            for cat in small_categories:
                print(f"      {cat}: {dist[cat]} ({dist[cat]/len(strategy_df)*100:.1f}%)")
    
    return {
        'account_type_dist': account_type_dist,
        'strategy_classifications': all_classifications,
        'feature_data': df
    }

if __name__ == "__main__":
    try:
        results = analyze_all_classification_systems()
        print("\n=== 分析完成 ===")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()