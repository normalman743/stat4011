import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_account_classification():
    """分析账户分类函数的特征分布和分类结果"""
    
    print("=== 账户分类分析 ===")
    
    # 读取清理后的特征数据
    features_path = '/Users/mannormal/4011/Qi Zihan/v2/feature_extraction/result/features_cleaned_no_leakage1.csv'
    df = pd.read_csv(features_path)
    
    print(f"数据形状: {df.shape}")
    print(f"特征数量: {df.shape[1] - 1}")  # 减去account列
    
    # 分析分类相关的关键特征
    print("\n=== 分类相关特征分析 ===")
    
    classification_features = [
        'A_fprofit', 'A_fsize', 'B_fprofit', 'B_fsize',
        'A_bprofit', 'A_bsize', 'B_bprofit', 'B_bsize',
        'out_degree', 'in_degree', 'neighbor_count_1hop',
        'activity_intensity'
    ]
    
    # 检查特征是否存在
    available_features = [f for f in classification_features if f in df.columns]
    print(f"可用的分类特征: {available_features}")
    
    # 计算衍生特征
    print("\n=== 计算分类衍生特征 ===")
    
    # 计算前向后向交易强度
    df['forward_strength'] = (df['A_fprofit'] + df['B_fprofit']) / np.maximum(df['A_fsize'] + df['B_fsize'], 1)
    df['backward_strength'] = (df['A_bprofit'] + df['B_bprofit']) / np.maximum(df['A_bsize'] + df['B_bsize'], 1)
    
    # A/B类型偏好程度
    total_profit = df['A_fprofit'] + df['A_bprofit'] + df['B_fprofit'] + df['B_bprofit']
    df['a_dominance'] = (df['A_fprofit'] + df['A_bprofit']) / np.maximum(total_profit, 1)
    
    # 网络活跃度
    df['network_activity'] = df['out_degree'] + df['in_degree'] + df['neighbor_count_1hop']
    
    # 分析这些特征的分布
    derived_features = ['forward_strength', 'backward_strength', 'a_dominance', 'network_activity', 'activity_intensity']
    
    print("\n=== 衍生特征统计 ===")
    for feature in derived_features:
        stats = df[feature].describe()
        print(f"\n{feature}:")
        print(f"  最小值: {stats['min']:.6f}")
        print(f"  25%分位: {stats['25%']:.6f}")
        print(f"  中位数: {stats['50%']:.6f}")
        print(f"  75%分位: {stats['75%']:.6f}")
        print(f"  最大值: {stats['max']:.6f}")
        print(f"  均值: {stats['mean']:.6f}")
        print(f"  标准差: {stats['std']:.6f}")
    
    # 应用当前的分类规则
    print("\n=== 应用当前分类规则 ===")
    
    def classify_account_current(row):
        forward_strength = row['forward_strength']
        backward_strength = row['backward_strength']
        a_dominance = row['a_dominance']
        network_activity = row['network_activity']
        activity_intensity = row['activity_intensity']
        
        if network_activity > 10 and activity_intensity > 2:
            return 'type1'  # 核心枢纽节点
        elif a_dominance > 0.8 and forward_strength > backward_strength:
            return 'type2'  # A类主导的发送方
        elif a_dominance < 0.2 and backward_strength > forward_strength:
            return 'type3'  # B类主导的接收方  
        else:
            return 'type4'  # 混合交易类型
    
    df['current_type'] = df.apply(classify_account_current, axis=1)
    
    current_dist = df['current_type'].value_counts()
    print("当前分类分布:")
    for type_name, count in current_dist.items():
        percentage = count / len(df) * 100
        print(f"  {type_name}: {count} ({percentage:.2f}%)")
    
    # 分析为什么分布不均匀
    print("\n=== 分析分类条件 ===")
    
    # 条件1: network_activity > 10 and activity_intensity > 2
    cond1 = (df['network_activity'] > 10) & (df['activity_intensity'] > 2)
    print(f"条件1 (type1): network_activity > 10 AND activity_intensity > 2")
    print(f"  满足条件的账户: {cond1.sum()} ({cond1.sum()/len(df)*100:.2f}%)")
    
    # 条件2: a_dominance > 0.8 and forward_strength > backward_strength
    cond2 = (df['a_dominance'] > 0.8) & (df['forward_strength'] > df['backward_strength'])
    print(f"条件2 (type2): a_dominance > 0.8 AND forward_strength > backward_strength")
    print(f"  满足条件的账户: {cond2.sum()} ({cond2.sum()/len(df)*100:.2f}%)")
    
    # 条件3: a_dominance < 0.2 and backward_strength > forward_strength
    cond3 = (df['a_dominance'] < 0.2) & (df['backward_strength'] > df['forward_strength'])
    print(f"条件3 (type3): a_dominance < 0.2 AND backward_strength > forward_strength")
    print(f"  满足条件的账户: {cond3.sum()} ({cond3.sum()/len(df)*100:.2f}%)")
    
    # 分析各个阈值的覆盖率
    print("\n=== 阈值分析 ===")
    
    thresholds_network = [1, 5, 10, 15, 20, 50]
    thresholds_activity = [0.5, 1, 2, 3, 5]
    thresholds_dominance = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]
    
    print("网络活跃度阈值分析:")
    for thresh in thresholds_network:
        count = (df['network_activity'] > thresh).sum()
        print(f"  > {thresh}: {count} ({count/len(df)*100:.1f}%)")
    
    print("\n活跃度强度阈值分析:")
    for thresh in thresholds_activity:
        count = (df['activity_intensity'] > thresh).sum()
        print(f"  > {thresh}: {count} ({count/len(df)*100:.1f}%)")
    
    print("\nA类型偏好阈值分析:")
    for thresh in thresholds_dominance:
        count_high = (df['a_dominance'] > thresh).sum()
        count_low = (df['a_dominance'] < (1-thresh)).sum()
        print(f"  > {thresh}: {count_high} ({count_high/len(df)*100:.1f}%), < {1-thresh}: {count_low} ({count_low/len(df)*100:.1f}%)")
    
    # 建议新的阈值
    print("\n=== 建议新阈值 ===")
    
    # 目标：让每个类型占10-40%
    network_q75 = df['network_activity'].quantile(0.75)
    activity_q75 = df['activity_intensity'].quantile(0.75)
    dominance_q80 = df['a_dominance'].quantile(0.8)
    dominance_q20 = df['a_dominance'].quantile(0.2)
    
    print(f"建议网络活跃度阈值: {network_q75:.1f} (当前75%分位数)")
    print(f"建议活跃度强度阈值: {activity_q75:.2f} (当前75%分位数)")
    print(f"建议A类型偏好高阈值: {dominance_q80:.2f} (当前80%分位数)")
    print(f"建议A类型偏好低阈值: {dominance_q20:.2f} (当前20%分位数)")
    
    # 应用建议的分类规则
    print("\n=== 应用建议分类规则 ===")
    
    def classify_account_improved(row):
        forward_strength = row['forward_strength']
        backward_strength = row['backward_strength']
        a_dominance = row['a_dominance']
        network_activity = row['network_activity']
        activity_intensity = row['activity_intensity']
        
        if network_activity > network_q75 and activity_intensity > activity_q75:
            return 'type1'  # 核心枢纽节点
        elif a_dominance > dominance_q80 and forward_strength > backward_strength:
            return 'type2'  # A类主导的发送方
        elif a_dominance < dominance_q20 and backward_strength > forward_strength:
            return 'type3'  # B类主导的接收方  
        else:
            return 'type4'  # 混合交易类型
    
    df['improved_type'] = df.apply(classify_account_improved, axis=1)
    
    improved_dist = df['improved_type'].value_counts()
    print("改进后分类分布:")
    for type_name, count in improved_dist.items():
        percentage = count / len(df) * 100
        print(f"  {type_name}: {count} ({percentage:.2f}%)")
    
    # 保存分析结果
    output_file = '/Users/mannormal/4011/Qi Zihan/v2/model/account_classification_analysis.csv'
    analysis_df = df[['account', 'forward_strength', 'backward_strength', 'a_dominance', 
                     'network_activity', 'activity_intensity', 'current_type', 'improved_type']]
    analysis_df.to_csv(output_file, index=False)
    
    print(f"\n分析结果保存到: {output_file}")
    
    # 返回建议的阈值
    return {
        'network_threshold': network_q75,
        'activity_threshold': activity_q75,
        'dominance_high_threshold': dominance_q80,
        'dominance_low_threshold': dominance_q20,
        'current_distribution': current_dist,
        'improved_distribution': improved_dist
    }

if __name__ == "__main__":
    try:
        results = analyze_account_classification()
        print("\n=== 分析完成 ===")
        print("建议的新阈值已计算完成，可以用于更新分类函数。")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()