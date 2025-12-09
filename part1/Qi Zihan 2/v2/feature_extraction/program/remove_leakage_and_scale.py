import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def remove_leakage_features_and_scale():
    """
    删除标签泄露特征并对金钱相关特征进行0-1放缩
    """
    print("=== 删除标签泄露特征并进行数值放缩 ===")
    
    # 读取原始特征文件
    input_file = "/Users/mannormal/4011/Qi Zihan/v2/feature_extraction/result/all_features_no_leakage_yyyymmdd.csv"
    df = pd.read_csv(input_file)
    
    print(f"原始特征数量: {df.shape[1]-1} 个特征, {df.shape[0]} 个账户")
    
    # 定义需要删除的泄露特征
    leakage_features = [
        # 基于bad标签的特征
        'bad_fprofit', 'bad_fpprofit', 'bad_fsize',
        'bad_bprofit', 'bad_bpprofit', 'bad_bsize',
        
        # 基于abnormal标签的特征 
        'abnormal_fprofit', 'abnormal_fpprofit', 'abnormal_fsize',
        'abnormal_bprofit', 'abnormal_bpprofit', 'abnormal_bsize',
        
        # 邻居bad比例特征
        'neighbor_bad_ratio_train_only'
    ]
    
    print("\n=== 删除泄露特征 ===")
    # 检查哪些泄露特征实际存在
    existing_leakage = [col for col in leakage_features if col in df.columns]
    print(f"检测到的泄露特征: {existing_leakage}")
    
    # 删除泄露特征
    df_clean = df.drop(columns=existing_leakage, errors='ignore')
    print(f"删除后特征数量: {df_clean.shape[1]-1} 个特征")
    
    print("\n=== 应用对数变换 ===")
    
    # 定义涉及金钱的特征（需要放缩的特征）
    money_features = [
        # profit相关特征
        'normal_fprofit', 'normal_fpprofit',
        'normal_bprofit', 'normal_bpprofit', 
        'A_fprofit', 'A_fpprofit', 'A_bprofit', 'A_bpprofit',
        'B_fprofit', 'B_fpprofit', 'B_bprofit', 'B_bpprofit',
        
        # 交易价值相关
        'transaction_value_variance',
        'gas_price_deviation'
    ]
    
    # 检查实际存在的金钱特征
    existing_money_features = [col for col in money_features if col in df_clean.columns]
    print(f"检测到的金钱特征: {existing_money_features}")
    
    # 对每个金钱特征进行处理
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    for feature in existing_money_features:
        print(f"处理特征: {feature}")
        
        # 获取非空值
        non_null_mask = df_clean[feature].notna()
        values = df_clean.loc[non_null_mask, feature].values.reshape(-1, 1)
        
        if len(values) > 0:
            # 处理极值和负值
            # 对于可能有负值的profit特征，先加上偏移量使其非负
            if 'profit' in feature:
                min_val = values.min()
                if min_val < 0:
                    values = values - min_val  # 平移使最小值为0
                    print(f"  {feature}: 平移 {-min_val:.2e} 使其非负")
            
            # 对于极大值，使用对数变换
            if values.max() > 1e10:  # 如果最大值很大
                # 先取对数，再归一化
                values_log = np.log1p(values.flatten())  # log1p处理0值
                values_scaled = scaler.fit_transform(values_log.reshape(-1, 1)).flatten()
                print(f"  {feature}: 对数变换 + 0-1缩放, 原始范围: [{values.min():.2e}, {values.max():.2e}]")
            else:
                # 直接归一化
                values_scaled = scaler.fit_transform(values).flatten()
                print(f"  {feature}: 直接0-1缩放, 原始范围: [{values.min():.2e}, {values.max():.2e}]")
            
            # 更新数据
            df_clean.loc[non_null_mask, feature] = values_scaled
            print(f"  缩放后范围: [{values_scaled.min():.6f}, {values_scaled.max():.6f}]")
        else:
            print(f"  {feature}: 无有效数据，跳过")
    
    print("\n=== 验证结果 ===")
    # 验证缩放结果
    for feature in existing_money_features:
        if feature in df_clean.columns:
            vals = df_clean[feature].dropna()
            if len(vals) > 0:
                print(f"{feature}: [{vals.min():.6f}, {vals.max():.6f}], 均值: {vals.mean():.6f}")
    
    # 保存处理后的数据
    output_file = "/Users/mannormal/4011/Qi Zihan/v2/feature_extraction/result/features_cleaned_scaled.csv"
    df_clean.to_csv(output_file, index=False)
    
    print(f"\n=== 处理完成 ===")
    print(f"最终特征数量: {df_clean.shape[1]-1} 个特征")
    print(f"删除的泄露特征数量: {len(existing_leakage)} 个")
    print(f"缩放的金钱特征数量: {len(existing_money_features)} 个")
    print(f"结果保存到: {output_file}")
    
    return df_clean

if __name__ == "__main__":
    try:
        result_df = remove_leakage_features_and_scale()
        print("处理成功完成！")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()