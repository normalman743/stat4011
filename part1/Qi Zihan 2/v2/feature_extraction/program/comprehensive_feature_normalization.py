import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
import warnings
import os

warnings.filterwarnings('ignore')

def comprehensive_feature_normalization():
    """
    对特征进行全面的标准化处理，解决极大数值问题
    """
    print("=== 全面特征标准化处理 ===")
    
    # 读取特征文件
    input_file = "/Users/mannormal/4011/Qi Zihan/v2/feature_extraction/result/all_features_no_leakage_yyyymmdd_no_flag_features.csv"
    
    if not os.path.exists(input_file):
        print(f"错误：文件不存在 {input_file}")
        return None
        
    df = pd.read_csv(input_file)
    print(f"原始数据: {df.shape[0]} 个账户, {df.shape[1]-1} 个特征")
    
    # 分析数据范围
    print("\n=== 数据范围分析 ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'account' in numeric_cols:
        numeric_cols.remove('account')
    
    extreme_features = []
    for col in numeric_cols:
        if col != 'account':
            col_data = df[col].dropna()
            if len(col_data) > 0:
                max_val = col_data.max()
                min_val = col_data.min()
                if abs(max_val) > 1e12 or abs(min_val) > 1e12:
                    extreme_features.append((col, min_val, max_val))
                    print(f"{col}: [{min_val:.2e}, {max_val:.2e}] - 极值特征")
    
    print(f"\n检测到 {len(extreme_features)} 个极值特征需要处理")
    
    # 创建数据副本
    df_processed = df.copy()
    
    # 定义不同类型的特征处理策略
    
    # 1. 金钱/利润相关特征 - 使用对数变换
    profit_features = [col for col in numeric_cols if 'profit' in col.lower()]
    
    # 2. 时间相关特征 - 特殊处理
    time_features = ['first_transaction_time', 'last_transaction_time', 'activity_span_days', 
                    'max_transaction_gap_hours']
    
    # 3. 方差类特征 - 对数变换
    variance_features = [col for col in numeric_cols if 'variance' in col.lower()]
    
    # 4. 比例类特征 - 已经在[0,1]范围，无需处理
    ratio_features = [col for col in numeric_cols if 'ratio' in col.lower() or 'entropy' in col.lower()]
    
    # 5. 计数类特征 - 可能需要对数变换
    count_features = [col for col in numeric_cols if 'count' in col.lower() or 'degree' in col.lower()]
    
    print("\n=== 特征分类处理 ===")
    
    # 处理利润特征
    print(f"\n1. 处理利润特征 ({len(profit_features)} 个)")
    for feature in profit_features:
        if feature in df_processed.columns:
            print(f"  处理 {feature}")
            col_data = df_processed[feature].copy()
            
            # 处理负值和零值
            min_val = col_data.min()
            if min_val <= 0:
                # 平移使所有值为正
                col_data = col_data - min_val + 1
                print(f"    平移 {-min_val + 1:.2e} 使其为正")
            
            # 对数变换
            col_data_log = np.log10(col_data + 1)
            
            # 标准化到[0,1]
            scaler = MinMaxScaler()
            col_data_scaled = scaler.fit_transform(col_data_log.values.reshape(-1, 1)).flatten()
            
            df_processed[feature] = col_data_scaled
            print(f"    最终范围: [{col_data_scaled.min():.6f}, {col_data_scaled.max():.6f}]")
    
    # 处理方差特征
    print(f"\n2. 处理方差特征 ({len(variance_features)} 个)")
    for feature in variance_features:
        if feature in df_processed.columns:
            print(f"  处理 {feature}")
            col_data = df_processed[feature].copy()
            
            # 方差总是非负的，直接对数变换
            col_data_log = np.log10(col_data + 1)
            
            # 标准化
            scaler = MinMaxScaler()
            col_data_scaled = scaler.fit_transform(col_data_log.values.reshape(-1, 1)).flatten()
            
            df_processed[feature] = col_data_scaled
            print(f"    最终范围: [{col_data_scaled.min():.6f}, {col_data_scaled.max():.6f}]")
    
    # 处理时间特征
    print(f"\n3. 处理时间特征 ({len(time_features)} 个)")
    for feature in time_features:
        if feature in df_processed.columns:
            print(f"  处理 {feature}")
            col_data = df_processed[feature].copy()
            
            if feature in ['first_transaction_time', 'last_transaction_time']:
                # 转换为相对天数
                min_time = col_data.min()
                col_data = col_data - min_time
                print(f"    转换为相对时间，基准: {min_time}")
            
            # 如果值很大，使用对数变换
            if col_data.max() > 1000:
                col_data = np.log10(col_data + 1)
                print(f"    应用对数变换")
            
            # 标准化
            scaler = MinMaxScaler()
            col_data_scaled = scaler.fit_transform(col_data.values.reshape(-1, 1)).flatten()
            
            df_processed[feature] = col_data_scaled
            print(f"    最终范围: [{col_data_scaled.min():.6f}, {col_data_scaled.max():.6f}]")
    
    # 处理计数特征
    print(f"\n4. 处理计数特征 ({len(count_features)} 个)")
    for feature in count_features:
        if feature in df_processed.columns:
            print(f"  处理 {feature}")
            col_data = df_processed[feature].copy()
            
            # 如果最大值很大，使用对数变换
            if col_data.max() > 100:
                col_data = np.log10(col_data + 1)
                print(f"    应用对数变换")
            
            # 标准化
            scaler = MinMaxScaler()
            col_data_scaled = scaler.fit_transform(col_data.values.reshape(-1, 1)).flatten()
            
            df_processed[feature] = col_data_scaled
            print(f"    最终范围: [{col_data_scaled.min():.6f}, {col_data_scaled.max():.6f}]")
    
    # 处理其他极值特征
    other_extreme = []
    for col, min_val, max_val in extreme_features:
        if (col not in profit_features and col not in variance_features and 
            col not in time_features and col not in count_features and
            col not in ratio_features):
            other_extreme.append(col)
    
    print(f"\n5. 处理其他极值特征 ({len(other_extreme)} 个)")
    for feature in other_extreme:
        if feature in df_processed.columns:
            print(f"  处理 {feature}")
            col_data = df_processed[feature].copy()
            
            # 检查是否有负值
            min_val = col_data.min()
            if min_val < 0:
                col_data = col_data - min_val + 1
                print(f"    平移 {-min_val + 1:.2e} 使其为正")
            
            # 对数变换
            col_data_log = np.log10(col_data + 1)
            
            # 标准化
            scaler = MinMaxScaler()
            col_data_scaled = scaler.fit_transform(col_data_log.values.reshape(-1, 1)).flatten()
            
            df_processed[feature] = col_data_scaled
            print(f"    最终范围: [{col_data_scaled.min():.6f}, {col_data_scaled.max():.6f}]")
    
    # 处理剩余的数值特征
    remaining_features = []
    processed_features = set(profit_features + variance_features + time_features + 
                           count_features + ratio_features + other_extreme)
    
    for col in numeric_cols:
        if col not in processed_features and col != 'account':
            remaining_features.append(col)
    
    print(f"\n6. 标准化剩余特征 ({len(remaining_features)} 个)")
    for feature in remaining_features:
        if feature in df_processed.columns:
            col_data = df_processed[feature].copy()
            
            # 检查范围
            if col_data.max() > 1 or col_data.min() < 0:
                scaler = MinMaxScaler()
                col_data_scaled = scaler.fit_transform(col_data.values.reshape(-1, 1)).flatten()
                df_processed[feature] = col_data_scaled
                print(f"  {feature}: 标准化到[0,1]")
    
    print("\n=== 最终验证 ===")
    # 验证所有数值特征的范围
    extreme_count = 0
    for col in numeric_cols:
        if col != 'account' and col in df_processed.columns:
            col_data = df_processed[col].dropna()
            if len(col_data) > 0:
                max_val = col_data.max()
                min_val = col_data.min()
                if abs(max_val) > 10 or abs(min_val) < -1:
                    print(f"⚠️  {col}: [{min_val:.6f}, {max_val:.6f}] - 仍有异常值")
                    extreme_count += 1
    
    if extreme_count == 0:
        print("✅ 所有特征都已正确标准化")
    else:
        print(f"⚠️  仍有 {extreme_count} 个特征存在异常值")
    
    # 保存结果
    output_file = "/Users/mannormal/4011/Qi Zihan/v2/feature_extraction/result/features_normalized.csv"
    df_processed.to_csv(output_file, index=False)
    
    print(f"\n=== 处理完成 ===")
    print(f"处理前特征数量: {df.shape[1]-1}")
    print(f"处理后特征数量: {df_processed.shape[1]-1}")
    print(f"数据保存到: {output_file}")
    
    # 生成处理报告
    generate_processing_report(df, df_processed, output_file)
    
    return df_processed

def generate_processing_report(df_original, df_processed, output_file):
    """生成处理报告"""
    report_file = output_file.replace('.csv', '_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("特征标准化处理报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"原始数据: {df_original.shape[0]} 行, {df_original.shape[1]} 列\n")
        f.write(f"处理后数据: {df_processed.shape[0]} 行, {df_processed.shape[1]} 列\n\n")
        
        numeric_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
        if 'account' in numeric_cols:
            numeric_cols.remove('account')
        
        f.write("数值特征范围对比:\n")
        f.write("-" * 50 + "\n")
        
        for col in numeric_cols:
            if col in df_processed.columns:
                orig_data = df_original[col].dropna()
                proc_data = df_processed[col].dropna()
                
                if len(orig_data) > 0 and len(proc_data) > 0:
                    f.write(f"{col}:\n")
                    f.write(f"  原始: [{orig_data.min():.2e}, {orig_data.max():.2e}]\n")
                    f.write(f"  处理后: [{proc_data.min():.6f}, {proc_data.max():.6f}]\n\n")
    
    print(f"处理报告保存到: {report_file}")

if __name__ == "__main__":
    try:
        result_df = comprehensive_feature_normalization()
        if result_df is not None:
            print("✅ 特征标准化处理成功完成！")
        else:
            print("❌ 处理失败")
            
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()