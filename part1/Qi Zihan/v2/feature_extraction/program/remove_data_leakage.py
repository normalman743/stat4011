import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

def remove_data_leakage_features():
    """
    删除数据泄露特征、重复列和零值列
    """
    print("=== 删除数据泄露特征 ===")
    
    # 读取特征文件
    input_file = "/Users/mannormal/4011/Qi Zihan/v2/feature_extraction/result/features_normalized.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return None
        
    df = pd.read_csv(input_file)
    original_shape = df.shape
    print(f"原始数据: {original_shape[0]} 行, {original_shape[1]} 列")
    
    # 1. 删除明确的数据泄露特征
    print("\n=== 1. 删除明确的数据泄露特征 ===")
    immediate_leakage_features = [
        'is_train',
        'is_test', 
        'neighbor_bad_ratio_train_only',
        'train_neighbor_ratio'
    ]
    
    # 检查哪些泄露特征实际存在
    existing_leakage = [col for col in immediate_leakage_features if col in df.columns]
    print(f"检测到的泄露特征: {existing_leakage}")
    
    # 删除泄露特征
    df_clean = df.drop(columns=existing_leakage, errors='ignore')
    print(f"删除 {len(existing_leakage)} 个泄露特征后: {df_clean.shape[1]} 列")
    
    # 2. 检查并删除重复列
    print("\n=== 2. 检查重复列 ===")
    
    # 获取数值列（排除account列）
    numeric_cols = [col for col in df_clean.columns if col != 'account']
    duplicate_pairs = []
    columns_to_drop = set()
    
    print("检查列之间的相关性...")
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols[i+1:], i+1):
            try:
                # 检查是否完全相同
                if df_clean[col1].equals(df_clean[col2]):
                    duplicate_pairs.append((col1, col2))
                    columns_to_drop.add(col2)  # 保留第一个，删除第二个
                    print(f"  发现完全重复列: {col1} == {col2}")
                # 检查高度相关（相关系数 > 0.999）
                elif not df_clean[col1].isna().all() and not df_clean[col2].isna().all():
                    corr = df_clean[[col1, col2]].corr().iloc[0, 1]
                    if abs(corr) > 0.999 and not np.isnan(corr):
                        duplicate_pairs.append((col1, col2))
                        columns_to_drop.add(col2)
                        print(f"  发现高度相关列: {col1} ≈ {col2} (相关系数: {corr:.6f})")
            except:
                continue
    
    if columns_to_drop:
        df_clean = df_clean.drop(columns=list(columns_to_drop))
        print(f"删除 {len(columns_to_drop)} 个重复/高度相关列: {list(columns_to_drop)}")
    else:
        print("未发现重复列")
    
    print(f"删除重复列后: {df_clean.shape[1]} 列")
    
    # 3. 检查并删除零值列
    print("\n=== 3. 检查零值列 ===")
    
    zero_value_cols = []
    near_zero_cols = []
    
    for col in numeric_cols:
        if col in df_clean.columns:
            col_data = df_clean[col].dropna()
            if len(col_data) == 0:
                zero_value_cols.append(col)
                print(f"  全部为空值: {col}")
            elif (col_data == 0).all():
                zero_value_cols.append(col)
                print(f"  全部为零值: {col}")
            elif (col_data == 0).sum() / len(col_data) > 0.95:
                near_zero_cols.append(col)
                zero_ratio = (col_data == 0).sum() / len(col_data)
                print(f"  近零列 ({zero_ratio:.1%}为零): {col}")
    
    # 删除零值列
    if zero_value_cols:
        df_clean = df_clean.drop(columns=zero_value_cols)
        print(f"删除 {len(zero_value_cols)} 个零值列: {zero_value_cols}")
    else:
        print("未发现零值列")
    
    # 对于近零列，询问是否删除
    if near_zero_cols:
        print(f"发现 {len(near_zero_cols)} 个近零列（>95%为零），建议删除以减少噪声")
        # 自动删除近零列
        df_clean = df_clean.drop(columns=near_zero_cols)
        print(f"删除 {len(near_zero_cols)} 个近零列: {near_zero_cols}")
    
    print(f"删除零值列后: {df_clean.shape[1]} 列")
    
    # 4. 检查方差过低的特征
    print("\n=== 4. 检查低方差特征 ===")
    
    low_variance_cols = []
    variance_threshold = 1e-6
    
    for col in numeric_cols:
        if col in df_clean.columns:
            col_data = df_clean[col].dropna()
            if len(col_data) > 1:
                variance = col_data.var()
                if variance < variance_threshold:
                    low_variance_cols.append(col)
                    print(f"  低方差特征: {col} (方差: {variance:.2e})")
    
    if low_variance_cols:
        df_clean = df_clean.drop(columns=low_variance_cols)
        print(f"删除 {len(low_variance_cols)} 个低方差特征")
    else:
        print("未发现低方差特征")
    
    print(f"删除低方差特征后: {df_clean.shape[1]} 列")
    
    # 5. 最终检查和统计
    print("\n=== 5. 最终统计 ===")
    
    final_shape = df_clean.shape
    removed_cols = original_shape[1] - final_shape[1]
    
    print(f"原始特征数: {original_shape[1]}")
    print(f"最终特征数: {final_shape[1]}")
    print(f"删除特征数: {removed_cols}")
    print(f"删除比例: {removed_cols/original_shape[1]*100:.1f}%")
    
    # 按类别统计删除的特征
    total_removed = len(existing_leakage) + len(columns_to_drop) + len(zero_value_cols) + len(near_zero_cols) + len(low_variance_cols)
    print(f"\n删除特征详情:")
    print(f"  数据泄露特征: {len(existing_leakage)} 个")
    print(f"  重复/高相关特征: {len(columns_to_drop)} 个")
    print(f"  零值特征: {len(zero_value_cols)} 个")
    print(f"  近零特征: {len(near_zero_cols)} 个")
    print(f"  低方差特征: {len(low_variance_cols)} 个")
    print(f"  总计: {total_removed} 个")
    
    # 保存清理后的数据
    output_file = "/Users/mannormal/4011/Qi Zihan/v2/feature_extraction/result/features_cleaned_no_leakage1.csv"
    df_clean.to_csv(output_file, index=False)
    
    print(f"\n✅ 数据清理完成！")
    print(f"清理后数据保存到: {output_file}")
    
    # 显示剩余的特征列表
    remaining_features = [col for col in df_clean.columns if col != 'account']
    print(f"\n剩余特征 ({len(remaining_features)} 个):")
    for i, col in enumerate(remaining_features, 1):
        print(f"{i:2d}. {col}")
    
    # 生成清理报告
    generate_cleaning_report(
        original_shape, final_shape, 
        existing_leakage, columns_to_drop, zero_value_cols, 
        near_zero_cols, low_variance_cols,
        output_file
    )
    
    return df_clean

def generate_cleaning_report(original_shape, final_shape, leakage_cols, 
                           duplicate_cols, zero_cols, near_zero_cols, low_var_cols, output_file):
    """生成详细的清理报告"""
    
    report_file = output_file.replace('.csv', '_cleaning_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("数据清理报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"清理前数据: {original_shape[0]} 行, {original_shape[1]} 列\n")
        f.write(f"清理后数据: {final_shape[0]} 行, {final_shape[1]} 列\n")
        f.write(f"删除特征数: {original_shape[1] - final_shape[1]} 个\n\n")
        
        f.write("删除的特征详情:\n")
        f.write("-" * 30 + "\n")
        
        f.write(f"\n1. 数据泄露特征 ({len(leakage_cols)} 个):\n")
        for col in leakage_cols:
            f.write(f"   - {col}\n")
        
        f.write(f"\n2. 重复/高相关特征 ({len(duplicate_cols)} 个):\n")
        for col in duplicate_cols:
            f.write(f"   - {col}\n")
        
        f.write(f"\n3. 零值特征 ({len(zero_cols)} 个):\n")
        for col in zero_cols:
            f.write(f"   - {col}\n")
        
        f.write(f"\n4. 近零特征 ({len(near_zero_cols)} 个):\n")
        for col in near_zero_cols:
            f.write(f"   - {col}\n")
        
        f.write(f"\n5. 低方差特征 ({len(low_var_cols)} 个):\n")
        for col in low_var_cols:
            f.write(f"   - {col}\n")
    
    print(f"清理报告保存到: {report_file}")

if __name__ == "__main__":
    try:
        result_df = remove_data_leakage_features()
        if result_df is not None:
            print("\n🎉 数据泄露特征清理成功完成！")
            print("现在可以用清理后的数据重新训练模型了。")
        else:
            print("❌ 清理失败")
            
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()