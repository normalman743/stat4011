#!/usr/bin/env python3
"""
脚本用于删除基于flag计算的特征列，防止数据泄漏
基于parallel_feature_extraction.py分析结果
"""

import pandas as pd
import os
from glob import glob

def drop_flag_based_features():
    """
    删除基于flag值计算的特征列
    这些特征在parallel_feature_extraction.py中通过flag=-1,0,1计算得出
    """
    
    # 基于代码分析，这些特征是通过flag值计算的
    flag_based_features = [
        # Forward path features based on flag values
        'normal_fprofit',    # flag = -1
        'normal_fpprofit', 
        'normal_fsize',
        'abnormal_fprofit',  # flag = 0  
        'abnormal_fpprofit',
        'abnormal_fsize',
        'bad_fprofit',       # flag = 1
        'bad_fpprofit',
        'bad_fsize',
        
        # Backward path features based on flag values  
        'normal_bprofit',    # flag = -1
        'normal_bpprofit',
        'normal_bsize',
        'abnormal_bprofit',  # flag = 0
        'abnormal_bpprofit', 
        'abnormal_bsize',
        'bad_bprofit',       # flag = 1
        'bad_bpprofit',
        'bad_bsize'
    ]
    
    print("标识出的基于flag计算的特征:", flag_based_features)
    print(f"总共 {len(flag_based_features)} 个特征需要删除")
    
    return flag_based_features

def process_csv_files(input_dir, output_dir):
    """
    处理指定目录下所有包含all_feature相关的CSV文件
    """
    # 查找所有相关CSV文件
    pattern = os.path.join(input_dir, "*all_feature*.csv")
    csv_files = glob(pattern)
    
    print(f"在 {input_dir} 找到 {len(csv_files)} 个CSV文件:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    if not csv_files:
        print("未找到匹配的CSV文件!")
        return
    
    flag_features = drop_flag_based_features()
    
    # 处理每个文件
    for csv_file in csv_files:
        print(f"\n处理文件: {os.path.basename(csv_file)}")
        
        try:
            # 读取CSV
            df = pd.read_csv(csv_file)
            original_shape = df.shape
            print(f"原始形状: {original_shape}")
            
            # 检查哪些flag特征存在
            existing_flag_features = [col for col in flag_features if col in df.columns]
            missing_flag_features = [col for col in flag_features if col not in df.columns]
            
            if existing_flag_features:
                print(f"找到 {len(existing_flag_features)} 个flag特征:")
                for feature in existing_flag_features:
                    print(f"  - {feature}")
                
                # 删除flag特征
                df_cleaned = df.drop(columns=existing_flag_features)
                new_shape = df_cleaned.shape
                print(f"清理后形状: {new_shape}")
                print(f"删除了 {original_shape[1] - new_shape[1]} 列")
                
                # 保存清理后的文件到 output_dir
                base_name = os.path.splitext(os.path.basename(csv_file))[0]
                output_file = os.path.join(output_dir, f"{base_name}_no_flag_features.csv")
                # 确保输出目录存在
                os.makedirs(output_dir, exist_ok=True)
                df_cleaned.to_csv(output_file, index=False)
                print(f"保存到: {output_file}")
                
            else:
                print("未找到任何flag特征列")
            
            if missing_flag_features:
                print(f"缺失的flag特征 ({len(missing_flag_features)}):")
                for feature in missing_flag_features[:5]:  # 只显示前5个
                    print(f"  - {feature}")
                if len(missing_flag_features) > 5:
                    print(f"  - ... 还有 {len(missing_flag_features)-5} 个")
        
        except Exception as e:
            print(f"处理 {csv_file} 时出错: {e}")

def main():
    """主函数"""
    print("=== 删除基于Flag计算的特征 ===")
    print("基于 parallel_feature_extraction.py 分析结果\n")
    
    # 目标目录
    input_dir = "/Users/mannormal/4011/Qi Zihan/v1/feature_extraction/generated_features"
    output_dir = "/Users/mannormal/4011/Qi Zihan/v2/feature_extraction/result"
    # 检查目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 目录不存在 {input_dir}")
        return
    
    # 处理CSV文件
    process_csv_files(input_dir, output_dir)

    print("\n=== 处理完成 ===")
    print("所有清理后的文件都添加了 '_no_flag_features' 后缀")

if __name__ == "__main__":
    main()