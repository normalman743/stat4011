import pandas as pd
import numpy as np

def process_transaction_data():
    """专门处理这个交易数据文件"""
    print("Loading transaction data...")
    df = pd.read_csv("/Users/mannormal/4011/Qi Zihan/original_data/transactions.csv")
    print(f"Loaded {len(df)} transactions")
    
    # 转换value列为数值
    df['value'] = pd.to_numeric(df['value'], errors='coerce').fillna(0)
    
    # 分析三个数值列
    cols = ['value', 'gas', 'gas_price']
    print("\n=== 原始数据范围 ===")
    for col in cols:
        print(f"{col}: {df[col].min():.0f} - {df[col].max():.0f}")
    
    # 应用对数变换 (推荐方法)
    print("\n=== 应用对数变换 ===")
    for col in cols:
        df[f'{col}_log'] = np.log1p(df[col])
        print(f"{col}_log: {df[f'{col}_log'].min():.4f} - {df[f'{col}_log'].max():.4f}")
    
    # 保存处理后的数据
    output_file = "/Users/mannormal/4011/Qi Zihan/original_data/transactions_scaled.csv"
    df.to_csv(output_file, index=False)
    print(f"\n处理后数据保存到: {output_file}")

    # 显示缩放效果对比
    print("\n=== 缩放效果对比 ===")
    for col in cols:
        original_range = df[col].max() - df[col].min()
        scaled_range = df[f'{col}_log'].max() - df[f'{col}_log'].min()
        print(f"{col}: 原始范围 {original_range:.0e} -> 对数范围 {scaled_range:.2f}")

if __name__ == "__main__":
    process_transaction_data()