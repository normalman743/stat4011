import csv

def count_unique_accounts_csv(csv_file):
    """
    使用标准库处理CSV文件
    """
    unique_accounts = set()
    
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            # 添加from_account（如果不为空）
            if row['from_account'].strip():
                unique_accounts.add(row['from_account'].strip())
            
            # 添加to_account（如果不为空）
            if row['to_account'].strip():
                unique_accounts.add(row['to_account'].strip())
    
    return len(unique_accounts), sorted(list(unique_accounts))

# 使用
csv_file = 'Qi Zihan/original_data/transactions.csv'  # 替换为你的CSV文件名
count, accounts = count_unique_accounts_csv(csv_file)

print(f"唯一账户数量: {count}")
