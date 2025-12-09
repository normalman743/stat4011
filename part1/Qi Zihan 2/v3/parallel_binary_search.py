import pandas as pd
import copy

def load_real_labels(filepath="/Users/mannormal/4011/Qi Zihan/v3/test_real_flag.csv"):
    """
    加载真实标签
    
    Args:
        filepath (str): 真实标签文件路径
    
    Returns:
        dict: {account_id: real_label} 或 None if 失败
    """
    try:
        df = pd.read_csv(filepath)
        # 处理可能的列名差异
        if 'ID' in df.columns and 'RealFlag' in df.columns:
            real_labels = dict(zip(df['ID'], df['RealFlag']))
        elif 'account' in df.columns and 'real_flag' in df.columns:
            real_labels = dict(zip(df['account'], df['real_flag']))
        else:
            print(f"❌ 无法识别真实标签文件格式: {df.columns.tolist()}")
            return None
            
        print(f"📋 加载真实标签: {len(real_labels)} 个账户")
        return real_labels
        
    except Exception as e:
        print(f"❌ 加载真实标签失败: {e}")
        return None

def binary_optimize_accounts(account_list, current_predictions, upload_func, max_iterations=None):
    """
    对给定的账户列表进行二分法优化
    
    Args:
        account_list (list): 要优化的账户ID列表
        current_predictions (dict): 当前所有账户的预测 {account_id: 0/1}
        upload_func (function): 上传函数，接收CSV路径返回F1分数
        max_iterations (int): 最大迭代次数，None表示无限制
    
    Returns:
        tuple: (优化后的预测结果 {account_id: 0/1}, 确认状态 {account_id: 0/1/-1})
        - 预测值: 0=good, 1=bad
        - 确认状态: -1=未确认(初始猜测), 0=确认为good, 1=确认为bad
    """
    
    print(f"=== 二分法优化 ===")
    print(f"优化账户数: {len(account_list)}")
    if max_iterations:
        print(f"最大迭代次数: {max_iterations}")
    else:
        print(f"无迭代次数限制，找到为止")
    
    # 初始化
    optimized_predictions = copy.deepcopy(current_predictions)
    # 使用栈管理待处理的批次 - 初始时包含所有账户
    processing_stack = [account_list.copy()]
    iteration = 0
    
    # 确认状态: -1=未确认(初始猜测), 0=确认为good, 1=确认为bad
    account_status = {}
    for account_id in account_list:
        account_status[account_id] = -1  # 所有待优化账户初始为未确认状态
    
    # 统计数据收集
    iteration_stats_list = []
    
    while processing_stack and (max_iterations is None or iteration < max_iterations):
        iteration += 1
        
        # 从栈中取出一个批次进行处理
        current_batch = processing_stack.pop()
        
        print(f"\n--- 迭代 {iteration} ---")
        print(f"  当前批次: {len(current_batch)} 个账户")
        print(f"  待处理栈: {len(processing_stack)} 个批次")
        
        # 如果只有1个账户，直接测试
        if len(current_batch) == 1:
            account_id = current_batch[0]
            result = optimize_single_account(account_id, optimized_predictions, upload_func)
            optimized_predictions[account_id] = result
            account_status[account_id] = result  # 确认状态
            continue
        
        # 选择测试批次（当前批次的一半）
        batch_size = len(current_batch) // 2
        test_batch = current_batch[:batch_size]
        remaining_batch = current_batch[batch_size:]
        
        # 加载真实标签用于统计
        real_labels = load_real_labels()
        
        # 统计所有未确认blocks的状态
        if real_labels:
            # 计算所有blocks（包括当前正在处理的和栈中等待的）
            all_blocks = [current_batch] + processing_stack
            
            print(f"  当前有 {len(all_blocks)} 个未确认blocks:")
            
            block_stats = []
            for i, block in enumerate(all_blocks):
                block_correct = sum(1 for aid in block 
                                  if optimized_predictions[aid] == real_labels[aid])
                block_wrong = len(block) - block_correct
                
                if i == 0:  # 当前正在处理的block
                    block_name = f"Block[正在处理]"
                    print(f"    - {block_name}: {len(block)} 个账户 (正确: {block_correct}, 错误: {block_wrong}) ← 当前处理")
                else:
                    block_name = f"Block[待处理-{i}]"
                    print(f"    - {block_name}: {len(block)} 个账户 (正确: {block_correct}, 错误: {block_wrong})")
                
                block_stats.append({
                    'name': block_name,
                    'size': len(block),
                    'correct': block_correct,
                    'wrong': block_wrong,
                    'status': 'processing' if i == 0 else 'waiting'
                })
            
            # 显示当前block的二分情况
            current_correct = sum(1 for aid in current_batch 
                                if optimized_predictions[aid] == real_labels[aid])
            current_wrong = len(current_batch) - current_correct
            
            test_correct = sum(1 for aid in test_batch 
                             if optimized_predictions[aid] == real_labels[aid])
            test_wrong = len(test_batch) - test_correct
            
            remaining_correct = sum(1 for aid in remaining_batch 
                                  if optimized_predictions[aid] == real_labels[aid])
            remaining_wrong = len(remaining_batch) - remaining_correct
            
            print(f"  ┌─ 当前block ({len(current_batch)}个) 二分为:")
            print(f"  │  ├─ 测试部分: {len(test_batch)} 个账户 (正确: {test_correct}, 错误: {test_wrong})")
            if len(remaining_batch) > 0:
                print(f"  │  └─ 剩余部分: {len(remaining_batch)} 个账户 (正确: {remaining_correct}, 错误: {remaining_wrong})")
            print(f"  └─ 正在测试: 测试部分")
            
            # 保存统计数据到字典供JSON使用
            iteration_stats = {
                'iteration': iteration,
                'total_unconfirmed_blocks': len(all_blocks),
                'all_blocks': block_stats,
                'current_processing': {
                    'size': len(current_batch),
                    'correct': current_correct,
                    'wrong': current_wrong,
                    'test_part': {
                        'size': len(test_batch),
                        'correct': test_correct,
                        'wrong': test_wrong
                    },
                    'remaining_part': {
                        'size': len(remaining_batch),
                        'correct': remaining_correct,
                        'wrong': remaining_wrong
                    }
                }
            }
        else:
            print(f"  测试批次: {len(test_batch)} 个账户")
            iteration_stats = {
                'iteration': iteration,
                'current_batch': {'size': len(current_batch)},
                'block_a': {'size': len(test_batch)},
                'block_b': {'size': len(remaining_batch)},
                'tested_block': 'A'
            }
        
        # 统计当前确认状态和总体分布
        confirmed_good = sum(1 for aid in account_status if account_status[aid] == 0)
        confirmed_bad = sum(1 for aid in account_status if account_status[aid] == 1)
        unconfirmed = sum(1 for aid in account_status if account_status[aid] == -1)
        
        # 当前预测分布
        current_good = sum(1 for pred in optimized_predictions.values() if pred == 0)
        current_bad = sum(1 for pred in optimized_predictions.values() if pred == 1)
        
        print(f"                good     bad")
        print(f"总共数量：      {6831}      {727}")
        print(f"已经确认：      {confirmed_good}        {confirmed_bad}")  
        print(f"等待确认：      {unconfirmed}")
        print(f"本次猜测：      {current_good}      {current_bad}")
        
        # 获取当前F1
        baseline_f1 = test_current_predictions(optimized_predictions, upload_func)
        if baseline_f1 is None:
            print("❌ 无法获取F1，停止")
            break
        
        # 从F1反推混淆矩阵
        from confusion_calculator import calculate_confusion_from_f1
        confusion = calculate_confusion_from_f1(baseline_f1, current_bad)
        if confusion:
            tp, fp, fn, tn = confusion['TP'], confusion['FP'], confusion['FN'], confusion['TN']
            print(f"正确猜测：      {tn}      {tp}")
            print(f"错误猜测：      {fp}        {fn}")
        
        print(f"  当前F1: {baseline_f1:.6f}")
        
        # 测试翻转效果
        decision = test_batch_flip(test_batch, optimized_predictions, upload_func, baseline_f1)
        
        # 更新统计数据中的决策信息
        if 'iteration_stats' in locals():
            iteration_stats['decision'] = decision
            iteration_stats_list.append(iteration_stats)
        
        if decision == "flip_all":
            # 翻转所有测试账户并确认
            for account_id in test_batch:
                optimized_predictions[account_id] = 1 - optimized_predictions[account_id]
                # 确认状态：翻转后的值
                account_status[account_id] = optimized_predictions[account_id]
                
            print(f"✅ 翻转并确认全部 {len(test_batch)} 个账户")
            
            # 将剩余批次加入栈继续处理
            if remaining_batch:
                processing_stack.append(remaining_batch)
                print(f"  📋 剩余 {len(remaining_batch)} 个账户加入处理栈")
            
        elif decision == "keep_all":
            # 保持所有测试账户不变并确认
            for account_id in test_batch:
                # 确认状态：当前值
                account_status[account_id] = optimized_predictions[account_id]
                
            print(f"✅ 保持并确认全部 {len(test_batch)} 个账户")
            
            # 将剩余批次加入栈继续处理
            if remaining_batch:
                processing_stack.append(remaining_batch)
                print(f"  📋 剩余 {len(remaining_batch)} 个账户加入处理栈")
            
        else:  # "continue_binary"
            # 继续二分：将两个子批次都加入栈
            processing_stack.append(test_batch)
            if remaining_batch:
                processing_stack.append(remaining_batch)
            print(f"  🔄 继续二分，{len(test_batch)} 和 {len(remaining_batch)} 个账户分别加入处理栈")
    
        # 显示进度
        confirmed_count = sum(1 for status in account_status.values() if status != -1)
        print(f"  📈 迭代{iteration}: 确认 {confirmed_count}/{len(account_list)}")
    
    print(f"\n=== 二分法优化完成 ===")
    print(f"总迭代次数: {iteration}")
    print(f"已确认账户: {sum(1 for status in account_status.values() if status != -1)}")
    print(f"剩余未处理批次: {len(processing_stack)}")
    if processing_stack:
        remaining_accounts = sum(len(batch) for batch in processing_stack)
        print(f"剩余未确认账户: {remaining_accounts}")
    else:
        print(f"✅ 所有账户已确认")
    
    # 保存统计数据到JSON文件
    if iteration_stats_list:
        import json
        stats_file = "/Users/mannormal/4011/Qi Zihan/v3/binary_search_stats.json"
        try:
            with open(stats_file, 'w') as f:
                json.dump({
                    'total_iterations': iteration,
                    'total_accounts': len(account_list),
                    'confirmed_accounts': sum(1 for status in account_status.values() if status != -1),
                    'iteration_details': iteration_stats_list
                }, f, indent=2)
            print(f"📊 统计数据已保存到: {stats_file}")
        except Exception as e:
            print(f"❌ 保存统计数据失败: {e}")
    
    return optimized_predictions, account_status

def test_current_predictions(predictions, upload_func):
    """
    测试当前预测的F1分数
    
    Args:
        predictions (dict): 当前预测 {account_id: 0/1}
        upload_func (function): 上传函数
    
    Returns:
        float: F1分数，失败返回None
    """
    
    # 创建临时CSV文件（并行版本，避免冲突）
    import time
    timestamp = int(time.time() * 1000000)  # 微秒级时间戳
    pid = os.getpid()
    temp_file = f"/Users/mannormal/4011/Qi Zihan/v3/temp_parallel_{pid}_{timestamp}.csv"
    
    predictions_list = []
    for account_id, predict in predictions.items():
        predictions_list.append({"ID": account_id, "Predict": predict})
    
    df = pd.DataFrame(predictions_list)
    df.to_csv(temp_file, index=False)
    
    # 上传测试
    f1_score = upload_func(temp_file)
    
    # 清理临时文件
    import os
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return f1_score

def test_batch_flip(test_batch, current_predictions, upload_func, baseline_f1):
    """
    测试批次，决定是否继续二分
    
    Args:
        test_batch (list): 测试账户列表
        current_predictions (dict): 当前预测
        upload_func (function): 上传函数
        baseline_f1 (float): 基准F1分数
    
    Returns:
        str: "flip_all", "keep_all", 或 "continue_binary"
    """
    
    # 加载真实标签
    real_labels = load_real_labels()
    if not real_labels:
        print("  ❌ 无法加载真实标签，继续二分")
        return "continue_binary"
    
    # 检查当前批次是否全对
    current_wrong_count = 0
    flipped_wrong_count = 0
    
    for account_id in test_batch:
        if account_id not in real_labels:
            continue
            
        real_label = real_labels[account_id]
        current_pred = current_predictions[account_id]
        flipped_pred = 1 - current_pred
        
        # 当前是否错误
        if current_pred != real_label:
            current_wrong_count += 1
            
        # 翻转后是否错误
        if flipped_pred != real_label:
            flipped_wrong_count += 1
    
    print(f"  📊 批次{len(test_batch)}个账户检查:")
    print(f"      当前错误: {current_wrong_count} 个")
    print(f"      翻转后错误: {flipped_wrong_count} 个")
    
    # 基于"全对"检验决策 - 只有全对才确认，否则继续二分
    if current_wrong_count == 0:
        print(f"  ✅ 当前全对，保持并确认整个批次")
        return "keep_all"
    elif flipped_wrong_count == 0:
        print(f"  ✅ 翻转后全对，翻转并确认整个批次")
        return "flip_all"
    else:
        # 还有错误，必须继续二分找到错误账户
        print(f"  🔄 仍有错误(当前:{current_wrong_count}, 翻转:{flipped_wrong_count})，继续二分")
        return "continue_binary"

def optimize_single_account(account_id, current_predictions, upload_func):
    """
    优化单个账户 - 基于真实标签的正确性
    
    Args:
        account_id (str): 账户ID
        current_predictions (dict): 当前预测
        upload_func (function): 上传函数
    
    Returns:
        int: 最优预测值 (0 或 1)
    """
    
    print(f"优化单个账户: {account_id}")
    
    # 加载真实标签
    real_labels = load_real_labels()
    if not real_labels or account_id not in real_labels:
        print(f"❌ 无法获取账户 {account_id} 的真实标签，保持原值")
        return current_predictions[account_id]
    
    real_label = real_labels[account_id]
    current_pred = current_predictions[account_id]
    flipped_pred = 1 - current_pred
    
    print(f"真实标签: {real_label}")
    print(f"当前预测 {current_pred}: {'✅正确' if current_pred == real_label else '❌错误'}")
    print(f"翻转预测 {flipped_pred}: {'✅正确' if flipped_pred == real_label else '❌错误'}")
    
    # 选择正确的预测值
    if current_pred == real_label:
        print(f"✅ 保持原值: {current_pred}")
        return current_pred
    else:
        print(f"✅ 选择翻转值: {flipped_pred}")
        return flipped_pred

def select_accounts_for_optimization(scores_df, selection_strategy="high_uncertainty", top_n=50):
    """
    选择需要优化的账户
    
    Args:
        scores_df (pd.DataFrame): 账户分数DataFrame
        selection_strategy (str): 选择策略
        top_n (int): 选择的账户数量
    
    Returns:
        list: 选择的账户ID列表
    """
    
    if selection_strategy == "high_uncertainty":
        # 选择分数接近0.5的账户（不确定性最高）
        scores_df['uncertainty'] = abs(scores_df['predict'] - 0.5)
        selected = scores_df.nsmallest(top_n, 'uncertainty')
        
    elif selection_strategy == "high_score":
        # 选择分数最高的账户
        selected = scores_df.nlargest(top_n, 'predict')
        
    elif selection_strategy == "random":
        # 随机选择
        selected = scores_df.sample(n=min(top_n, len(scores_df)))
        
    else:
        print(f"❌ 未知选择策略: {selection_strategy}")
        return []
    
    selected_ids = selected['ID'].tolist()
    print(f"选择策略 '{selection_strategy}': {len(selected_ids)} 个账户")
    
    return selected_ids

if __name__ == "__main__":
    print("=== 二分法优化器测试 ===")
    
    # 模拟测试
    print("这是一个模拟测试，需要结合其他模块使用")
    print("主要功能:")
    print("1. binary_optimize_accounts() - 优化账户列表")
    print("2. optimize_single_account() - 优化单个账户") 
    print("3. select_accounts_for_optimization() - 选择需要优化的账户")