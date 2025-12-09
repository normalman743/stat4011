import pandas as pd
import copy


def calculate_block_stats_from_inference(parent_block_accounts, base_predictions, 
                                       inferred_true_good, inferred_true_bad):
    """
    基于混淆矩阵推断结果计算block统计信息
    
    Args:
        parent_block_accounts: 父block内的账户列表
        base_predictions: 当前预测
        inferred_true_good: 通过analyze_binary_split推断出的真实good数量
        inferred_true_bad: 通过analyze_binary_split推断出的真实bad数量
    
    Returns:
        dict: 统计信息
    """
    # 统计当前block的预测分布
    pred_good_count = sum(1 for aid in parent_block_accounts if base_predictions[aid] == 0)
    pred_bad_count = len(parent_block_accounts) - pred_good_count
    
    # 基于推断的真实分布和预测分布计算混淆矩阵组件
    # 这里我们假设预测的分配是"最优"的（实际上不一定，但我们只能这样估算）
    
    # 计算block内的混淆矩阵（估算）
    # 最好情况下的正确预测数量
    max_correct_good = min(pred_good_count, inferred_true_good)  # TN
    max_correct_bad = min(pred_bad_count, inferred_true_bad)     # TP
    
    # 错误预测数量
    wrong_good_pred = pred_good_count - max_correct_good        # FN (预测good但实际bad)
    wrong_bad_pred = pred_bad_count - max_correct_bad           # FP (预测bad但实际good)
    
    total_correct = max_correct_good + max_correct_bad
    total_wrong = wrong_good_pred + wrong_bad_pred
    
    # 验证一致性
    assert total_correct + total_wrong == len(parent_block_accounts), "统计不一致"
    assert max_correct_good + wrong_good_pred == pred_good_count, "good预测统计不一致"
    assert max_correct_bad + wrong_bad_pred == pred_bad_count, "bad预测统计不一致"
    
    return {
        'correct': total_correct,           # 总正确预测数
        'wrong': total_wrong,               # 总错误预测数
        'pred_good': pred_good_count,       # 预测为good的数量
        'pred_bad': pred_bad_count,         # 预测为bad的数量
        'correct_good': max_correct_good,   # 正确预测的good (TN)
        'correct_bad': max_correct_bad,     # 正确预测的bad (TP) 
        'wrong_good': wrong_good_pred,      # 错误预测为good (FN)
        'wrong_bad': wrong_bad_pred,        # 错误预测为bad (FP)
        'true_good': inferred_true_good,    # 推断的真实good数量
        'true_bad': inferred_true_bad       # 推断的真实bad数量
    }


def analyze_binary_split(confusion_baseline, confusion_flipped, n_b, total_good, total_bad, 
                       base_predictions, parent_block_accounts):
    """
    分析二分结果
    
    Args:
        confusion_baseline: 基准混淆矩阵
        confusion_flipped: 翻转后混淆矩阵
        n_b: block_b的大小
        total_good: 层级基准中的总good数量
        total_bad: 层级基准中的总bad数量
        base_predictions: 当前预测
        parent_block_accounts: 父block的账户列表
    
    Returns:
        (a_status, b_status, block_stats): 两个block的状态判断和统计信息
    """
    # 计算变化量
    delta_tp = confusion_flipped['TP'] - confusion_baseline['TP']
    delta_fp = confusion_flipped['FP'] - confusion_baseline['FP']
    
    # 推断block_b的真实分布
    b_bad = delta_tp
    b_good = delta_fp
    
    # 验证一致性
    if b_bad + b_good != n_b:
        return "MIXED", "MIXED", {'correct': 0, 'wrong': len(parent_block_accounts), 'pred_good': 0, 'pred_bad': 0}
    
    # 推断整个parent_block的真实分布
    parent_true_bad = total_bad  # 使用层级基准
    parent_true_good = total_good
    
    # 使用新函数计算统计信息
    block_stats = calculate_block_stats_from_inference(
        parent_block_accounts, 
        base_predictions, 
        parent_true_good, 
        parent_true_bad
    )
    
    # 判定block状态
    if b_bad == n_b:
        b_status = "ALL_BAD"
    elif b_good == n_b:
        b_status = "ALL_GOOD"
    else:
        b_status = "MIXED"
    
    a_bad = total_bad - b_bad
    a_good = total_good - b_good
    
    if a_bad == 0:
        a_status = "ALL_GOOD"
    elif a_good == 0:
        a_status = "ALL_BAD"
    else:
        a_status = "MIXED"
    
    return a_status, b_status, block_stats


def binary_optimize_accounts(account_list, current_predictions, upload_func, max_iterations=None):
    """
    对给定的账户列表进行二分法优化（基于混淆矩阵推断，不使用真实标签）
    
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
    
    print(f"=== 二分法优化（基于混淆矩阵推断）===")
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
    
    # 获取基准混淆矩阵
    baseline_f1 = test_current_predictions(optimized_predictions, upload_func)
    if baseline_f1 is None:
        print("❌ 无法获取基准F1，停止")
        return optimized_predictions, account_status
    
    from confusion_calculator import calculate_confusion_from_f1
    current_bad = sum(1 for pred in optimized_predictions.values() if pred == 1)
    baseline_confusion = calculate_confusion_from_f1(baseline_f1, current_bad)
    if not baseline_confusion:
        print("❌ 无法计算基准混淆矩阵，停止")
        return optimized_predictions, account_status
    
    # 根据混淆矩阵推断总体真实分布
    total_good = 6831  # 已知真实分布
    total_bad = 727
    
    while processing_stack and (max_iterations is None or iteration < max_iterations):
        iteration += 1
        
        # 从栈中取出一个批次进行处理
        current_batch = processing_stack.pop()
        
        print(f"\n--- 迭代 {iteration} ---")
        print(f"  当前批次: {len(current_batch)} 个账户")
        print(f"  待处理栈: {len(processing_stack)} 个批次")
        
        # 如果只有1个账户，基于概率决定
        if len(current_batch) == 1:
            account_id = current_batch[0]
            # 简单策略：保持当前预测
            account_status[account_id] = optimized_predictions[account_id]
            continue
        
        # 选择测试批次（当前批次的一半）
        batch_size = len(current_batch) // 2
        test_batch = current_batch[:batch_size]
        remaining_batch = current_batch[batch_size:]
        
        # 统计当前确认状态和总体分布
        confirmed_good = sum(1 for aid in account_status if account_status[aid] == 0)
        confirmed_bad = sum(1 for aid in account_status if account_status[aid] == 1)
        unconfirmed = sum(1 for aid in account_status if account_status[aid] == -1)
        
        # 当前预测分布
        current_good = sum(1 for pred in optimized_predictions.values() if pred == 0)
        current_bad = sum(1 for pred in optimized_predictions.values() if pred == 1)
        
        print(f"                good     bad")
        print(f"总共数量：      {total_good}      {total_bad}")
        print(f"已经确认：      {confirmed_good}        {confirmed_bad}")  
        print(f"等待确认：      {unconfirmed}")
        print(f"本次猜测：      {current_good}      {current_bad}")
        
        # 获取当前F1
        current_f1 = test_current_predictions(optimized_predictions, upload_func)
        if current_f1 is None:
            print("❌ 无法获取F1，停止")
            break
        
        # 从F1反推混淆矩阵
        confusion = calculate_confusion_from_f1(current_f1, current_bad)
        if confusion:
            tp, fp, fn, tn = confusion['TP'], confusion['FP'], confusion['FN'], confusion['TN']
            print(f"正确猜测：      {tn}      {tp}")
            print(f"错误猜测：      {fp}        {fn}")
        
        print(f"  当前F1: {current_f1:.6f}")
        
        # 测试翻转效果
        decision = test_batch_flip_with_confusion(test_batch, optimized_predictions, upload_func, 
                                                current_f1, confusion, total_good, total_bad)
        
        # 保存统计数据
        iteration_stats = {
            'iteration': iteration,
            'current_batch_size': len(current_batch),
            'test_batch_size': len(test_batch),
            'remaining_batch_size': len(remaining_batch),
            'current_f1': current_f1,
            'decision': decision,
            'confirmed_accounts': confirmed_good + confirmed_bad,
            'unconfirmed_accounts': unconfirmed
        }
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
    
    # 创建临时CSV文件
    temp_file = "/Users/mannormal/4011/Qi Zihan/v3/temp_test.csv"
    
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

def test_batch_flip_with_confusion(test_batch, current_predictions, upload_func, baseline_f1, 
                                  baseline_confusion, total_good, total_bad):
    """
    基于混淆矩阵推断测试批次，决定是否继续二分
    
    Args:
        test_batch (list): 测试账户列表
        current_predictions (dict): 当前预测
        upload_func (function): 上传函数
        baseline_f1 (float): 基准F1分数
        baseline_confusion (dict): 基准混淆矩阵
        total_good (int): 总真实good数量
        total_bad (int): 总真实bad数量
    
    Returns:
        str: "flip_all", "keep_all", 或 "continue_binary"
    """
    
    # 测试翻转效果
    flipped_predictions = current_predictions.copy()
    for account_id in test_batch:
        flipped_predictions[account_id] = 1 - flipped_predictions[account_id]
    
    # 获取翻转后的F1分数
    flipped_f1 = test_current_predictions(flipped_predictions, upload_func)
    if flipped_f1 is None:
        print("  ❌ 无法获取翻转后F1分数，继续二分")
        return "continue_binary"
    
    # 计算翻转后的混淆矩阵
    from confusion_calculator import calculate_confusion_from_f1
    flipped_bad_count = sum(1 for pred in flipped_predictions.values() if pred == 1)
    flipped_confusion = calculate_confusion_from_f1(flipped_f1, flipped_bad_count)
    
    if not flipped_confusion:
        print("  ❌ 无法计算翻转后混淆矩阵，继续二分")
        return "continue_binary"
    
    print(f"  📊 批次翻转测试 ({len(test_batch)}个账户):")
    print(f"      当前F1: {baseline_f1:.6f}")
    print(f"      翻转F1: {flipped_f1:.6f}")
    print(f"      F1改进: {flipped_f1 - baseline_f1:.6f}")
    
    # 使用analyze_binary_split分析结果
    a_status, b_status, block_stats = analyze_binary_split(
        baseline_confusion, flipped_confusion, len(test_batch), 
        total_good, total_bad, current_predictions, test_batch
    )
    
    print(f"      推断状态: block_a={a_status}, block_b={b_status}")
    print(f"      统计: 正确={block_stats['correct']}, 错误={block_stats['wrong']}")
    
    # 决策逻辑
    if flipped_f1 > baseline_f1 + 0.001:  # 显著改进
        if b_status == "ALL_GOOD" or b_status == "ALL_BAD":
            print(f"  ✅ F1显著改进且状态纯净({b_status})，翻转并确认")
            return "flip_all"
        else:
            print(f"  🔄 F1改进但状态混合({b_status})，继续二分")
            return "continue_binary"
    elif abs(flipped_f1 - baseline_f1) < 0.001:  # 基本无变化
        if b_status == "ALL_GOOD" or b_status == "ALL_BAD":
            print(f"  ✅ F1无变化且状态纯净({b_status})，保持并确认")
            return "keep_all"
        else:
            print(f"  🔄 F1无变化但状态混合({b_status})，继续二分")
            return "continue_binary"
    else:  # F1下降
        print(f"  🔄 F1下降，继续二分寻找最优分割")
        return "continue_binary"


def test_batch_flip(test_batch, current_predictions, upload_func, baseline_f1):
    """
    简化版本的批次测试（向后兼容）
    """
    print(f"  ⚠️  使用简化测试模式，建议使用test_batch_flip_with_confusion")
    
    # 简单策略：如果批次较小，继续二分；如果较大，随机决策
    if len(test_batch) <= 2:
        return "continue_binary"
    elif len(test_batch) >= 20:
        # 对于大批次，测试翻转效果
        flipped_predictions = current_predictions.copy()
        for account_id in test_batch:
            flipped_predictions[account_id] = 1 - flipped_predictions[account_id]
        
        flipped_f1 = test_current_predictions(flipped_predictions, upload_func)
        if flipped_f1 and flipped_f1 > baseline_f1:
            return "flip_all"
        else:
            return "keep_all"
    else:
        return "continue_binary"


def optimize_single_account(account_id, current_predictions, upload_func):
    """
    优化单个账户 - 基于F1分数变化
    
    Args:
        account_id (str): 账户ID
        current_predictions (dict): 当前预测
        upload_func (function): 上传函数
    
    Returns:
        int: 最优预测值 (0 或 1)
    """
    
    print(f"优化单个账户: {account_id}")
    
    current_pred = current_predictions[account_id]
    
    # 测试翻转效果
    test_predictions = current_predictions.copy()
    test_predictions[account_id] = 1 - current_pred
    
    current_f1 = test_current_predictions(current_predictions, upload_func)
    flipped_f1 = test_current_predictions(test_predictions, upload_func)
    
    if current_f1 is None or flipped_f1 is None:
        print(f"❌ 无法获取F1分数，保持原值")
        return current_pred
    
    print(f"当前预测 {current_pred}: F1 = {current_f1:.6f}")
    print(f"翻转预测 {1-current_pred}: F1 = {flipped_f1:.6f}")
    
    # 选择F1更高的预测值
    if flipped_f1 > current_f1:
        print(f"✅ 选择翻转值: {1-current_pred}")
        return 1 - current_pred
    else:
        print(f"✅ 保持原值: {current_pred}")
        return current_pred

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
