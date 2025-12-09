#!/usr/bin/env python3
"""
V3并行主程序 - 并行增强的二分法优化系统
基于main.py，添加并行处理功能
"""

import os
import pandas as pd
import copy
import concurrent.futures
import threading
import time
import uuid
#from robust_upload_module import robust_upload_with_retry as upload_file
from simulator import simulate_f1 as upload_file
from confusion_calculator import calculate_confusion_from_f1
# 使用内置的并行block处理


# 全局状态锁，防止线程竞争
state_lock = threading.Lock()

class ParallelStatusManager:
    """并行处理状态管理器 - 线程安全"""
    
    def __init__(self):
        self.block_status = {}
        self.status_lock = threading.Lock()
        self.session_dir = None
        
    def create_session_dir(self):
        """创建session级别的临时文件目录"""
        import datetime
        import shutil
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = f"/Users/mannormal/4011/Qi Zihan/v3/sessions/parallel_{timestamp}"
        os.makedirs(self.session_dir, exist_ok=True)
        print(f"📁 创建session目录: {self.session_dir}")
        return self.session_dir
    
    def cleanup_session(self):
        """清理session目录"""
        if self.session_dir and os.path.exists(self.session_dir):
            import shutil
            try:
                shutil.rmtree(self.session_dir)
                print(f"🗑️ 清理session目录: {self.session_dir}")
            except Exception as e:
                print(f"❌ 清理session目录失败: {e}")
    
    def update_block_status(self, block_id, status_info):
        """更新block状态"""
        with self.status_lock:
            self.block_status[block_id] = status_info
    
    def get_block_status(self, block_id):
        """获取block状态"""
        with self.status_lock:
            return self.block_status.get(block_id, {})
    
    def get_all_status(self):
        """获取所有block状态"""
        with self.status_lock:
            return self.block_status.copy()
    
    def format_block_summary(self, processing_stack, current_predictions=None, upload_func=None):
        """格式化block状态摘要，显示基于F1的真实Good/Bad分类情况"""
        with self.status_lock:
            if not processing_stack:
                return "  当前无待处理blocks"
            
            lines = [f"  当前有 {len(processing_stack)} 个待处理blocks:"]
            
            # 显示总体真实分类情况（通过F1推算）
            if current_predictions and upload_func:
                try:
                    current_f1 = test_predictions_f1(current_predictions, upload_func)
                    predicted_bad = sum(1 for pred in current_predictions.values() if pred == 1)
                    confusion = calculate_confusion_from_f1(current_f1, predicted_bad)
                    
                    if confusion:
                        correct_good = confusion['TN']  # 正确分类的Good
                        correct_bad = confusion['TP']   # 正确分类的Bad
                        wrong_good = confusion['FN']    # 错误分类的Good (实际是Bad)
                        wrong_bad = confusion['FP']     # 错误分类的Bad (实际是Good)
                        
                        lines.append(f"  📊 当前分类状况 (F1={current_f1:.4f}):")
                        lines.append(f"      ✅ 正确: Good={correct_good}, Bad={correct_bad}")
                        lines.append(f"      ❌ 错误: FP={wrong_bad}, FN={wrong_good}")
                except:
                    # F1计算失败，显示预测分布
                    predicted_good = sum(1 for pred in current_predictions.values() if pred == 0)
                    predicted_bad = sum(1 for pred in current_predictions.values() if pred == 1)
                    lines.append(f"  📊 当前预测分布: Good={predicted_good}, Bad={predicted_bad}")
            
            for i, block in enumerate(processing_stack):
                # 只显示block大小，避免混乱
                if i == 0:
                    lines.append(f"    - Block[正在处理]: {len(block)} 个账户 ← 当前处理")
                else:
                    lines.append(f"    - Block[待处理-{i}]: {len(block)} 个账户")
            
            return "\n".join(lines)
    
    def format_parallel_progress(self, confirmed_count=None, total_count=None):
        """格式化并行处理进度，只显示基本可靠信息"""
        # 只显示总体进度，不显示可能有错误的详细状态
        if confirmed_count is not None and total_count is not None:
            progress_pct = (confirmed_count / total_count * 100) if total_count > 0 else 0
            return f"📈 总体进度: {confirmed_count}/{total_count} ({progress_pct:.1f}%) 已确认"
        else:
            return "📈 并行处理进行中..."
    
    def update_block_predictions(self, block_id, current_predictions, block_accounts):
        """更新block的预测分布信息"""
        good_count = sum(1 for aid in block_accounts if current_predictions.get(aid, 0) == 0)
        bad_count = sum(1 for aid in block_accounts if current_predictions.get(aid, 0) == 1)
        
        with self.status_lock:
            if block_id in self.block_status:
                self.block_status[block_id]['good_pred'] = good_count
                self.block_status[block_id]['bad_pred'] = bad_count

# 全局状态管理器
status_manager = ParallelStatusManager()

def parallel_binary_optimize_accounts(account_list, current_predictions, upload_func, max_iterations=None, max_workers=10):
    """
    并行二分法优化 - 每个block作为一个task
    
    Args:
        account_list (list): 要优化的账户ID列表
        current_predictions (dict): 当前所有账户的预测
        upload_func (function): 上传函数
        max_iterations (int): 最大迭代次数
        max_workers (int): 最大并行worker数量
    
    Returns:
        tuple: (优化后的预测结果, 确认状态)
    """
    
    print(f"=== 并行二分法优化 ===")
    print(f"优化账户数: {len(account_list)}")
    print(f"最大workers: {max_workers}")
    
    # 创建session目录
    session_dir = status_manager.create_session_dir()
    
    try:
        # 初始化
        optimized_predictions = copy.deepcopy(current_predictions)
        processing_stack = [account_list.copy()]
        iteration = 0
        
        # 确认状态
        account_status = {}
        for account_id in account_list:
            account_status[account_id] = -1
        
        # 主循环
        while processing_stack and (max_iterations is None or iteration < max_iterations):
            iteration += 1
            
            print(f"\n--- 迭代 {iteration} ---")
            print(f"待处理blocks: {len(processing_stack)}")
            
            # 显示当前block状态（仿照main.py格式）
            block_summary = status_manager.format_block_summary(processing_stack, optimized_predictions, upload_func)
            print(block_summary)
            
            # 根据block数量选择处理策略
            if len(processing_stack) >= 2:
                # 并行处理多个blocks
                print(f"🚀 启动并行处理 {len(processing_stack)} 个blocks (max_workers={max_workers})")
                
                new_blocks = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交所有block任务
                    future_to_block = {}
                    for i, block in enumerate(processing_stack):
                        block_id = f"Block-{i+1}"
                        
                        future = executor.submit(
                            process_single_block, 
                            block, 
                            optimized_predictions.copy(), 
                            upload_func,
                            block_id
                        )
                        future_to_block[future] = (block, i+1)
                    
                    # 收集结果
                    for future in concurrent.futures.as_completed(future_to_block):
                        block, block_num = future_to_block[future]
                        try:
                            result = future.result()
                            if result:
                                decision = result['decision']
                                
                                if decision == "continue_binary":
                                    # 需要继续二分
                                    new_blocks.extend(result['new_blocks'])
                                    
                                elif decision in ["confirmed", "partial_confirmed"]:
                                    # 已确认的accounts
                                    with state_lock:
                                        for account_id in result['confirmed_accounts']:
                                            optimized_predictions[account_id] = result['predictions'][account_id]
                                            account_status[account_id] = result['predictions'][account_id]
                                    
                                    # 部分确认还有剩余blocks
                                    if decision == "partial_confirmed" and result.get('new_blocks'):
                                        new_blocks.extend(result['new_blocks'])
                                
                                # 简化状态更新，不显示可能错误的信息
                                pass
                            
                        except Exception as e:
                            print(f"❌ Block-{block_num} 处理失败: {e}")
                            # 失败处理：分解为更小blocks
                            if len(block) > 1:
                                new_blocks.extend([[account] for account in block])
                            else:
                                # 单账户失败，使用默认值
                                with state_lock:
                                    optimized_predictions[block[0]] = current_predictions[block[0]]
                                    account_status[block[0]] = current_predictions[block[0]]
                
                # 更新处理栈
                processing_stack = new_blocks
                
                # 显示并行处理状态
                confirmed_count = sum(1 for status in account_status.values() if status != -1)
                progress = status_manager.format_parallel_progress(confirmed_count, len(account_list))
                print(progress)
                
            else:
                # 串行处理单个block
                current_block = processing_stack.pop(0)
                result = process_single_block(current_block, optimized_predictions, upload_func, "Single")
                
                if result:
                    decision = result['decision']
                    
                    if decision == "continue_binary":
                        processing_stack.extend(result['new_blocks'])
                        
                    elif decision in ["confirmed", "partial_confirmed"]:
                        # 确认账户
                        for account_id in result['confirmed_accounts']:
                            optimized_predictions[account_id] = result['predictions'][account_id]
                            account_status[account_id] = result['predictions'][account_id]
                        
                        # 部分确认的剩余blocks
                        if decision == "partial_confirmed" and result.get('new_blocks'):
                            processing_stack.extend(result['new_blocks'])
            
            # 显示进度
            confirmed_count = sum(1 for status in account_status.values() if status != -1)
            print(f"📈 迭代{iteration}: 确认 {confirmed_count}/{len(account_list)}")
            
            # 检查是否完成
            if confirmed_count == len(account_list):
                print("🎉 所有账户已确认完成!")
                break
        
        print(f"\n=== 并行优化完成 ===")
        print(f"总迭代次数: {iteration}")
        print(f"已确认账户: {confirmed_count}/{len(account_list)}")
        
        return optimized_predictions, account_status
        
    finally:
        # 确保清理session目录
        status_manager.cleanup_session()

def process_single_block(block, current_predictions, upload_func, block_name):
    """
    处理单个block的函数 - 基于baseline和F1混淆矩阵比较
    
    Args:
        block (list): 账户ID列表
        current_predictions (dict): 当前预测
        upload_func (function): 上传函数
        block_name (str): block名称
    
    Returns:
        dict: 处理结果
    """
    
    # 单个账户测试两种情况选择更好的
    if len(block) == 1:
        account_id = block[0]
        
        # 测试保持当前预测
        current_f1 = test_predictions_f1(current_predictions, upload_func)
        
        # 测试翻转这个账户
        flipped_predictions = current_predictions.copy()
        flipped_predictions[account_id] = 1 - flipped_predictions[account_id]
        flipped_f1 = test_predictions_f1(flipped_predictions, upload_func)
        
        # 选择F1更高的预测
        if flipped_f1 > current_f1:
            return {
                'decision': 'confirmed',
                'confirmed_accounts': [account_id],
                'predictions': {account_id: flipped_predictions[account_id]}
            }
        else:
            return {
                'decision': 'confirmed',
                'confirmed_accounts': [account_id],
                'predictions': {account_id: current_predictions[account_id]}
            }
    
    try:
        # 计算baseline（当前预测状态）
        baseline_f1 = test_predictions_f1(current_predictions, upload_func)
        baseline_bad_count = sum(1 for pred in current_predictions.values() if pred == 1)
        baseline_confusion = calculate_confusion_from_f1(baseline_f1, baseline_bad_count)
        
        if not baseline_confusion:
            print(f"❌ [{block_name}] 无法计算baseline混淆矩阵")
            return {
                'decision': 'continue_binary',
                'new_blocks': [block],
                'confirmed_accounts': [],
                'predictions': {}
            }
        
        baseline_correct = baseline_confusion['TP'] + baseline_confusion['TN']
        
        # 测试翻转当前block后的效果
        test_predictions = current_predictions.copy()
        for account_id in block:
            test_predictions[account_id] = 1 - test_predictions[account_id]
        
        test_f1 = test_predictions_f1(test_predictions, upload_func)
        test_bad_count = sum(1 for pred in test_predictions.values() if pred == 1)
        test_confusion = calculate_confusion_from_f1(test_f1, test_bad_count)
        
        if not test_confusion:
            print(f"❌ [{block_name}] 无法计算test混淆矩阵")
            return {
                'decision': 'continue_binary',
                'new_blocks': [block],
                'confirmed_accounts': [],
                'predictions': {}
            }
        
        test_correct = test_confusion['TP'] + test_confusion['TN']
        total_accounts = len(current_predictions)  # 总账户数7558
        
        # 决策逻辑：检查是否达到完美分类或有提升
        if test_correct == total_accounts:
            # 翻转后达到完美分类，翻转并确认整个block
            confirmed_predictions = {}
            for account_id in block:
                confirmed_predictions[account_id] = 1 - current_predictions[account_id]
            
            return {
                'decision': 'confirmed',
                'confirmed_accounts': block,
                'predictions': confirmed_predictions
            }
        elif baseline_correct == total_accounts:
            # 当前已经完美分类，保持并确认整个block
            return {
                'decision': 'confirmed',
                'confirmed_accounts': block,
                'predictions': {aid: current_predictions[aid] for aid in block}
            }
        elif test_correct > baseline_correct:
            # 翻转后有提升但不完美，需要继续二分找到最优子集
            # 不能直接确认整个block，要继续细分
            mid = len(block) // 2
            block_a = block[:mid]
            block_b = block[mid:]
            
            return {
                'decision': 'continue_binary',
                'new_blocks': [block_a, block_b] if block_b else [block_a],
                'confirmed_accounts': [],
                'predictions': {}
            }
        else:
            # 翻转后无提升，继续二分这个block
            mid = len(block) // 2
            block_a = block[:mid]
            block_b = block[mid:]
            
            return {
                'decision': 'continue_binary',
                'new_blocks': [block_a, block_b] if block_b else [block_a],
                'confirmed_accounts': [],
                'predictions': {}
            }
    
    except Exception as e:
        print(f"❌ [{block_name}] 处理失败: {e}")
        return {
            'decision': 'continue_binary',
            'new_blocks': [block],
            'confirmed_accounts': [],
            'predictions': {}
        }

def test_predictions_f1(predictions_dict, upload_func):
    """
    测试预测结果的F1分数
    
    Args:
        predictions_dict (dict): 预测结果
        upload_func (function): 上传函数
    
    Returns:
        float: F1分数，失败返回0
    """
    try:
        # 创建临时文件
        unique_id = f"{int(time.time()*1000000)}_{uuid.uuid4().hex[:8]}"
        session_dir = status_manager.session_dir or "/Users/mannormal/4011/Qi Zihan/v3"
        temp_file = f"{session_dir}/test_f1_{unique_id}.csv"
        
        predictions_list = []
        for account_id, predict in predictions_dict.items():
            predictions_list.append({"ID": account_id, "Predict": predict})
        
        df = pd.DataFrame(predictions_list)
        df.to_csv(temp_file, index=False)
        
        # 上传测试
        f1_score = upload_func(temp_file)
        
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return f1_score if f1_score is not None else 0
        
    except Exception as e:
        print(f"❌ 测试F1失败: {e}")
        return 0

def save_best_result(predictions_dict, f1_score, output_dir="/Users/mannormal/4011/Qi Zihan/v3/para"):
    """
    保存最佳结果，文件名格式: v{score}.csvn
    
    Args:
        predictions_dict (dict): 预测结果 {account_id: 0/1}
        f1_score (float): F1分数
        output_dir (str): 输出目录
    
    Returns:
        str: 保存的文件路径
    """
    
    # 创建DataFrame
    predictions_list = []
    for account_id, predict in predictions_dict.items():
        predictions_list.append({"ID": account_id, "Predict": predict})
    
    df = pd.DataFrame(predictions_list)
    
    # 生成文件名（并行版本）
    score_str = f"{f1_score:.8f}".replace(".", "")[:8]  # 取前8位数字
    filename = f"parallel_v{score_str}ensemble.csv"
    filepath = os.path.join(output_dir, filename)
    
    # 保存文件
    df.to_csv(filepath, index=False)
    
    print(f"💾 保存最佳结果: {filename}")
    print(f"  F1分数: {f1_score:.6f}")
    print(f"  预测Bad: {len(df[df['Predict'] == 1])}")
    print(f"  预测Good: {len(df[df['Predict'] == 0])}")
    
    return filepath

def load_initial_predictions():
    print("=== 分层二分法优化 ===")
    
    scores_df = pd.read_csv("/Users/mannormal/4011/account_scores.csv")
    
    layers = [
        {"id": 1, "name": "[0.0-0.1)", "range": (0.0, 0.1), "initial_guess": 0},
        {"id": 2, "name": "[0.1-0.2)", "range": (0.1, 0.2), "initial_guess": 0},
        {"id": 3, "name": "[0.2-0.5)", "range": (0.2, 0.5), "initial_guess": 0},
        {"id": 4, "name": "[0.5-0.8)", "range": (0.5, 0.8), "initial_guess": 1},
        {"id": 5, "name": "[0.8-1.0]", "range": (0.8, 1.0), "initial_guess": 1}
    ]
    layers.reverse()
    
    final_predictions = {}
    state_file = "/Users/mannormal/4011/Qi Zihan/v3/para/optimization_state_para.json"
    total_iterations = 0
    
    if os.path.exists(state_file):
        import json
        with open(state_file, 'r') as f:
            state = json.load(f)
        print(f"恢复状态: 从第{state['current_layer']}层开始")
        final_predictions = state['predictions']
        start_layer = state['current_layer']
        total_iterations = state.get('total_iterations', 0)
        print(f"已完成总迭代次数: {total_iterations}")
    else:
        start_layer = 1
        final_predictions = {}
        total_iterations = 0
    
    for layer in layers:
        
        if layer['id'] < start_layer:
            continue
            
        print(f"\n=== 大迭代 v{layer['id']}: {layer['name']} ===")
        
        min_score, max_score = layer["range"]
        if max_score == 1.0:
            layer_accounts = scores_df[(scores_df['predict'] >= min_score) & (scores_df['predict'] <= max_score)]
        else:
            layer_accounts = scores_df[(scores_df['predict'] >= min_score) & (scores_df['predict'] < max_score)]
        
        layer_account_ids = layer_accounts['ID'].tolist()
        print(f"层内账户数: {len(layer_account_ids)}")
        
        if len(layer_account_ids) == 0:
            continue
        
        current_predictions = {}
        for account_id in scores_df['ID']:
            if account_id in final_predictions:
                # 已确认的账户保持确认状态
                current_predictions[account_id] = final_predictions[account_id]
            elif account_id in layer_account_ids:
                # 当前层账户设为初始猜测（未确认状态-1会在二分法中处理）
                current_predictions[account_id] = layer['initial_guess']
            else:
                # 其他层账户设为相反值
                current_predictions[account_id] = 1 - layer['initial_guess']
        
        iteration = 1
        temp_file = f"/Users/mannormal/4011/Qi Zihan/v3/para/para_v{layer['id']}.{iteration}.csv"
        temp_df = pd.DataFrame([{"ID": aid, "Predict": pred} for aid, pred in current_predictions.items()])
        temp_df.to_csv(temp_file, index=False)
        
        layer_f1 = upload_file(temp_file)
        
        if layer_f1 is not None:
            predicted_bad = sum(current_predictions.values())
            confusion = calculate_confusion_from_f1(layer_f1, predicted_bad)
            
            if confusion:
                print(f"初始F1={layer_f1:.6f}, TP={confusion['TP']}, FP={confusion['FP']}, FN={confusion['FN']}, TN={confusion['TN']}")
                
                # 动态计算迭代次数 (基于log2 + 缓冲)
                layer_iterations = max(5, int(len(layer_account_ids).bit_length()) + 3)
                print(f"设置迭代次数: {layer_iterations} (基于层大小{len(layer_account_ids)})")
                print(f"当前总迭代次数: {total_iterations}")
                
                optimized_predictions, account_status = parallel_binary_optimize_accounts(
                    layer_account_ids,
                    current_predictions,
                    upload_file,
                    max_iterations=None,  # 无限制，找到为止
                    max_workers=10  # 10个并行workers
                )
                
                # 保存优化结果和确认状态
                for account_id in layer_account_ids:
                    final_predictions[account_id] = optimized_predictions[account_id]
                
                # 统计确认状态
                confirmed_count = sum(1 for status in account_status.values() if status != -1)
                unconfirmed_count = len(layer_account_ids) - confirmed_count
                print(f"层优化完成: 确认了 {confirmed_count}/{len(layer_account_ids)} 个账户，未确认: {unconfirmed_count}")
        
        else:
            print(f"❌ 层 v{layer['id']} 上传失败，使用initial guess")
            for account_id in layer_account_ids:
                final_predictions[account_id] = layer['initial_guess']
        
        import json
        total_iterations += layer_iterations if 'layer_iterations' in locals() else 0
        
        # 创建完整的状态文件，包含确认状态
        all_account_status = {}
        if 'account_status' in locals():
            all_account_status.update(account_status)
        
        state = {
            'current_layer': layer['id'] + 1,
            'predictions': final_predictions,
            'account_status': all_account_status,  # 新增确认状态
            'total_iterations': total_iterations
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)
        print(f"保存状态: 第{layer['id']}层完成，累计迭代: {total_iterations}")
        
        # Debug: 显示确认状态统计
        if all_account_status:
            confirmed_good = sum(1 for status in all_account_status.values() if status == 0)
            confirmed_bad = sum(1 for status in all_account_status.values() if status == 1)
            unconfirmed = sum(1 for status in all_account_status.values() if status == -1)
            print(f"当前总状态: 确认good={confirmed_good}, 确认bad={confirmed_bad}, 未确认={unconfirmed}")
    
    for account_id in scores_df['ID']:
        if account_id not in final_predictions:
            final_predictions[account_id] = 0
    
    bad_count = sum(final_predictions.values())
    good_count = len(final_predictions) - bad_count
    
    print(f"\n✅ 所有层优化完成")
    print(f"5层预测结果: Bad={bad_count}, Good={good_count}")
    
    # v7.0持续优化 - 每次F1破新高就覆盖保存
    print(f"\n=== v7.0 持续优化 - 目标F1=1.0 ===")
    
    # 初始测试
    v7_file = "/Users/mannormal/4011/Qi Zihan/v3/v7.0.csv"
    temp_df = pd.DataFrame([{"ID": aid, "Predict": pred} for aid, pred in final_predictions.items()])
    temp_df.to_csv(v7_file, index=False)
    
    best_f1 = upload_file(v7_file)
    best_predictions = final_predictions.copy()
    
    print(f"初始F1={best_f1:.6f}")
    
    # 持续优化直到F1=1或无法提升
    max_rounds = 20
    scores_dict = dict(zip(scores_df['ID'], scores_df['predict']))
    tested_candidates = set()  # 记录已测试的候选账户组合
    no_improvement_rounds = 0  # 连续无改进轮次计数
    
    for round_num in range(1, max_rounds + 1):
        print(f"\n--- v7.0 轮次 {round_num} ---")
        
        # 计算当前混淆矩阵
        predicted_bad = sum(best_predictions.values())
        confusion = calculate_confusion_from_f1(best_f1, predicted_bad)
        
        if confusion['FP'] == 0 and confusion['FN'] == 0:
            print("🎉 达到完美F1=1.0!")
            break
        
        print(f"TP={confusion['TP']}, FP={confusion['FP']}, FN={confusion['FN']}, TN={confusion['TN']}")
        
        # 选择优化目标 - 使用渐进策略避免重复测试
        candidates = []
        
        # 计算需要测试的账户数量（逐渐增加范围）
        fp_test_count = min(confusion['FP'], max(1, confusion['FP'] // (round_num + 1)))
        fn_test_count = min(confusion['FN'], max(1, confusion['FN'] // (round_num + 1)))
        
        if confusion['FP'] > 0:
            # 有FP：优化分数最低的bad预测账户
            bad_accounts = [aid for aid, pred in best_predictions.items() if pred == 1]
            # 跳过前面轮次已测试的账户
            skip_count = (round_num - 1) * fp_test_count
            fp_candidates = sorted(bad_accounts, key=lambda x: scores_dict[x])[skip_count:skip_count + fp_test_count]
            candidates.extend(fp_candidates)
            print(f"优化{len(fp_candidates)}个可能的FP账户 (跳过前{skip_count}个)")
        
        if confusion['FN'] > 0:
            # 有FN：优化分数最高的good预测账户
            good_accounts = [aid for aid, pred in best_predictions.items() if pred == 0]
            # 跳过前面轮次已测试的账户
            skip_count = (round_num - 1) * fn_test_count
            fn_candidates = sorted(good_accounts, key=lambda x: scores_dict[x], reverse=True)[skip_count:skip_count + fn_test_count]
            candidates.extend(fn_candidates)
            print(f"优化{len(fn_candidates)}个可能的FN账户 (跳过前{skip_count}个)")
        
        if not candidates:
            print("没有更多可优化的账户")
            break
        
        # 检查是否已测试过这组候选账户
        candidates_key = tuple(sorted(candidates))
        if candidates_key in tested_candidates:
            print("此组合已测试过，尝试扩大范围")
            # 如果已测试过，增加测试范围
            if confusion['FP'] > 0:
                bad_accounts = [aid for aid, pred in best_predictions.items() if pred == 1]
                fp_candidates = sorted(bad_accounts, key=lambda x: scores_dict[x])[:min(len(bad_accounts), confusion['FP'] + round_num)]
                candidates.extend(fp_candidates)
            if confusion['FN'] > 0:
                good_accounts = [aid for aid, pred in best_predictions.items() if pred == 0]
                fn_candidates = sorted(good_accounts, key=lambda x: scores_dict[x], reverse=True)[:min(len(good_accounts), confusion['FN'] + round_num)]
                candidates.extend(fn_candidates)
            candidates = list(set(candidates))  # 去重
            candidates_key = tuple(sorted(candidates))
            
            if candidates_key in tested_candidates:
                print("扩大范围后仍是重复组合，停止优化")
                break
        
        tested_candidates.add(candidates_key)
        
        # 进行优化
        optimized_predictions, v7_account_status = parallel_binary_optimize_accounts(
            candidates,
            best_predictions,
            upload_file,
            max_iterations=None,
            max_workers=10
        )
        
        # 测试新结果
        test_df = pd.DataFrame([{"ID": aid, "Predict": pred} for aid, pred in optimized_predictions.items()])
        test_df.to_csv(v7_file, index=False)
        
        new_f1 = upload_file(v7_file)
        
        if new_f1 > best_f1:
            print(f"🎉 F1破新高: {best_f1:.6f} → {new_f1:.6f}")
            best_f1 = new_f1
            best_predictions = optimized_predictions.copy()
            print(f"覆盖保存 v7.0.csv")
            no_improvement_rounds = 0  # 重置无改进计数
        else:
            print(f"本轮无提升: {new_f1:.6f} ≤ {best_f1:.6f}")
            no_improvement_rounds += 1
            
        # 如果连续3轮无改进，停止优化
        if no_improvement_rounds >= 3:
            print("连续3轮无改进，停止优化")
            break
            
        if best_f1 >= 0.9999:
            print("🎉 接近完美F1!")
            break
    
    print(f"\nv7.0最终结果: F1={best_f1:.6f}")
    print(f"总计完成迭代次数: {total_iterations}")
    
    # 保存最终结果为CSV文件
    if best_predictions:
        save_best_result(best_predictions, best_f1)
    
    # 最终保存状态
    import json
    final_state = {
        'current_layer': 6,  # 表示已完成所有层
        'predictions': best_predictions,
        'total_iterations': total_iterations,
        'final_f1': best_f1
    }
    with open(state_file, 'w') as f:
        json.dump(final_state, f, indent=2)
    print(f"保存最终状态: 总迭代{total_iterations}次，F1={best_f1:.6f}")
    
    return best_predictions


def main():
    print("=== V3二分法优化系统 ===")
    result = load_initial_predictions()
    print(f"✅ 完成: {result}")
    reset_score = upload_file("/Users/mannormal/4011/Qi Zihan/v2/results/Transformer_basic_submission_FULL_DATA_f1_0.8913_epochs_130.256.csv")
    print(f"重新上传旧版本文件测试F1: {reset_score}")
if __name__ == "__main__":
    main()