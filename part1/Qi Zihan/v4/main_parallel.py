#!/usr/bin/env python3
"""
并行二分法优化系统 - 主程序
使用任务管理器和工作线程池进行并行优化
"""
import pandas as pd
import time
import os
import threading
import queue
import json
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from confusion_calculator import calculate_confusion_from_f1
#from upload_module import upload_file  # 正式realcase
from simulator import simulate_f1 as upload_file  # 模拟


# 添加打印锁和状态打印函数
print_lock = threading.Lock()

def print_status(confirmed_good, confirmed_bad, unconfirmed, 
                block_good, block_bad, correct_good, correct_bad, 
                wrong_good, wrong_bad, f1_score=None, worker_id=None, 
                description="", current_block=None, pending_blocks=None):
    """打印状态表格 - 包含block详细信息"""
    with print_lock:
        # 显示当前处理的block
        if current_block:
            print(f"📊 Worker-{worker_id} Block[正在处理]: {len(current_block.accounts)} 个账户 "
                  f"(正确: {current_block.estimated_correct}, 错误: {current_block.estimated_wrong}) ← 当前处理")
            
            # 显示待处理的block
            if pending_blocks and len(pending_blocks) > 0:
                for i, block in enumerate(pending_blocks[:3]):  # 只显示前3个待处理
                    print(f"    - Block[待处理-{i+1}]: {len(block.accounts)} 个账户 "
                          f"(预估正确: {block.estimated_correct}, 错误: {block.estimated_wrong})")
        else:
            print(f"\n📊 Worker-{worker_id} {description}" if worker_id else f"\n📊 {description}")
        
        print(f"                good     bad")
        print(f"总共数量：      6831     727")
        print(f"已经确认：      {confirmed_good}        {confirmed_bad}")  
        print(f"等待确认：      {unconfirmed}")
        print(f"本次猜测：      {block_good}      {block_bad}")
        print(f"正确猜测：      {correct_good}      {correct_bad}")
        print(f"错误猜测：      {wrong_good}      {wrong_bad}")
        if f1_score:
            print(f"当前F1：        {f1_score:.6f}")


@dataclass
class Block:
    id: str
    accounts: List[str]
    layer_id: int
    priority: int
    parent_id: Optional[str] = None
    created_time: float = None
    # 新增预估统计
    estimated_correct: int = 0
    estimated_wrong: int = 0
    status: str = "pending"  # pending, processing, completed
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = time.time()
    
    @property
    def size(self) -> int:
        return len(self.accounts)
    
    def __lt__(self, other):
        # 用于优先级队列排序：优先级小的先处理，同优先级按大小排序
        return (self.priority, -self.size) < (other.priority, -other.size)


def generate_block_id() -> str:
    """生成唯一的block ID"""
    return str(uuid.uuid4())[:8]


class TaskManager:
    def __init__(self, state_file="/Users/mannormal/4011/Qi Zihan/v4/parallel_state.json"):
        # 任务队列和状态
        self.task_queue = queue.PriorityQueue()  # 内置线程安全
        self.completed_blocks = {}  # {block_id: result}
        self.active_blocks = set()  # 正在处理的block_id集合
        
        # 全局状态 - 无锁
        self.global_predictions = {}  # {account_id: 0/1}
        self.account_status = {}      # {account_id: -1/0/1} -1=未确认
        
        # 统计信息
        self.total_iterations = 0
        
        # 新增：层级动态统计
        self.layer_stats = {}  # {layer_id: {'total_processed': 0, 'total_correct': 0, 'correct_rate': 0.5}}
        
        # 状态持久化
        self.state_file = state_file
        
        # 完成标志
        self.shutdown_event = threading.Event()
        
        print("🔧 TaskManager初始化完成")
    
    def estimate_block_performance(self, block: Block):
        """基于层统计预估block表现"""
        layer_stat = self.layer_stats.get(block.layer_id, {'correct_rate': 0.5})
        estimated_correct = int(len(block.accounts) * layer_stat['correct_rate'])
        block.estimated_correct = estimated_correct
        block.estimated_wrong = len(block.accounts) - estimated_correct
    
    def update_layer_stats(self, layer_id: int, actual_correct: int, block_size: int):
        """基于真实表现更新层统计"""
        if layer_id not in self.layer_stats:
            self.layer_stats[layer_id] = {'total_processed': 0, 'total_correct': 0, 'correct_rate': 0.5}
        
        self.layer_stats[layer_id]['total_processed'] += block_size
        self.layer_stats[layer_id]['total_correct'] += actual_correct
        self.layer_stats[layer_id]['correct_rate'] = (
            self.layer_stats[layer_id]['total_correct'] / 
            self.layer_stats[layer_id]['total_processed']
        )
        
        # 更新同层其他pending blocks的预估
        self._update_pending_blocks_estimates(layer_id)
    
    def _update_pending_blocks_estimates(self, layer_id: int):
        """更新同层待处理blocks的预估"""
        # 收集队列中的任务
        temp_tasks = []
        while not self.task_queue.empty():
            try:
                task = self.task_queue.get_nowait()
                temp_tasks.append(task)
            except queue.Empty:
                break
        
        # 更新同层block的预估
        for priority_tuple, block in temp_tasks:
            if block.layer_id == layer_id:
                self.estimate_block_performance(block)
        
        # 重新放回队列
        for task in temp_tasks:
            self.task_queue.put(task)
    
    def get_pending_blocks_preview(self, limit: int = 3) -> List[Block]:
        """获取待处理blocks的预览（不移除）"""
        temp_tasks = []
        pending_blocks = []
        
        try:
            while not self.task_queue.empty() and len(pending_blocks) < limit:
                task = self.task_queue.get_nowait()
                temp_tasks.append(task)
                pending_blocks.append(task[1])  # task[1] is the Block
        except queue.Empty:
            pass
        
        # 重新放回队列
        for task in temp_tasks:
            self.task_queue.put(task)
        
        return pending_blocks
    
    def add_block(self, block: Block):
        """添加新的block任务"""
        # 预估block表现
        self.estimate_block_performance(block)
        
        priority_tuple = (block.priority, -block.size, block.created_time)
        self.task_queue.put((priority_tuple, block))
    
    def get_next_task(self, preferred_layer_id: int, timeout=1.0) -> Optional[Block]:
        """获取下一个任务，优先获取指定层的任务"""
        temp_tasks = []
        found_preferred = None
        try:
            while not self.task_queue.empty():
                priority_tuple, block = self.task_queue.get_nowait()
                if block.layer_id == preferred_layer_id and found_preferred is None:
                    found_preferred = block
                else:
                    temp_tasks.append((priority_tuple, block))
            for task in temp_tasks:
                self.task_queue.put(task)
            if found_preferred:
                found_preferred.status = "processing"
                self.active_blocks.add(found_preferred.id)
                return found_preferred
            if not self.task_queue.empty():
                priority_tuple, block = self.task_queue.get(timeout=timeout)
                block.status = "processing"
                self.active_blocks.add(block.id)
                return block
        except queue.Empty:
            pass
        return None
    
    def complete_block(self, block_id: str, result_type: str, new_blocks: List[Block] = None):
        """完成一个block的处理"""
        self.completed_blocks[block_id] = {
            'result': result_type,
            'completed_time': time.time(),
            'new_blocks_count': len(new_blocks) if new_blocks else 0
        }
        
        self.active_blocks.discard(block_id)
        self.total_iterations += 1
        
        if new_blocks:
            for new_block in new_blocks:
                self.add_block(new_block)
    
    def update_global_predictions(self, account_updates: Dict[str, int]):
        """更新全局预测"""
        self.global_predictions.update(account_updates)
    
    def update_account_status(self, status_updates: Dict[str, int]):
        """更新账户确认状态"""
        self.account_status.update(status_updates)
    
    def get_global_predictions_copy(self) -> Dict[str, int]:
        """获取全局预测的副本"""
        return self.global_predictions.copy()
    
    def get_account_status_copy(self) -> Dict[str, int]:
        """获取账户状态的副本"""
        return self.account_status.copy()
    
    def confirm_accounts(self, accounts: List[str], predictions: Dict[str, int]):
        """确认账户的预测和状态"""
        self.update_global_predictions(predictions)
        status_updates = {aid: predictions[aid] for aid in accounts}
        self.update_account_status(status_updates)
    
    def is_all_complete(self) -> bool:
        """检查是否所有任务都已完成"""
        if self.shutdown_event.is_set():
            return True
        
        queue_empty = self.task_queue.empty()
        active_empty = len(self.active_blocks) == 0
        
        return queue_empty and active_empty
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_accounts = len(self.account_status)
        confirmed_accounts = sum(1 for status in self.account_status.values() if status != -1)
        pending_tasks = self.task_queue.qsize()
        active_tasks = len(self.active_blocks)
        return {
            'total_iterations': self.total_iterations,
            'total_accounts': total_accounts,
            'confirmed_accounts': confirmed_accounts,
            'pending_tasks': pending_tasks,
            'active_tasks': active_tasks,
            'completion_rate': confirmed_accounts / total_accounts if total_accounts > 0 else 0
        }
    
    def save_state(self):
        """保存当前状态到JSON文件"""
        try:
            # 收集所有待处理任务
            temp_tasks = []
            while not self.task_queue.empty():
                try:
                    task = self.task_queue.get_nowait()
                    temp_tasks.append(task)
                except queue.Empty:
                    break
            
            # 准备状态数据
            state_data = {
                'global_predictions': self.global_predictions,
                'account_status': self.account_status,
                'total_iterations': self.total_iterations,
                'completed_blocks': self.completed_blocks,
                'layer_stats': self.layer_stats,  # 保存层统计
                'pending_tasks': [
                    {
                        'id': block.id,
                        'accounts': block.accounts,
                        'layer_id': block.layer_id,
                        'priority': block.priority,
                        'parent_id': block.parent_id,
                        'estimated_correct': block.estimated_correct,
                        'estimated_wrong': block.estimated_wrong
                    }
                    for _, block in temp_tasks
                ],
                'saved_time': time.time()
            }
            
            # 恢复队列
            for task in temp_tasks:
                self.task_queue.put(task)
            
            # 写入文件
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            print(f"💾 状态已保存到 {self.state_file}")
            
        except Exception as e:
            print(f"❌ 保存状态失败: {e}")
    
    def load_state(self) -> bool:
        """从JSON文件恢复状态"""
        try:
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
            
            self.global_predictions = state_data.get('global_predictions', {})
            self.account_status = state_data.get('account_status', {})
            self.total_iterations = state_data.get('total_iterations', 0)
            self.completed_blocks = state_data.get('completed_blocks', {})
            self.layer_stats = state_data.get('layer_stats', {})  # 恢复层统计
            
            # 恢复待处理任务
            pending_tasks = state_data.get('pending_tasks', [])
            for task_data in pending_tasks:
                block = Block(
                    id=task_data['id'],
                    accounts=task_data['accounts'],
                    layer_id=task_data['layer_id'],
                    priority=task_data['priority'],
                    parent_id=task_data.get('parent_id')
                )
                # 恢复预估统计
                block.estimated_correct = task_data.get('estimated_correct', 0)
                block.estimated_wrong = task_data.get('estimated_wrong', 0)
                self.add_block(block)
            
            saved_time = state_data.get('saved_time', 0)
            print(f"📂 状态已恢复 (保存时间: {time.ctime(saved_time)})")
            print(f"   总迭代: {self.total_iterations}, 待处理任务: {len(pending_tasks)}")
            
            return True
            
        except FileNotFoundError:
            print("📂 未找到状态文件，从头开始")
            return False
        except Exception as e:
            print(f"❌ 恢复状态失败: {e}")
            return False
    
    def shutdown(self):
        """优雅关闭"""
        print("🛑 TaskManager正在关闭...")
        self.shutdown_event.set()
        self.save_state()
        print("✅ TaskManager已关闭")


class BinaryOptimizer:
    """二分优化器"""
    
    def __init__(self, task_manager: TaskManager):
        self.task_manager = task_manager
        
        # 层基准信息
        self.layer_baselines = {
            1: {'bad': 154, 'good': 6626, 'total': 6780},
            2: {'bad': 30, 'good': 45, 'total': 75},
            3: {'bad': 43, 'good': 61, 'total': 104},
            4: {'bad': 51, 'good': 62, 'total': 113},
            5: {'bad': 449, 'good': 37, 'total': 486}
        }
    
    def process_block(self, block: Block, worker_id: int) -> str:
        """
        处理单个block
        
        Args:
            block: 要处理的block
            worker_id: 工作线程ID
            
        Returns:
            处理结果类型
        """
        
        if block.size == 1:
            return self._process_single_account(block, worker_id)
        else:
            return self._process_batch_binary(block, worker_id)
    
    def _save_predictions_to_csv(self, predictions: Dict[str, int], filepath: str):
        """保存预测到CSV文件"""
        import pandas as pd
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        predictions_list = [{"ID": aid, "Predict": pred} for aid, pred in predictions.items()]
        df = pd.DataFrame(predictions_list)
        df.to_csv(filepath, index=False)

    def _test_predictions(self, predictions: Dict[str, int], description: str, worker_id: int) -> float:
        """测试预测并返回F1分数"""
        temp_dir = "/Users/mannormal/4011/Qi Zihan/v4/temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = f"{temp_dir}/test_{worker_id}_{int(time.time() * 1000)}.csv"
        try:
            self._save_predictions_to_csv(predictions, temp_file)
            f1_score = upload_file(temp_file)
            return f1_score
        finally:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"警告：清理临时文件失败 {temp_file}: {e}")

    def _create_test_batch_predictions(self, base_predictions: Dict[str, int], 
                                      test_batch: List[str], flip: bool = True) -> Dict[str, int]:
        """创建测试批次的预测（翻转或保持）"""
        test_predictions = base_predictions.copy()
        if flip:
            for account_id in test_batch:
                test_predictions[account_id] = 1 - test_predictions[account_id]
        return test_predictions

    def analyze_binary_split(self, confusion_baseline, confusion_flipped, n_b, total_good, total_bad, 
                           base_predictions, parent_block):
        """
        分析二分结果
        
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
            return "MIXED", "MIXED", {'correct': 0, 'wrong': len(parent_block.accounts), 'pred_good': 0, 'pred_bad': 0}
        
        # 判定block_b
        if b_bad == n_b:
            b_status = "ALL_BAD"
        elif b_good == n_b:
            b_status = "ALL_GOOD"
        else:
            b_status = "MIXED"
        
        # 判定block_a（剩下的）
        a_bad = total_bad - b_bad
        a_good = total_good - b_good
        
        if a_bad == 0:
            a_status = "ALL_GOOD"
        elif a_good == 0:
            a_status = "ALL_BAD"
        else:
            a_status = "MIXED"
        
        # 计算parent_block的真实分布和正确预测统计
        parent_true_good = a_good + b_good  
        parent_true_bad = a_bad + b_bad
        
        # 计算实际正确预测
        pred_good_count = sum(1 for aid in parent_block.accounts if base_predictions[aid] == 0)
        pred_bad_count = len(parent_block.accounts) - pred_good_count
        
        actual_correct = min(pred_good_count, parent_true_good) + min(pred_bad_count, parent_true_bad)
        actual_wrong = len(parent_block.accounts) - actual_correct
        
        # 更新层统计
        self.task_manager.update_layer_stats(parent_block.layer_id, actual_correct, len(parent_block.accounts))
        
        # 返回统计信息
        block_stats = {
            'correct': actual_correct,
            'wrong': actual_wrong,
            'pred_good': pred_good_count,
            'pred_bad': pred_bad_count
        }
        
        return a_status, b_status, block_stats

    def execute_split_decisions(self, a_status, b_status, block_a, block_b, 
                              base_predictions, parent_block):
        """
        执行分割决策
        """
        new_blocks = []
        
        # 处理block_a
        if a_status == "ALL_GOOD":
            predictions = {aid: base_predictions[aid] for aid in block_a}
            self.task_manager.confirm_accounts(block_a, predictions)
            
        elif a_status == "ALL_BAD":
            predictions = {aid: 1 - base_predictions[aid] for aid in block_a}
            self.task_manager.confirm_accounts(block_a, predictions)
            
        else:  # "MIXED"
            if len(block_a) > 1:
                mid_a = len(block_a) // 2
                new_blocks.append(Block(
                    generate_block_id(),
                    block_a[:mid_a],
                    parent_block.layer_id,
                    parent_block.priority,
                    parent_block.id
                ))
                new_blocks.append(Block(
                    generate_block_id(),
                    block_a[mid_a:],
                    parent_block.layer_id,
                    parent_block.priority,
                    parent_block.id
                ))
        
        # 处理block_b
        if b_status == "ALL_GOOD":
            predictions = {aid: base_predictions[aid] for aid in block_b}
            self.task_manager.confirm_accounts(block_b, predictions)
            
        elif b_status == "ALL_BAD":
            predictions = {aid: 1 - base_predictions[aid] for aid in block_b}
            self.task_manager.confirm_accounts(block_b, predictions)
            
        else:  # "MIXED"
            if len(block_b) > 1:
                mid_b = len(block_b) // 2
                new_blocks.append(Block(
                    generate_block_id(),
                    block_b[:mid_b],
                    parent_block.layer_id,
                    parent_block.priority,
                    parent_block.id
                ))
                new_blocks.append(Block(
                    generate_block_id(),
                    block_b[mid_b:],
                    parent_block.layer_id,
                    parent_block.priority,
                    parent_block.id
                ))
        
        return new_blocks

    def _process_single_account(self, block: Block, worker_id: int) -> str:
        """处理单个账户"""
        account_id = block.accounts[0]
        
        base_predictions = self.task_manager.get_global_predictions_copy()
        current_pred = base_predictions[account_id]

        # 测试原始预测
        f1_original = self._test_predictions(base_predictions, f"单账户{account_id}原值", worker_id)
        # 测试翻转预测
        flipped_predictions = self._create_test_batch_predictions(base_predictions, [account_id], flip=True)
        f1_flipped = self._test_predictions(flipped_predictions, f"单账户{account_id}翻转", worker_id)

        if f1_flipped is None or f1_original is None:
            print(f"❌ Worker-{worker_id} 单账户{account_id}测试失败")
            return "SINGLE_FAILED"

        if f1_flipped > f1_original:
            new_pred = 1 - current_pred
            predictions = {account_id: new_pred}
            self.task_manager.confirm_accounts([account_id], predictions)
            result_type = "SINGLE_FLIPPED"
        else:
            predictions = {account_id: current_pred}
            self.task_manager.confirm_accounts([account_id], predictions)
            result_type = "SINGLE_CONFIRMED"

        self.task_manager.complete_block(block.id, result_type)
        return result_type
    
    def _process_batch_binary(self, block: Block, worker_id: int) -> str:
        """处理批次二分"""
        
        # 二分账户列表
        mid_point = block.size // 2
        batch_A = block.accounts[:mid_point]
        batch_B = block.accounts[mid_point:]
        
        # 使用新的混淆矩阵分析方法
        decision, new_blocks = self._test_binary_split_confusion_based(
            batch_A, batch_B, block, worker_id
        )
        
        # 完成处理并添加新块
        self.task_manager.complete_block(block.id, decision, new_blocks)
        
        return decision
    
    def _test_binary_split_confusion_based(self, batch_A: List[str], batch_B: List[str], 
                                         parent_block: Block, worker_id: int) -> Tuple[str, List[Block]]:
        """
        基于混淆矩阵分析的二分测试
        
        Returns:
            (决策结果, 新产生的blocks列表)
        """
        
        base_predictions = self.task_manager.get_global_predictions_copy()
        
        # 测试基准
        f1_baseline = self._test_predictions(base_predictions, f"Block{parent_block.id[:8]}基准", worker_id)
        if f1_baseline is None:
            print(f"❌ Worker-{worker_id} 获取基准F1失败")
            return "BASELINE_FAILED", []
        
        # 测试batch_B翻转（我们分析batch_B）
        B_flipped_predictions = self._create_test_batch_predictions(base_predictions, batch_B, flip=True)
        f1_B_flipped = self._test_predictions(B_flipped_predictions, f"BatchB翻转", worker_id)
        if f1_B_flipped is None:
            print(f"❌ Worker-{worker_id} BatchB翻转测试失败")
            return "BATCH_B_FAILED", []
        
        # 获取混淆矩阵
        predicted_bad_baseline = sum(base_predictions.values())
        predicted_bad_B_flipped = predicted_bad_baseline
        for aid in batch_B:
            if base_predictions[aid] == 1:  # 原来是bad，翻转后是good
                predicted_bad_B_flipped -= 1
            else:  # 原来是good，翻转后是bad
                predicted_bad_B_flipped += 1
        
        confusion_baseline = calculate_confusion_from_f1(f1_baseline, predicted_bad_baseline)
        confusion_B_flipped = calculate_confusion_from_f1(f1_B_flipped, predicted_bad_B_flipped)
        
        if not confusion_baseline or not confusion_B_flipped:
            print(f"❌ Worker-{worker_id} 混淆矩阵计算失败")
            return "CONFUSION_FAILED", []
        
        # 获取层基准信息
        layer_info = self.layer_baselines[parent_block.layer_id]
        total_good = layer_info['good']
        total_bad = layer_info['bad']
        
        # 分析二分结果并获取统计信息
        a_status, b_status, block_stats = self.analyze_binary_split(
            confusion_baseline, confusion_B_flipped, len(batch_B), total_good, total_bad,
            base_predictions, parent_block
        )
        
        # 📍 显示状态 - 只显示当前处理和待处理的blocks
        account_status = self.task_manager.get_account_status_copy()
        confirmed_good = sum(1 for s in account_status.values() if s == 0)
        confirmed_bad = sum(1 for s in account_status.values() if s == 1)
        unconfirmed = sum(1 for s in account_status.values() if s == -1)
        
        # 获取待处理blocks预览
        pending_blocks = self.task_manager.get_pending_blocks_preview(3)
        
        print_status(confirmed_good, confirmed_bad, unconfirmed,
                     block_stats['pred_good'], block_stats['pred_bad'], 
                     block_stats['correct'], 0,  # 当前block正确预测的good和bad
                     block_stats['wrong'], 0,     # 当前block错误预测的good和bad
                     f1_baseline, worker_id, f"Block {parent_block.id[:8]} 基准状态",
                     current_block=parent_block, pending_blocks=pending_blocks)
        
        # 执行决策
        new_blocks = self.execute_split_decisions(
            a_status, b_status, batch_A, batch_B, base_predictions, parent_block
        )
        
        decision_summary = f"A_{a_status}_B_{b_status}"
        
        return decision_summary, new_blocks


def main():
    print("=== 并行二分法优化系统 ===")
    
    # 1. 初始化TaskManager
    task_manager = TaskManager()
    
    # 2. 尝试恢复状态
    if task_manager.load_state():
        print("📂 从保存状态恢复")
    else:
        print("🆕 从头开始")
        
        # 3. 读取数据并初始化
        scores_df = pd.read_csv("/Users/mannormal/4011/account_scores.csv")
        initialize_global_state(task_manager, scores_df)
        create_initial_blocks(task_manager, scores_df)
    
    # 4. 启动工作线程池
    print("\n🏭 启动工作线程池...")
    num_workers = 10
    workers = []
    
    for worker_id in range(num_workers):
        preferred_layer = (worker_id % 5) + 1
        
        worker = threading.Thread(
            target=worker_thread,
            args=(worker_id, preferred_layer, task_manager),
            daemon=True
        )
        worker.start()
        workers.append(worker)
    
    print(f"✅ 启动{num_workers}个工作线程")
    
    # 5. 等待所有任务完成
    for worker in workers:
        worker.join()
    
    # 6. 测试最终结果
    final_f1 = test_final_result(task_manager)
    
    # 7. 保存结果
    save_final_result(task_manager, final_f1)
    
    print(f"✅ 优化完成: F1={final_f1:.6f}")


def initialize_global_state(task_manager, scores_df):
    """初始化全局预测状态"""
    global_predictions = {}
    account_status = {}
    
    for _, row in scores_df.iterrows():
        account_id = row['ID']
        # 初始预测基于0.5阈值
        global_predictions[account_id] = 1 if row['predict'] > 0.5 else 0
        account_status[account_id] = -1  # 未确认
    
    task_manager.update_global_predictions(global_predictions)
    task_manager.update_account_status(account_status)


def create_initial_blocks(task_manager, scores_df):
    """创建5层初始Block"""
    layers = [
        {"id": 1, "range": (0.0, 0.1), "info": {'bad': 154, 'good': 6626}},
        {"id": 2, "range": (0.1, 0.2), "info": {'bad': 30, 'good': 45}},
        {"id": 3, "range": (0.2, 0.5), "info": {'bad': 43, 'good': 61}},
        {"id": 4, "range": (0.5, 0.8), "info": {'bad': 51, 'good': 62}},
        {"id": 5, "range": (0.8, 1.0), "info": {'bad': 449, 'good': 37}}
    ]
    
    for layer in layers:
        # 获取层内账户
        min_score, max_score = layer["range"]
        if max_score == 1.0:
            layer_df = scores_df[(scores_df['predict'] >= min_score) & (scores_df['predict'] <= max_score)]
        else:
            layer_df = scores_df[(scores_df['predict'] >= min_score) & (scores_df['predict'] < max_score)]
        
        layer_accounts = layer_df['ID'].tolist()
        
        if len(layer_accounts) > 0:
            block = Block(
                id=generate_block_id(),
                accounts=layer_accounts,
                layer_id=layer['id'],
                priority=layer['id']
            )
            task_manager.add_block(block)
            print(f"✅ 创建Layer {layer['id']}: {len(layer_accounts)}个账户")


def worker_thread(worker_id: int, preferred_layer_id: int, task_manager: TaskManager):
    """工作线程函数"""
    thread_name = f"Worker-{worker_id}"
    threading.current_thread().name = thread_name
    
    print(f"🚀 {thread_name} 启动 (优先Layer {preferred_layer_id})")
    
    optimizer = BinaryOptimizer(task_manager)
    processed_count = 0
    idle_count = 0
    max_idle = 30  # 最大空闲次数
    
    try:
        while not task_manager.is_all_complete():
            # 获取下一个任务
            block = task_manager.get_next_task(preferred_layer_id, timeout=2.0)
            
            if block is None:
                idle_count += 1
                if idle_count >= max_idle:
                    if task_manager.is_all_complete():
                        break
                    idle_count = 0  # 重置计数器
                
                time.sleep(1)  # 短暂休息
                continue
            
            idle_count = 0  # 重置空闲计数
            
            # 处理block
            try:
                result = optimizer.process_block(block, worker_id)
                processed_count += 1
                
                if processed_count % 10 == 0:  # 每处理10个任务打印一次统计
                    stats = task_manager.get_stats()
                    print()
                    print()
                    print()
                    print(f"📊 {thread_name} 已处理{processed_count}个任务, "
                          f"全局进度: {stats['confirmed_accounts']}/{stats['total_accounts']} "
                          f"({stats['completion_rate']:.1%})")
                    print()
                    print()
                    print()
                
            except Exception as e:
                print(f"❌ {thread_name} 处理Block {block.id[:8]}时出错: {e}")
                # 标记处理失败，但继续工作
                task_manager.complete_block(block.id, "PROCESSING_ERROR")
    
    except Exception as e:
        print(f"❌ {thread_name} 发生严重错误: {e}")
    
    finally:
        print(f"🏁 {thread_name} 退出，共处理{processed_count}个任务")


def save_predictions_to_csv(predictions: dict, filepath: str):
    """保存预测结果到CSV文件"""
    import pandas as pd
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    predictions_list = [{"ID": aid, "Predict": pred} for aid, pred in predictions.items()]
    df = pd.DataFrame(predictions_list)
    df.to_csv(filepath, index=False)


def test_final_result(task_manager: TaskManager) -> float:
    """测试最终结果"""
    print("🧪 测试最终结果...")
    final_predictions = task_manager.get_global_predictions_copy()
    temp_file = "/Users/mannormal/4011/Qi Zihan/v4/temp/final_test.csv"
    save_predictions_to_csv(final_predictions, temp_file)
    f1_score = upload_file(temp_file)
    try:
        import os
        os.remove(temp_file)
    except:
        pass
    if f1_score is not None:
        print(f"🎯 最终F1分数: {f1_score:.6f}")
    else:
        print("❌ 最终结果测试失败")
    return f1_score


def save_final_result(task_manager, f1_score):
    """保存最终结果"""
    final_predictions = task_manager.get_global_predictions_copy()
    
    # 保存结果
    timestamp = int(time.time())
    result_path = f"/Users/mannormal/4011/Qi Zihan/v4/parallel_result_{timestamp}.csv"
    save_predictions_to_csv(final_predictions, result_path)
    
    print(f"💾 结果已保存: {result_path}")


if __name__ == "__main__":
    main()