#!/usr/bin/env python3
"""
重构的Block二分处理系统
"""

import queue
import threading
import time
import json
import os
import pandas as pd
from simulator import simulate_f1 as upload_file
from enhanced_confusion_calculator import get_A_B_only, LAYER_CONFIG

class BlockQueueProcessor:
    def __init__(self):
        self.main_queue = queue.Queue()      # 待处理blocks
        self.json_queue = queue.Queue()      # JSON写入消息  
        self.result_queue = queue.Queue()    # 新生成blocks
        self.account_status_file = "account_status.json"
        self.num_workers = 1
        self.worker_threads = []
        self.json_thread = None
        self.running = True
        self.account_scores = self.load_account_scores()
        self.iteration_count = 0  # 添加迭代计数器
        self.iteration_lock = threading.Lock()  # 添加线程锁
        
    def load_account_scores(self):
        """加载账户分数"""
        df = pd.read_csv("/Users/mannormal/4011/account_scores.csv")
        return dict(zip(df['ID'], df['predict']))

    def get_accounts_by_layer(self, layer_name):
        """根据层级获取账户列表"""
        account_scores = self.account_scores
        aids = []
        if layer_name == 'high_good':
            aids = [aid for aid, score in account_scores.items() if 0.0 <= score < 0.1]
        elif layer_name == 'mid':
            aids = [aid for aid, score in account_scores.items() if 0.1 <= score < 0.8]
        elif layer_name == 'high_bad':
            aids = [aid for aid, score in account_scores.items() if 0.8 <= score <= 1.0]
        return aids

    def get_block_accounts(self, block):
        """获取属于当前block的账户列表"""
        if 'accounts' in block:
            return block['accounts']
        else:
            # 初始block，使用layer获取
            return self.get_accounts_by_layer(block['layer'])

    def initialize_blocks(self):
        """初始化三个初始block"""
        initial_blocks = [
            {
                'id': 'high_good',
                'layer': 'high_good', 
                'real_good': 6626,
                'real_bad': 154,
                'predict':0
            },
            {
                'id': 'mid',
                'layer': 'mid',
                'real_good': 168,
                'real_bad': 124,
                'predict':0
            },
            {
                'id': 'high_bad', 
                'layer': 'high_bad',
                'real_good': 37,
                'real_bad': 449,
                'predict':1
            }
        ]
        
        for block in initial_blocks:
            self.main_queue.put(block)
        
        # 初始化所有账户状态为-1
        account_status = {}
        for account_id in self.account_scores.keys():
            account_status[account_id] = -1
        
        self.json_queue.put({
            "type": "init_all",
            "data": account_status
        })
    
    def worker_process(self, worker_id):
        """Worker线程主函数"""
        while self.running:
            if self.main_queue.empty():
                time.sleep(1)
                if self.main_queue.empty():
                    break
                continue
                
            block = self.main_queue.get()
            
            # 增加迭代计数
            with self.iteration_lock:
                self.iteration_count += 1

            new_blocks = self.process_single_block(block, worker_id)
            
            # 将新blocks加入结果队列
            for new_block in new_blocks:
                self.result_queue.put(new_block)
                
            self.main_queue.task_done()
    
    def create_prediction_csv(self, block, worker_id):
        """创建预测CSV文件 - 对block进行A/B分组测试"""
        block_accounts = self.get_block_accounts(block)
        
        # 将block账户分成A、B两组
        mid_point = len(block_accounts) // 2
        group_a = block_accounts[:mid_point]
        group_b = block_accounts[mid_point:]
        
        temp_file = f"/tmp/temp_block_{worker_id}_{threading.current_thread().ident}.csv"
        
        predictions = []
        for account_id in self.account_scores.keys():
            if account_id in group_a:
                predict = block['predict']  # A组预测为block指定的值
            elif account_id in group_b:
                predict = 0 if block['predict'] == 1 else 1 
            else:
                predict = 0 if block['predict'] == 1 else 1 
            
            predictions.append({"ID": account_id, "Predict": predict})
        
        print(f"A组账户数: {len(group_a)}, B组账户数: {len(group_b)}")
        print(f"总预测为0: {sum(1 for p in predictions if p['Predict'] == 0)}")
        print(f"总预测为1: {sum(1 for p in predictions if p['Predict'] == 1)}")
        predictbad = sum(1 for p in predictions if p['Predict'] == 1)
        df = pd.DataFrame(predictions)
        df.to_csv(temp_file, index=False)
        return temp_file, predictbad

    def process_single_block(self, block, worker_id):
        """处理单个block"""
        print(f"\n=== 处理 {block['id']} ===")
        
        # 显示block信息
        print(f"real_good: {block['real_good']}")
        print(f"real_bad: {block['real_bad']}")
        
        # 创建临时CSV文件进行A/B测试
        temp_csv, predicted_bad_count = self.create_prediction_csv(block, worker_id)
        
        # 上传获取F1分数
        f1_score = upload_file(temp_csv)
        
        # 清理临时文件
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        
        if f1_score is None:
            return []
        
        # 使用get_A_B_only获取A/B组情况
        base_layer = self.get_base_layer(block.get('layer', 'unknown'))
        ab_result = get_A_B_only(f1_score, predicted_bad_count, base_layer)
        
        if ab_result is None:
            return []
        
        # 显示A/B组统计
        print(f"group_A: good={ab_result['A_TP']}, bad={ab_result['A_FP']}")
        print(f"group_B: good={ab_result['B_TP']}, bad={ab_result['B_FP']}")
        
        new_blocks = []
        block_accounts = self.get_block_accounts(block)
        mid_point = len(block_accounts) // 2
        group_a_accounts = block_accounts[:mid_point]
        group_b_accounts = block_accounts[mid_point:]
        
        # 检查A组
        if ab_result['A_TP'] == 0:
            print("归类 group_A -> bad")
            self.json_queue.put({
                "type": "update_group",
                "accounts": group_a_accounts,
                "flag": 1
            })
        elif ab_result['A_FP'] == 0:
            print("归类 group_A -> good")
            self.json_queue.put({
                "type": "update_group", 
                "accounts": group_a_accounts,
                "flag": 0
            })
        else:
            print("归类 group_A -> 继续二分")
            new_block_a = {
                'id': f"{block['id']}_A",
                'layer': f"{block.get('layer', 'unknown')}_A",
                'real_good': ab_result['A_TP'],
                'real_bad': ab_result['A_FP'],
                'accounts': group_a_accounts,
                'predict': block['predict']
            }
            new_blocks.append(new_block_a)
        
        # 检查B组
        if ab_result['B_TP'] == 0:
            print("归类 group_B -> bad")
            self.json_queue.put({
                "type": "update_group",
                "accounts": group_b_accounts,
                "flag": 1
            })
        elif ab_result['B_FP'] == 0:
            print("归类 group_B -> good")
            self.json_queue.put({
                "type": "update_group",
                "accounts": group_b_accounts,
                "flag": 0
            })
        else:
            print("归类 group_B -> 继续二分")
            new_block_b = {
                'id': f"{block['id']}_B",
                'layer': f"{block.get('layer', 'unknown')}_B", 
                'real_good': ab_result['B_TP'],
                'real_bad': ab_result['B_FP'],
                'accounts': group_b_accounts,
                'predict': block['predict']
            }
            new_blocks.append(new_block_b)
        
        self.print_current_statistics()
        
        return new_blocks

    def get_base_layer(self, layer_name):
        """从复杂layer名中提取基础层级"""
        if layer_name.startswith('high_good'):
            return 'high_good'
        elif layer_name.startswith('mid'):
            return 'mid'
        elif layer_name.startswith('high_bad'):
            return 'high_bad'
        return layer_name
    
    def print_current_statistics(self):
        """打印当前统计信息"""
        # 读取当前JSON状态
        if os.path.exists(self.account_status_file):
            with open(self.account_status_file, 'r') as f:
                account_status = json.load(f)
        else:
            account_status = {}
        
        # 统计已确认的good/bad
        confirmed_good = sum(1 for status in account_status.values() if status == 0)
        confirmed_bad = sum(1 for status in account_status.values() if status == 1)
        
        # 按层级统计
        high_good_confirmed = 0
        mid_confirmed = 0  
        high_bad_confirmed = 0
        
        for account_id, status in account_status.items():
            if status != -1:  # 已确认
                score = self.account_scores.get(account_id, 0)
                if 0.0 <= score < 0.1:
                    high_good_confirmed += 1
                elif 0.1 <= score < 0.8:
                    mid_confirmed += 1
                elif 0.8 <= score <= 1.0:
                    high_bad_confirmed += 1
        
        print(f"\n已确认统计:")
        print(f"high_good: {high_good_confirmed}/6780")
        print(f"mid: {mid_confirmed}/292") 
        print(f"high_bad: {high_bad_confirmed}/486")
        print(f"总计: good={confirmed_good}/6831, bad={confirmed_bad}/727")
        print(f"迭代次数: {self.iteration_count}")
    def json_writer_process(self):
        """JSON写入线程主函数"""
        account_status = {}
        
        # 如果状态文件存在，先加载
        if os.path.exists(self.account_status_file):
            try:
                with open(self.account_status_file, 'r') as f:
                    account_status = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ JSON文件损坏，重新初始化: {self.account_status_file}")
                account_status = {}
                # 删除损坏的文件
                os.remove(self.account_status_file)
    
        while self.running or not self.json_queue.empty():
            if self.json_queue.empty():
                time.sleep(0.1)
                continue
                
            message = self.json_queue.get()
            
            if message['type'] == 'init_all':
                account_status = message['data']
            elif message['type'] == 'update_group':
                for account_id in message['accounts']:
                    account_status[str(account_id)] = message['flag']
        
            # 写入文件 - 使用临时文件确保原子性
            temp_file = f"{self.account_status_file}.tmp"
            try:
                with open(temp_file, 'w') as f:
                    json.dump(account_status, f, indent=2)
                # 原子性移动
                if os.path.exists(self.account_status_file):
                    os.remove(self.account_status_file)
                os.rename(temp_file, self.account_status_file)
            except Exception as e:
                print(f"⚠️ JSON写入失败: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
            self.json_queue.task_done()
    
    def run_processing(self):
        """主处理流程"""
        # 启动JSON写入线程
        self.json_thread = threading.Thread(target=self.json_writer_process)
        self.json_thread.start()
        
        # 启动worker线程
        for i in range(1, self.num_workers + 1):
            worker_thread = threading.Thread(target=self.worker_process, args=(i,))
            worker_thread.start()
            self.worker_threads.append(worker_thread)
        
        # 主线程回调：结果队列 → 主队列
        while any(t.is_alive() for t in self.worker_threads) or not self.result_queue.empty():
            if not self.result_queue.empty():
                new_block = self.result_queue.get()
                self.main_queue.put(new_block)
                self.result_queue.task_done()
            else:
                time.sleep(0.1)
        
        # 等待所有worker完成
        for worker_thread in self.worker_threads:
            worker_thread.join()
        
        # 停止JSON写入线程
        self.running = False
        self.json_thread.join()
        
        print("\n=== 处理完成 ===")


def main():
    processor = BlockQueueProcessor()
    
    # 1. 初始化blocks到主队列
    processor.initialize_blocks()
    
    # 2. 开始处理
    processor.run_processing()


if __name__ == "__main__":
    main()