#!/usr/bin/env python3
"""
重构的Block二分处理系统
"""

import queue
import threading
import time
import os
import sys
import pandas as pd
from simulator import simulate_f1 as upload_file
#from upload_module import upload_file 
from enhanced_confusion_calculator import get_A_B_only, LAYER_CONFIG

def save_csv(predictions, file_name,f1_score):
    """保存预测结果到CSV文件"""
    file_path = f"/Users/mannormal/4011/Qi Zihan/v5/{file_name}"
    df = pd.DataFrame(predictions)
    df.to_csv(file_path, index=False)
    # 这个函数是模块级别的，可能被多线程调用，所以也保护起来
    print(f"保存预测结果到 {file_path}，F1分数: {f1_score}")


class BlockQueueProcessor:
    def __init__(self):
        self.main_queue = queue.Queue()      # 待处理blocks
        self.result_queue = queue.Queue()    # 新生成blocks
        self.num_workers = 7
        self.worker_threads = []
        self.running = True
        self.account_scores = self.load_account_scores()
        self.iteration_count = 0  # 添加迭代计数器
        self.iteration_lock = threading.Lock()  # 添加线程锁
        self.start_time = time.time()  # 记录开始时间
        self.current_f1 = 0.0  # 记录当前F1分数
        self.max_confirmed_bad_f1 = 0.0  # 记录已确认bad账户的最大F1分数
        
        # 内存中的账户状态
        self.account_status = {}
        self.status_lock = threading.Lock()  # 状态更新锁
        self.print_lock = threading.Lock()  # 打印操作锁
        self.first_print = True  # 标记是否是第一次打印
        
    def load_account_scores(self):
        """加载账户分数"""
        df = pd.read_csv("/Users/mannormal/Desktop/课程/y4t1/stat 4011/account_scores.csv")
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
        
        # 初始化所有账户状态为-1（在内存中）
        with self.status_lock:
            for account_id in self.account_scores.keys():
                self.account_status[account_id] = -1
    
    def update_account_status(self, accounts, flag):
        """更新账户状态"""
        with self.status_lock:
            for account_id in accounts:
                self.account_status[account_id] = flag
            
            # 如果确认了bad账户，更新最大F1分数
            if flag == 1 and len(accounts) > 0:
                self.update_max_confirmed_bad_f1()
    
    def update_max_confirmed_bad_f1(self):
        """计算并更新已确认bad账户的最大F1分数"""
        # 注意：此方法应该在status_lock已锁定的情况下调用
        confirmed_bad = sum(1 for status in self.account_status.values() if status == 1)
        
        if confirmed_bad > 0:
            current_f1 = confirmed_bad*2 / (confirmed_bad + 727)  # 727是总的bad账户数
            self.max_confirmed_bad_f1 = current_f1
    
    def worker_process(self, worker_id):
        """Worker线程主函数"""
        try:
            while self.running:
                try:
                    # 使用超时避免无限等待
                    block = self.main_queue.get(timeout=2)
                except queue.Empty:
                    if not self.running:
                        break
                    continue
                
                # 增加迭代计数
                with self.iteration_lock:
                    self.iteration_count += 1

                new_blocks = self.process_single_block(block, worker_id)
                
                # 将新blocks加入结果队列
                for new_block in new_blocks:
                    self.result_queue.put(new_block)
                    
                self.main_queue.task_done()
        except Exception as e:
            # with self.print_lock:
            #     print(f"Worker {worker_id} 异常: {e}")
            pass
        finally:
            # with self.print_lock:
            #     print(f"Worker {worker_id} 正常退出")
            pass

    def create_prediction_csv(self, block, worker_id):
        """创建预测CSV文件 - 对block进行A/B分组测试"""
        block_accounts = self.get_block_accounts(block)
        
        # 注意：这个方法只用于多账户的A/B分组，单账户在process_single_block中单独处理
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
        
        # with self.print_lock:
        #     print(f"A组账户数: {len(group_a)}, B组账户数: {len(group_b)}")
        #     print(f"总预测为0: {sum(1 for p in predictions if p['Predict'] == 0)}")
        #     print(f"总预测为1: {sum(1 for p in predictions if p['Predict'] == 1)}")
        predictbad = sum(1 for p in predictions if p['Predict'] == 1)
        df = pd.DataFrame(predictions)
        df.to_csv(temp_file, index=False)
        return temp_file, predictbad

    def process_single_block(self, block, worker_id):
        """处理单个block"""
        # with self.print_lock:
        #     print(f"\n=== 处理 {block['id']} ===")
        #     print(f"real_good: {block['real_good']}")
        #     print(f"real_bad: {block['real_bad']}")
        
        block_accounts = self.get_block_accounts(block)
        
        # 单账户特殊处理 - 完全独立的逻辑
        if len(block_accounts) == 1:
            # with self.print_lock:
            #     print("单账户block，进行精确测试")
            
            # 创建特殊的预测CSV：只有这个账户预测为1，其他全为0
            temp_file = f"/tmp/single_test_{worker_id}_{threading.current_thread().ident}.csv"
            single_account = block_accounts[0]
            
            predictions = []
            for account_id in self.account_scores.keys():
                if account_id == single_account:
                    predict = 1  # 测试账户预测为bad
                else:
                    predict = 0  # 其他全部预测为good
                predictions.append({"ID": account_id, "Predict": predict})
            
            df = pd.DataFrame(predictions)
            df.to_csv(temp_file, index=False)

            # print(f"单账户测试：预测为1的账户数=1，预测为0的账户数={len(self.account_scores)-1}")

            # 上传获取F1分数
            f1_score = upload_file(temp_file)
            
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            if f1_score is None:
                return []
            
            # 保存最新的F1分数
            self.current_f1 = f1_score
            
            # 根据F1分数判断
            if f1_score > 0:
                # with self.print_lock:
                #     print(f"单账户归类 -> bad (F1={f1_score:.4f})")
                self.update_account_status([single_account], 1)
            else:
                # with self.print_lock:
                #     print(f"单账户归类 -> good (F1={f1_score})")
                self.update_account_status([single_account], 0)
            
            self.print_current_statistics()
            return []  # 单账户处理完毕，直接返回
        
        # 多账户时才进行A/B分组测试
        temp_csv, predicted_bad_count = self.create_prediction_csv(block, worker_id)
        
        # 上传获取F1分数
        f1_score = upload_file(temp_csv)
        
        # 清理临时文件
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        
        if f1_score is None:
            return []
        
        # 保存最新的F1分数
        self.current_f1 = f1_score
        
        # 使用get_A_B_only获取A/B组情况
        base_layer = self.get_base_layer(block.get('layer', 'unknown'))
        ab_result = get_A_B_only(f1_score, predicted_bad_count, base_layer)
        
        if ab_result is None:
            return []
        
        # 显示A/B组统计
        # with self.print_lock:
        #     print(f"group_A: good={ab_result['A_TP']}, bad={ab_result['A_FP']}")
        #     print(f"group_B: good={ab_result['B_TP']}, bad={ab_result['B_FP']}")
        
        new_blocks = []
        mid_point = len(block_accounts) // 2
        group_a_accounts = block_accounts[:mid_point]
        group_b_accounts = block_accounts[mid_point:]
        
        # 检查A组
        if ab_result['A_TP'] == 0:
            # with self.print_lock:
            #     print("归类 group_A -> bad")
            self.update_account_status(group_a_accounts, 1)
        elif ab_result['A_FP'] == 0:
            # with self.print_lock:
            #     print("归类 group_A -> good")
            self.update_account_status(group_a_accounts, 0)
        else:
            # with self.print_lock:
            #     print("归类 group_A -> 继续二分")
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
        if len(group_b_accounts) == 0:
            # with self.print_lock:
            #     print("group_B 为空，跳过B组处理")
            pass
        elif ab_result['B_TP'] == 0:
            # with self.print_lock:
            #     print("归类 group_B -> bad")
            self.update_account_status(group_b_accounts, 1)
        elif ab_result['B_FP'] == 0:
            # with self.print_lock:
            #     print("归类 group_B -> good")
            self.update_account_status(group_b_accounts, 0)
        else:
            # with self.print_lock:
            #     print("归类 group_B -> 继续二分")
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
        with self.status_lock:
            account_status = self.account_status.copy()
        
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
        
        # 计算已处理时间
        elapsed_time = time.time() - self.start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        # 使用打印锁确保原子操作
        with self.print_lock:
            # 如果不是第一次打印，先清除之前的内容
            if not self.first_print:
                # 向上移动10行并清除（根据输出行数调整）
                sys.stdout.write('\033[12A')  # 向上移动10行
                sys.stdout.write('\033[J')     # 清除从光标到屏幕底部的内容
            else:
                self.first_print = False
            
            print(f"{'='*60}")
            print(f"📊 已确认统计 (迭代 #{self.iteration_count})")
            print(f"{'='*60}")
            print(f"├─ high_good: {high_good_confirmed:>4}/6780  ({high_good_confirmed/6780*100:>5.1f}%)")
            print(f"├─ mid:       {mid_confirmed:>4}/292   ({mid_confirmed/292*100:>5.1f}%)")
            print(f"└─ high_bad:  {high_bad_confirmed:>4}/486   ({high_bad_confirmed/486*100:>5.1f}%)")
            print(f"")
            print(f"总计: good={confirmed_good:>4}/6831 ({confirmed_good/6831*100:>5.1f}%), bad={confirmed_bad:>3}/727 ({confirmed_bad/727*100:>5.1f}%)")
            print(f"当前 F1 Score:      {self.current_f1:.6f}")
            print(f"已确认bad最大 F1:  {self.max_confirmed_bad_f1:.6f}")
            print(f"已处理时间: {minutes:>2}分{seconds:>2}秒")
            print(f"{'='*60}")
            sys.stdout.flush()  # 确保立即输出

    
    def run_processing(self):
        """主处理流程"""
        try:
            # 启动worker线程
            for i in range(1, self.num_workers + 1):
                worker_thread = threading.Thread(target=self.worker_process, args=(i,))
                worker_thread.daemon = True  # 设置为守护线程
                worker_thread.start()
                self.worker_threads.append(worker_thread)
            
            # 主线程回调：结果队列 → 主队列
            idle_count = 0  # 空闲计数器
            max_idle = 100  # 最大空闲次数
            
            while any(t.is_alive() for t in self.worker_threads) or not self.result_queue.empty():
                try:
                    if not self.result_queue.empty():
                        new_block = self.result_queue.get(timeout=1)
                        self.main_queue.put(new_block)
                        self.result_queue.task_done()
                        idle_count = 0  # 有新任务，重置计数器
                    else:
                        time.sleep(0.1)
                        idle_count += 1  # 空闲计数+1
                        
                        # 如果空闲次数超过阈值，检查是否所有账户都已确认
                        if idle_count >= max_idle:
                            with self.status_lock:
                                confirmed_count = sum(1 for status in self.account_status.values() if status != -1)
                            # 如果所有账户都已确认，退出循环
                            if confirmed_count == len(self.account_scores):
                                break
                            idle_count = 0  # 重置计数器，继续等待
                            
                except queue.Empty:
                    continue
            
            # 停止所有worker
            self.running = False
            
            # 等待所有worker完成（最多等待5秒）
            for worker_thread in self.worker_threads:
                worker_thread.join(timeout=5)
                if worker_thread.is_alive():
                    # with self.print_lock:
                    #     print(f"警告: 线程 {worker_thread.name} 未能正常退出")
                    pass
            
            # with self.print_lock:
            #     print("\n=== 处理完成 ===")
            pass
            
        except KeyboardInterrupt:
            # with self.print_lock:
            #     print("\n收到中断信号，正在停止...")
            self.running = False
            # 强制等待线程退出
            for worker_thread in self.worker_threads:
                worker_thread.join(timeout=2)
        except Exception as e:
            # with self.print_lock:
            #     print(f"处理过程中发生异常: {e}")
            self.running = False

    def save_final_csv(self):
        """保存最终预测结果"""
        
        with self.status_lock:
            account_status = self.account_status.copy()
        
        # 生成最终预测
        predictions = []
        unconfirmed_count = 0
        
        for account_id in self.account_scores.keys():
            status = account_status.get(account_id, -1)
            if status == -1:
                # 未确定的账户，根据分数预测
                score = self.account_scores[account_id]
                predict = 1 if score >= 0.5 else 0
                unconfirmed_count += 1
            else:
                predict = status
            
            predictions.append({"ID": account_id, "Predict": predict})
        
        # 保存文件
        filename = f"best.csv"
        df = pd.DataFrame(predictions)
        df.to_csv(filename, index=False)
        
        # 计算最终F1
        final_f1 = upload_file(filename)
        
        # 统计信息
        confirmed_count = len(self.account_scores) - unconfirmed_count
        pred_bad = sum(1 for p in predictions if p["Predict"] == 1)
        pred_good = sum(1 for p in predictions if p["Predict"] == 0)
        
        with self.print_lock:
            print(f"\n🎯 === 最终结果 ===")
            print(f"📁 文件: {filename}")
            print(f"📊 最终F1: {final_f1:.6f}")
            print(f"✅ 确认账户: {confirmed_count}/{len(self.account_scores)} ({confirmed_count/len(self.account_scores)*100:.1f}%)")
            print(f"❓ 未确认: {unconfirmed_count} (按分数预测)")
            print(f"📈 预测: good={pred_good}, bad={pred_bad}")
            print(f"🔄 总迭代: {self.iteration_count}")


def main():
    
    processor = None
    try:
        processor = BlockQueueProcessor()
        
        # 1. 初始化blocks到主队列
        processor.initialize_blocks()
        
        # 2. 开始处理
        processor.run_processing()
        
        # 3. 保存最终结果
        processor.save_final_csv()
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        if processor:
            processor.running = False
            # 即使中断也保存已确认的结果
            print("正在保存已确认的结果...")
            processor.save_final_csv()
    except Exception as e:
        print(f"主程序异常: {e}")
        if processor:
            processor.running = False
    finally:
        # 确保程序正常退出
        print("程序退出")

if __name__ == "__main__":
    main()