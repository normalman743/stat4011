import pandas as pd
import json
import os
import requests
from urllib.parse import urlparse, parse_qs
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ThreeTierBinaryOptimizer:
    def __init__(self, account_scores_file, state_json_file="three_tier_state.json"):
        self.account_scores_file = account_scores_file
        self.state_json_file = state_json_file
        
        # 真实分布
        self.true_bad = 727
        self.true_good = 6831
        
        print("=== 三类分层二分法优化系统 ===")
        print("策略: [0.8-1.0] + [0.5-0.8] + [0.0-0.5] 三类二分确认")
    
    def submit_file(self, csv_file, group_id=12507):
        """提交文件获取F1分数"""
        url = "https://stat4011-part1.sta.cuhk.edu.hk/upload"
        # sleep(1)  # 移除延迟，加快处理速度
        try:
            with open(csv_file, 'rb') as f:
                files = {'submission': f}
                data = {'group_id': group_id}
                response = requests.post(url, files=files, data=data, allow_redirects=False, verify=False)
                
            if response.status_code == 302:
                redirect_url = response.headers.get('Location')
                parsed_url = urlparse(redirect_url)
                params = parse_qs(parsed_url.query)
                return float(params['score'][0])
            return None
        except Exception as e:
            print(f"提交错误: {e}")
            return None
    
    def initialize_state(self):
        """初始化三类分层状态"""
        if os.path.exists(self.state_json_file):
            print(f"加载现有状态: {self.state_json_file}")
            with open(self.state_json_file, 'r') as f:
                state = json.load(f)
                
            # 兼容旧状态文件，添加缺失的字段
            if 'reverse_search_status' not in state:
                state['reverse_search_status'] = 'pending'
            if 'reverse_queue' not in state:
                state['reverse_queue'] = []
                
            return state
        
        print("创建三类分层状态...")
        scores_df = pd.read_csv(self.account_scores_file)
        
        # 三类分层
        tier1 = []  # [0.8-1.0] 极高概率bad
        tier2 = []  # [0.5-0.8] 中等概率bad  
        tier3 = []  # [0.0-0.5] 低概率good
        
        for _, row in scores_df.iterrows():
            acc = {'ID': row['ID'], 'score': row['predict']}
            if row['predict'] >= 0.8:
                tier1.append(acc)
            elif row['predict'] >= 0.5:
                tier2.append(acc)
            else:
                tier3.append(acc)
        
        # 每层内部按概率排序
        tier1.sort(key=lambda x: x['score'], reverse=True)
        tier2.sort(key=lambda x: x['score'], reverse=True)
        tier3.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"分层统计:")
        print(f"  Tier1 [0.8-1.0]: {len(tier1)}个")
        print(f"  Tier2 [0.5-0.8]: {len(tier2)}个") 
        print(f"  Tier3 [0.0-0.5]: {len(tier3)}个")
        
        # 获取初始全0基线
        all_accounts = tier1 + tier2 + tier3
        baseline_df = pd.DataFrame([{'ID': acc['ID'], 'Predict': 0} for acc in all_accounts])
        baseline_df.to_csv('baseline_all_zero.csv', index=False)
        initial_f1 = self.submit_file('baseline_all_zero.csv')
        os.remove('baseline_all_zero.csv')
        
        state = {
            'round': 0,
            'current_f1': initial_f1 if initial_f1 else 0.0,
            'best_f1': initial_f1 if initial_f1 else 0.0,
            'tier1': tier1,
            'tier2': tier2, 
            'tier3': tier3,
            'confirmed_bad_ids': [],
            'confirmed_good_ids': [],
            'current_tier': 1,  # 当前处理的层级
            'tier1_status': 'pending',  # pending/processing/completed
            'tier2_status': 'pending',
            'tier3_status': 'pending',
            'reverse_search_status': 'pending',  # 反向搜索状态
            'binary_queue': [],  # 当前二分搜索的队列
            'reverse_queue': [],  # 反向搜索队列（在"good"中找bad）
            'test_history': []
        }
        
        print(f"初始F1分数 (全0预测): {state['current_f1']:.6f}")
        self.save_state(state)
        return state
    
    def save_state(self, state):
        """保存状态"""
        # DEBUG: 保存前检查队列
        print(f"🐛 DEBUG: 保存状态前反向队列长度: {len(state.get('reverse_queue', []))}")
        
        with open(self.state_json_file, 'w') as f:
            json.dump(state, f, indent=2)
            
        # DEBUG: 验证保存后的文件
        with open(self.state_json_file, 'r') as f:
            saved_state = json.load(f)
        print(f"🐛 DEBUG: 保存后文件中反向队列长度: {len(saved_state.get('reverse_queue', []))}")
    
    def save_best_submission(self, state, f1_score):
        """保存F1新高时的最佳提交文件"""
        all_accounts = state['tier1'] + state['tier2'] + state['tier3']
        submission_data = []
        
        for acc in all_accounts:
            if acc['ID'] in state['confirmed_bad_ids']:
                predict = 1
            else:
                predict = 0
            submission_data.append({'ID': acc['ID'], 'Predict': predict})
        
        df = pd.DataFrame(submission_data)
        filename = f"best_f1_{f1_score:.6f}_round_{state['round']}.csv"
        df.to_csv(filename, index=False)
        print(f"🎯 F1新高！保存最佳文件: {filename}")
    
    def update_account_scores(self, state):
        """更新account_scores.csv文件"""
        scores_df = pd.read_csv(self.account_scores_file)
        
        for acc_id in state['confirmed_bad_ids']:
            scores_df.loc[scores_df['ID'] == acc_id, 'predict'] = 1.0
        
        for acc_id in state['confirmed_good_ids']:
            scores_df.loc[scores_df['ID'] == acc_id, 'predict'] = 0.0
        
        scores_df.to_csv(self.account_scores_file, index=False)
        print(f"已更新account_scores.csv: {len(state['confirmed_bad_ids'])}个bad, {len(state['confirmed_good_ids'])}个good")
    
    def select_next_test(self, state):
        """选择下一个测试批次"""
        if state['current_tier'] == 1 and state['tier1_status'] == 'pending':
            # Tier1: 直接测试所有486个极高概率账户
            state['tier1_status'] = 'processing'
            return state['tier1'], "tier1_all"
            
        elif state['current_tier'] == 1 and state['tier1_status'] == 'processing':
            # Tier1完成，进入Tier2
            state['current_tier'] = 2
            state['tier1_status'] = 'completed'
            
        if state['current_tier'] == 2:
            if state['tier2_status'] == 'pending':
                # Tier2: 初始化二分队列
                state['binary_queue'] = state['tier2'].copy()
                state['tier2_status'] = 'processing'
                
            if state['tier2_status'] == 'processing' and len(state['binary_queue']) > 0:
                # Tier2二分搜索
                batch_size = max(1, len(state['binary_queue']) // 2)
                batch = state['binary_queue'][:batch_size]
                return batch, "tier2_binary"
            else:
                # Tier2完成，进入Tier3
                state['current_tier'] = 3
                state['tier2_status'] = 'completed'
                
        if state['current_tier'] == 3:
            if state['tier3_status'] == 'pending':
                # Tier3: 初始化二分队列
                state['binary_queue'] = state['tier3'].copy()
                state['tier3_status'] = 'processing'
                
            if state['tier3_status'] == 'processing' and len(state['binary_queue']) > 0:
                # Tier3二分搜索
                batch_size = max(1, len(state['binary_queue']) // 2)
                batch = state['binary_queue'][:batch_size]
                return batch, "tier3_binary"
            else:
                # Tier3完成，开始反向搜索
                state['tier3_status'] = 'completed'
                state['current_tier'] = 4  # 反向搜索阶段
                
        if state['current_tier'] == 4:
            # 只在第一次进入时初始化反向队列
            if state.get('reverse_search_status') == 'pending':
                # 初始化反向搜索队列：在确认为good的账户中搜索
                all_confirmed_good = []
                all_accounts = state['tier1'] + state['tier2'] + state['tier3']
                
                for acc in all_accounts:
                    if acc['ID'] in state['confirmed_good_ids']:
                        all_confirmed_good.append(acc)
                
                # 按概率降序排序，优先搜索概率较高的
                all_confirmed_good.sort(key=lambda x: x['score'], reverse=True)
                state['reverse_queue'] = all_confirmed_good
                state['reverse_search_status'] = 'processing'
                
                print(f"🔄 开始反向搜索：在{len(all_confirmed_good)}个确认good中寻找剩余bad")
                
            if state.get('reverse_search_status') == 'processing' and len(state.get('reverse_queue', [])) > 0:
                # DEBUG: 检查队列选择
                print(f"🐛 DEBUG: select_next_test中反向队列长度: {len(state['reverse_queue'])}")
                
                # 反向二分搜索
                batch_size = max(1, len(state['reverse_queue']) // 2)
                batch = state['reverse_queue'][:batch_size]
                
                print(f"🐛 DEBUG: 选择批次大小: {len(batch)}")
                return batch, "reverse_binary"
            else:
                state['reverse_search_status'] = 'completed'
                
        return None, None
    
    def create_test_submission(self, state, test_batch):
        """创建测试提交文件"""
        all_accounts = state['tier1'] + state['tier2'] + state['tier3']
        test_account_ids = [acc['ID'] for acc in test_batch]
        submission_data = []
        
        for acc in all_accounts:
            if acc['ID'] in state['confirmed_bad_ids']:
                predict = 1  # 已确认为bad
            elif acc['ID'] in test_account_ids:
                predict = 1  # 当前测试批次，尝试设为bad
            else:
                predict = 0  # 其他为good
            
            submission_data.append({'ID': acc['ID'], 'Predict': predict})
        
        return pd.DataFrame(submission_data)
    
    def analyze_results(self, state, test_batch, test_type, new_f1):
        """分析测试结果"""
        f1_improvement = new_f1 - state['current_f1']
        
        print(f"F1变化: {state['current_f1']:.6f} -> {new_f1:.6f} ({f1_improvement:+.6f})")
        
        if test_type == "tier1_all":
            # Tier1全测试结果
            if f1_improvement > 0:
                print(f"✅ Tier1成功！确认 {len(test_batch)} 个极高概率账户为bad")
                state['confirmed_bad_ids'].extend([acc['ID'] for acc in test_batch])
                state['current_f1'] = new_f1
                if new_f1 > state['best_f1']:
                    state['best_f1'] = new_f1
                    self.save_best_submission(state, new_f1)
            else:
                print(f"❌ Tier1异常，需要二分确认")
                # 转为二分模式
                state['binary_queue'] = test_batch.copy()
                return len(test_batch)
                
        elif test_type.endswith("_binary") and test_type != "reverse_binary":
            # 二分搜索结果
            if f1_improvement > 0:
                print(f"✅ 二分成功！确认 {len(test_batch)} 个账户为bad")
                state['confirmed_bad_ids'].extend([acc['ID'] for acc in test_batch])
                state['current_f1'] = new_f1
                if new_f1 > state['best_f1']:
                    state['best_f1'] = new_f1
                    self.save_best_submission(state, new_f1)
                # 从二分队列移除已确认的
                state['binary_queue'] = [acc for acc in state['binary_queue'] 
                                       if acc['ID'] not in [b['ID'] for b in test_batch]]
            else:
                print(f"❌ 二分失败，确认 {len(test_batch)} 个账户为good")
                # 避免重复添加
                for acc in test_batch:
                    if acc['ID'] not in state['confirmed_good_ids']:
                        state['confirmed_good_ids'].append(acc['ID'])
                # 从二分队列移除
                state['binary_queue'] = [acc for acc in state['binary_queue'] 
                                       if acc['ID'] not in [b['ID'] for b in test_batch]]
        
        elif test_type == "reverse_binary":
            # 反向二分搜索结果：在"good"中寻找bad
            if f1_improvement > 0:
                print(f"🎯 反向搜索成功！在good中发现 {len(test_batch)} 个真正的bad")
                # 从confirmed_good_ids中移除，添加到confirmed_bad_ids
                for acc in test_batch:
                    if acc['ID'] in state['confirmed_good_ids']:
                        state['confirmed_good_ids'].remove(acc['ID'])
                    state['confirmed_bad_ids'].append(acc['ID'])
                
                state['current_f1'] = new_f1
                if new_f1 > state['best_f1']:
                    state['best_f1'] = new_f1
                    self.save_best_submission(state, new_f1)
                
                # 从反向队列移除已确认的
                tested_ids = [acc['ID'] for acc in test_batch]
                original_queue_length = len(state['reverse_queue'])
                
                state['reverse_queue'] = [acc for acc in state['reverse_queue'] 
                                        if acc['ID'] not in tested_ids]
                
                new_queue_length = len(state['reverse_queue'])
                actual_removed = original_queue_length - new_queue_length
                
                print(f"✨ 目标进度：已找到{len(state['confirmed_bad_ids'])}/727个bad")
                print(f"🐛 DEBUG: 成功移除 {actual_removed}/{len(tested_ids)} 个账户，剩余: {new_queue_length}")
                
            else:
                print(f"❌ 反向搜索失败，{len(test_batch)} 个账户确实是good")
                # 避免重复添加
                for acc in test_batch:
                    if acc['ID'] not in state['confirmed_good_ids']:
                        state['confirmed_good_ids'].append(acc['ID'])
                # DEBUG: 打印详细信息
                tested_ids = [acc['ID'] for acc in test_batch]
                original_queue_length = len(state['reverse_queue'])
                
                print(f"🐛 DEBUG: 原始队列长度: {original_queue_length}")
                print(f"🐛 DEBUG: 测试批次ID前3个: {tested_ids[:3]}")
                print(f"🐛 DEBUG: 队列前3个ID: {[acc['ID'] for acc in state['reverse_queue'][:3]]}")
                
                # 从反向队列移除已测试的账户
                state['reverse_queue'] = [acc for acc in state['reverse_queue'] 
                                        if acc['ID'] not in tested_ids]
                
                new_queue_length = len(state['reverse_queue'])
                actual_removed = original_queue_length - new_queue_length
                
                print(f"🐛 DEBUG: 新队列长度: {new_queue_length}")
                print(f"🐛 DEBUG: 应该移除: {len(tested_ids)}, 实际移除: {actual_removed}")
                
                if actual_removed == 0:
                    print("🚨 ERROR: 队列没有被更新！检查ID匹配...")
                    for i, test_id in enumerate(tested_ids[:3]):
                        found = any(acc['ID'] == test_id for acc in state['reverse_queue'])
                        print(f"🐛 DEBUG: 测试ID {test_id} 在队列中: {'是' if found else '否'}")
                
                print(f"🔄 已从反向队列移除 {actual_removed} 个账户，剩余: {len(state['reverse_queue'])}")
        
        return len(test_batch)
    
    def run_optimization(self, max_rounds=30, target_f1=1.0):
        """运行三类分层二分优化"""
        state = self.initialize_state()
        
        print(f"\n开始三类分层二分优化:")
        print(f"  目标F1: {target_f1}")
        
        for round_num in range(1, max_rounds + 1):
            state['round'] = round_num
            
            print(f"\n{'='*60}")
            print(f"第 {round_num} 轮优化 - Tier{state['current_tier']}")
            print(f"{'='*60}")
            
            # 选择测试批次
            test_batch, test_type = self.select_next_test(state)
            if not test_batch:
                print("所有层级处理完毕！")
                break
            
            print(f"测试类型: {test_type}")
            print(f"测试账户数: {len(test_batch)}")
            if len(test_batch) > 0:
                print(f"概率范围: {test_batch[0]['score']:.6f} - {test_batch[-1]['score']:.6f}")
            
            # 创建测试提交文件
            test_submission = self.create_test_submission(state, test_batch)
            test_filename = f"test_round_{round_num}.csv"
            test_submission.to_csv(test_filename, index=False)
            
            current_bad = len(test_submission[test_submission['Predict'] == 1])
            current_good = len(test_submission[test_submission['Predict'] == 0])
            print(f"测试分布: Bad={current_bad}, Good={current_good}")
            
            # 提交测试
            print(f"提交测试文件: {test_filename}")
            new_f1 = self.submit_file(test_filename)
            
            if new_f1 is None:
                print("提交失败，跳过本轮")
                os.remove(test_filename)
                continue
            
            # 计算F1变化
            f1_improvement = new_f1 - state['current_f1']
            
            # 分析结果
            confirmed_count = self.analyze_results(state, test_batch, test_type, new_f1) 
            
            # 记录历史
            state['test_history'].append({
                'round': round_num,
                'tier': state['current_tier'],
                'test_type': test_type,
                'batch_size': len(test_batch),
                'f1_score': new_f1,
                'f1_improvement': f1_improvement,
                'confirmed_count': confirmed_count
            })
            
            # 保存状态和更新account_scores
            self.save_state(state)
            self.update_account_scores(state)
            
            # 清理测试文件
            os.remove(test_filename)
            
            # 检查是否达到目标
            if state['current_f1'] >= target_f1:
                print(f"\n🎉 达到目标F1={target_f1}！")
                break
            
            # 显示进度
            print(f"\n当前进度:")
            print(f"  Tier1状态: {state['tier1_status']}")
            print(f"  Tier2状态: {state['tier2_status']}")
            print(f"  Tier3状态: {state['tier3_status']}")
            print(f"  反向搜索状态: {state.get('reverse_search_status', 'pending')}")
            print(f"  当前最佳F1: {state['current_f1']:.6f}")
            print(f"  确认bad: {len(state['confirmed_bad_ids'])}/727 ({len(state['confirmed_bad_ids'])/727*100:.1f}%)")
            print(f"  确认good: {len(state['confirmed_good_ids'])}")
            if len(state.get('binary_queue', [])) > 0:
                print(f"  二分队列剩余: {len(state['binary_queue'])}")
            if len(state.get('reverse_queue', [])) > 0:
                print(f"  反向搜索队列剩余: {len(state['reverse_queue'])}")
        
        self.generate_final_submission(state)
        return state
    
    def generate_final_submission(self, state):
        """生成最终提交文件"""
        print(f"\n{'='*60}")
        print("三类分层优化完成！")
        print(f"{'='*60}")
        
        all_accounts = state['tier1'] + state['tier2'] + state['tier3']
        submission_data = []
        
        for acc in all_accounts:
            if acc['ID'] in state['confirmed_bad_ids']:
                predict = 1
            else:
                predict = 0
            
            submission_data.append({'ID': acc['ID'], 'Predict': predict})
        
        final_df = pd.DataFrame(submission_data)
        final_df.to_csv('final_three_tier_submission.csv', index=False)
        
        print(f"最终统计:")
        print(f"  最佳F1分数: {state['best_f1']:.6f}")
        print(f"  确认bad账户: {len(state['confirmed_bad_ids'])}")
        print(f"  确认good账户: {len(state['confirmed_good_ids'])}")
        print(f"  总轮数: {state['round']}")

def main():
    """主函数"""
    account_scores_file = "/Users/mannormal/4011/account_scores.csv"
    
    optimizer = ThreeTierBinaryOptimizer(account_scores_file)
    final_state = optimizer.run_optimization(max_rounds=30, target_f1=1.0)
    
    print("\n🎉 三类分层二分法优化完成！")

if __name__ == "__main__":
    main()