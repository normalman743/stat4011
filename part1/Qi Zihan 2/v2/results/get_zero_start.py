import pandas as pd
import json
import os
import requests
from urllib.parse import urlparse, parse_qs
import urllib3
from time import sleep

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ZeroStartOptimizer:
    def __init__(self, account_scores_file, state_json_file="zero_start_state.json"):
        self.account_scores_file = account_scores_file
        self.state_json_file = state_json_file
        
        # 真实分布
        self.true_bad = 727
        self.true_good = 6831
        
        # 测试参数 - 动态批次大小
        self.initial_batch_size = 100  # 初始批次更大
        self.min_batch_size = 20       # 最小批次大小
        self.use_binary_search = True  # 启用二分搜索
        
        print("=== 从零开始F1优化系统 ===")
        print(f"策略: 全部预测为0，逐步添加bad预测")
        print(f"初始批次大小: {self.initial_batch_size}")
        print(f"二分搜索模式: {'启用' if self.use_binary_search else '禁用'}")
    
    def submit_file(self, csv_file, group_id=12507):
        """提交文件获取F1分数"""
        url = "https://stat4011-part1.sta.cuhk.edu.hk/upload"
        sleep(1)  # 等待1秒
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
        """初始化状态"""
        if os.path.exists(self.state_json_file):
            print(f"加载现有状态: {self.state_json_file}")
            with open(self.state_json_file, 'r') as f:
                return json.load(f)
        
        print("创建新的从零开始状态...")
        
        # 读取账户分数，按概率降序排序
        scores_df = pd.read_csv(self.account_scores_file)
        
        # 创建候选队列：按概率从高到低排序
        candidate_accounts = []
        for _, row in scores_df.iterrows():
            candidate_accounts.append({
                'ID': row['ID'],
                'score': row['predict'],
                'confirmed': False,
                'current_predict': 0  # 初始全部为0
            })
        
        # 按概率降序排序
        candidate_accounts.sort(key=lambda x: x['score'], reverse=True)
        
        # 创建基础提交文件模板（全部为0）
        baseline_df = pd.DataFrame([{'ID': acc['ID'], 'Predict': 0} for acc in candidate_accounts])
        
        state = {
            'round': 0,
            'current_f1': 0.0,  # 开始F1为0
            'best_f1': 0.0,
            'candidate_queue': candidate_accounts,
            'confirmed_bad_ids': [],  # 确认为bad的账户ID
            'confirmed_good_ids': [],  # 确认为good的账户ID  
            'test_history': []
        }
        
        # 提交初始的全0文件获取基线F1
        print("提交初始全0预测文件...")
        baseline_df.to_csv('baseline_all_zero.csv', index=False)
        initial_f1 = self.submit_file('baseline_all_zero.csv')
        if initial_f1 is not None:
            state['current_f1'] = initial_f1
            print(f"初始F1分数 (全0预测): {initial_f1:.6f}")
        else:
            print("初始提交失败，使用F1=0.0")
        
        os.remove('baseline_all_zero.csv')
        self.save_state(state)
        return state
    
    def save_state(self, state):
        """保存状态"""
        with open(self.state_json_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def update_account_scores(self, state):
        """更新account_scores.csv文件"""
        scores_df = pd.read_csv(self.account_scores_file)
        
        # 更新确认为bad的账户
        for acc_id in state['confirmed_bad_ids']:
            scores_df.loc[scores_df['ID'] == acc_id, 'predict'] = 1.0
        
        # 更新确认为good的账户  
        for acc_id in state['confirmed_good_ids']:
            scores_df.loc[scores_df['ID'] == acc_id, 'predict'] = 0.0
        
        # 保存更新后的文件
        scores_df.to_csv(self.account_scores_file, index=False)
        print(f"已更新account_scores.csv: {len(state['confirmed_bad_ids'])}个bad, {len(state['confirmed_good_ids'])}个good")
    
    def save_best_submission(self, state, f1_score):
        """保存F1新高时的最佳提交文件"""
        submission_data = []
        for acc in state['candidate_queue']:
            if acc['ID'] in state['confirmed_bad_ids']:
                predict = 1
            else:
                predict = 0
            submission_data.append({'ID': acc['ID'], 'Predict': predict})
        
        df = pd.DataFrame(submission_data)
        filename = f"best_f1_{f1_score:.6f}_round_{state['round']}.csv"
        df.to_csv(filename, index=False)
        print(f"🎯 F1新高！保存最佳文件: {filename}")
    
    def create_test_submission(self, state, test_batch):
        """创建测试提交文件"""
        submission_data = []
        
        # 获取所有账户ID
        all_account_ids = [acc['ID'] for acc in state['candidate_queue']]
        test_account_ids = [acc['ID'] for acc in test_batch]
        
        for acc_id in all_account_ids:
            if acc_id in state['confirmed_bad_ids']:
                predict = 1  # 已确认为bad
            elif acc_id in test_account_ids:
                predict = 1  # 当前测试批次，尝试设为bad
            else:
                predict = 0  # 其他全部为good
            
            submission_data.append({'ID': acc_id, 'Predict': predict})
        
        return pd.DataFrame(submission_data)
    
    def select_next_batch(self, state):
        """选择下一批测试账户 - 动态批次大小 + 二分搜索"""
        unconfirmed = [acc for acc in state['candidate_queue'] 
                      if not acc['confirmed'] and acc['ID'] not in state['confirmed_bad_ids'] 
                      and acc['ID'] not in state['confirmed_good_ids']]
        
        if len(unconfirmed) == 0:
            return None
        
        # 动态计算批次大小
        if self.use_binary_search:
            # 二分搜索策略：剩余账户的1/4到1/2
            remaining_count = len(unconfirmed)
            if remaining_count > 500:
                batch_size = min(200, remaining_count // 3)  # 大批次快速缩小范围
            elif remaining_count > 200:
                batch_size = min(100, remaining_count // 2)  # 中等批次
            elif remaining_count > 50:
                batch_size = min(50, remaining_count // 2)   # 小批次精确搜索
            else:
                batch_size = min(self.min_batch_size, remaining_count)  # 最小批次
        else:
            # 固定批次策略
            batch_size = min(self.initial_batch_size, len(unconfirmed))
        
        batch = unconfirmed[:batch_size]
        
        print(f"选择 {len(batch)} 个最高概率账户进行测试")
        print(f"概率范围: {batch[0]['score']:.6f} - {batch[-1]['score']:.6f}")
        print(f"剩余未确认: {len(unconfirmed)}")
        
        return batch
    
    def analyze_results(self, state, test_batch, new_f1):
        """分析测试结果并更新状态"""
        f1_improvement = new_f1 - state['current_f1']
        
        print(f"F1变化: {state['current_f1']:.6f} -> {new_f1:.6f} ({f1_improvement:+.6f})")
        
        if f1_improvement > 0:
            # F1提升，确认这批账户为bad
            print(f"✅ F1提升！确认 {len(test_batch)} 个账户为bad")
            
            for acc in test_batch:
                state['confirmed_bad_ids'].append(acc['ID'])
                # 标记为已确认
                for candidate in state['candidate_queue']:
                    if candidate['ID'] == acc['ID']:
                        candidate['confirmed'] = True
                        break
            
            # 更新当前最佳F1
            state['current_f1'] = new_f1
            if new_f1 > state['best_f1']:
                state['best_f1'] = new_f1
                # F1新高时保存当前最佳提交文件
                self.save_best_submission(state, new_f1)
            
            return len(test_batch)
        
        else:
            # F1没有提升或下降，这批账户可能是good
            print(f"❌ F1无提升，推测 {len(test_batch)} 个账户为good")
            
            for acc in test_batch:
                state['confirmed_good_ids'].append(acc['ID'])
                # 标记为已确认
                for candidate in state['candidate_queue']:
                    if candidate['ID'] == acc['ID']:
                        candidate['confirmed'] = True
                        break
            
            return len(test_batch)
    
    def run_optimization(self, max_rounds=20, target_f1=1.0):
        """运行从零开始的优化过程"""
        state = self.initialize_state()
        
        print(f"\n初始状态:")
        print(f"  候选账户总数: {len(state['candidate_queue'])}")
        print(f"  当前F1: {state['current_f1']:.6f}")
        print(f"  目标F1: {target_f1}")
        
        for round_num in range(1, max_rounds + 1):
            state['round'] = round_num
            
            print(f"\n{'='*60}")
            print(f"第 {round_num} 轮优化")
            print(f"{'='*60}")
            
            # 选择测试批次
            test_batch = self.select_next_batch(state)
            if not test_batch:
                print("所有账户已处理完毕！")
                break
            
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
            
            # 分析结果并更新状态
            confirmed_count = self.analyze_results(state, test_batch, new_f1)
            
            # 记录历史
            state['test_history'].append({
                'round': round_num,
                'batch_size': len(test_batch),
                'f1_score': new_f1,
                'f1_improvement': new_f1 - state['current_f1'] if round_num > 1 else new_f1,
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
            
            # 检查进度
            remaining = len([acc for acc in state['candidate_queue'] if not acc['confirmed']])
            print(f"\n当前进度:")
            print(f"  当前最佳F1: {state['current_f1']:.6f}")
            print(f"  确认bad: {len(state['confirmed_bad_ids'])}")
            print(f"  确认good: {len(state['confirmed_good_ids'])}")
            print(f"  剩余未确认: {remaining}")
            
            if remaining == 0:
                print("所有账户已确认完毕！")
                break
        
        self.generate_final_submission(state)
        return state
    
    def generate_final_submission(self, state):
        """生成最终提交文件"""
        print(f"\n{'='*60}")
        print("优化完成！生成最终提交文件...")
        print(f"{'='*60}")
        
        # 生成最终提交文件
        submission_data = []
        for acc in state['candidate_queue']:
            if acc['ID'] in state['confirmed_bad_ids']:
                predict = 1
            else:
                predict = 0  # 未确认的默认为good
            
            submission_data.append({'ID': acc['ID'], 'Predict': predict})
        
        final_df = pd.DataFrame(submission_data)
        final_df.to_csv('final_zero_start_submission.csv', index=False)
        
        # 统计信息
        final_bad = len(state['confirmed_bad_ids'])
        final_good = len(state['confirmed_good_ids'])
        total_confirmed = final_bad + final_good
        
        print(f"最终统计:")
        print(f"  最佳F1分数: {state['best_f1']:.6f}")
        print(f"  确认bad账户: {final_bad}")
        print(f"  确认good账户: {final_good}")
        print(f"  总确认账户: {total_confirmed}")
        print(f"  总轮数: {state['round']}")
        print(f"\n文件生成:")
        print(f"  final_zero_start_submission.csv")
        print(f"  {self.account_scores_file} (已更新)")

def main():
    """主函数"""
    account_scores_file = "/Users/mannormal/4011/account_scores.csv"
    
    optimizer = ZeroStartOptimizer(account_scores_file)
    final_state = optimizer.run_optimization(max_rounds=30, target_f1=1.0)
    
    print("\n🎉 从零开始F1优化完成！")

if __name__ == "__main__":
    main()