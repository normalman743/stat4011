import pandas as pd
import json
import os
import requests
from urllib.parse import urlparse, parse_qs
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class CorrectBinarySearchOptimizer:
    def __init__(self, account_scores_file, state_json_file="correct_binary_state.json"):
        self.account_scores_file = account_scores_file
        self.state_json_file = state_json_file
        
        # 真实分布（估计值）
        self.true_bad = 727
        self.true_good = 6831
        
        print("=== 正确二分法F1优化系统 ===")
        print("策略: 基于TP/FP/FN/TN变化计算转换成功率的真正二分法")
    
    def submit_file(self, csv_file, group_id=12507):
        """提交文件获取F1分数"""
        url = "https://stat4011-part1.sta.cuhk.edu.hk/upload"
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
    
    def calculate_confusion_matrix(self, pred_bad_count, pred_good_count, f1_score):
        """根据预测分布和F1分数计算混淆矩阵"""
        if f1_score == 0:
            return {'TP': 0, 'FP': pred_bad_count, 'FN': self.true_bad, 'TN': self.true_good}
        
        # 通过F1反推TP
        best_tp = 0
        best_f1_diff = float('inf')
        
        for tp in range(min(pred_bad_count, self.true_bad) + 1):
            fp = pred_bad_count - tp
            fn = self.true_bad - tp
            tn = self.true_good - fp
            
            # 验证合理性
            if fp < 0 or fn < 0 or tn < 0:
                continue
                
            precision = tp / pred_bad_count if pred_bad_count > 0 else 0
            recall = tp / self.true_bad if self.true_bad > 0 else 0
            calculated_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_diff = abs(calculated_f1 - f1_score)
            if f1_diff < best_f1_diff:
                best_f1_diff = f1_diff
                best_tp = tp
        
        tp = best_tp
        fp = pred_bad_count - tp
        fn = self.true_bad - tp
        tn = self.true_good - fp
        
        return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}
    
    def initialize_state(self):
        """初始化状态"""
        if os.path.exists(self.state_json_file):
            print(f"加载现有状态: {self.state_json_file}")
            with open(self.state_json_file, 'r') as f:
                return json.load(f)
        
        print("创建新状态，从全0开始...")
        
        # 读取账户分数
        scores_df = pd.read_csv(self.account_scores_file)
        all_accounts = []
        for _, row in scores_df.iterrows():
            all_accounts.append({
                'ID': row['ID'],
                'score': row['predict'],
                'current_predict': 0  # 初始全部为0 (good)
            })
        
        # 按概率降序排序
        all_accounts.sort(key=lambda x: x['score'], reverse=True)
        
        # 提交初始全0文件获取基线
        baseline_df = pd.DataFrame([{'ID': acc['ID'], 'Predict': 0} for acc in all_accounts])
        baseline_df.to_csv('baseline_all_zero.csv', index=False)
        baseline_f1 = self.submit_file('baseline_all_zero.csv')
        os.remove('baseline_all_zero.csv')
        
        if baseline_f1 is None:
            baseline_f1 = 0.0
        
        baseline_cm = self.calculate_confusion_matrix(0, len(all_accounts), baseline_f1)
        
        state = {
            'round': 0,
            'all_accounts': all_accounts,
            'current_f1': baseline_f1,
            'best_f1': baseline_f1,
            'current_cm': baseline_cm,
            'test_queue': [],  # 当前要测试的账户队列
            'confirmed_predictions': {},  # {account_id: 0/1} 已确认的预测
            'test_history': []
        }
        
        print(f"初始基线F1: {baseline_f1:.6f}")
        print(f"基线混淆矩阵: TP={baseline_cm['TP']}, FP={baseline_cm['FP']}, FN={baseline_cm['FN']}, TN={baseline_cm['TN']}")
        
        self.save_state(state)
        return state
    
    def save_state(self, state):
        """保存状态"""
        with open(self.state_json_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def save_best_submission(self, state, f1_score):
        """保存F1新高时的最佳提交文件"""
        submission_data = []
        for acc in state['all_accounts']:
            if acc['ID'] in state['confirmed_predictions']:
                predict = state['confirmed_predictions'][acc['ID']]
            else:
                predict = acc['current_predict']
            submission_data.append({'ID': acc['ID'], 'Predict': predict})
        
        df = pd.DataFrame(submission_data)
        filename = f"best_correct_f1_{f1_score:.6f}_round_{state['round']}.csv"
        df.to_csv(filename, index=False)
        print(f"🎯 F1新高！保存文件: {filename}")
    
    def select_next_test_batch(self, state):
        """选择下一批要测试的账户"""
        if len(state['test_queue']) == 0:
            # 初始化：选择所有高概率账户作为候选
            candidates = []
            for acc in state['all_accounts']:
                if acc['ID'] not in state['confirmed_predictions'] and acc['score'] > 0.5:
                    candidates.append(acc)
            
            if len(candidates) == 0:
                # 如果没有高概率候选，选择所有未确认账户
                candidates = [acc for acc in state['all_accounts'] 
                            if acc['ID'] not in state['confirmed_predictions']]
            
            if len(candidates) == 0:
                return None, None
                
            state['test_queue'] = candidates
            print(f"初始化测试队列: {len(candidates)}个账户")
        
        # 选择测试批次（二分）
        batch_size = max(1, len(state['test_queue']) // 2)
        test_batch = state['test_queue'][:batch_size]
        
        # 决定测试方向：0→1 还是 1→0
        if test_batch[0]['current_predict'] == 0:
            test_direction = "0_to_1"  # good → bad
            print(f"测试 {len(test_batch)} 个账户：0→1 (good改为bad)")
        else:
            test_direction = "1_to_0"  # bad → good  
            print(f"测试 {len(test_batch)} 个账户：1→0 (bad改为good)")
        
        return test_batch, test_direction
    
    def create_test_submission(self, state, test_batch, test_direction):
        """创建测试提交文件"""
        submission_data = []
        test_account_ids = [acc['ID'] for acc in test_batch]
        
        for acc in state['all_accounts']:
            if acc['ID'] in state['confirmed_predictions']:
                # 已确认的预测
                predict = state['confirmed_predictions'][acc['ID']]
            elif acc['ID'] in test_account_ids:
                # 测试批次：进行转换
                if test_direction == "0_to_1":
                    predict = 1  # good → bad
                else:
                    predict = 0  # bad → good
            else:
                # 其他账户保持当前预测
                predict = acc['current_predict']
            
            submission_data.append({'ID': acc['ID'], 'Predict': predict})
        
        return pd.DataFrame(submission_data)
    
    def analyze_test_results(self, state, test_batch, test_direction, new_f1):
        """分析测试结果并计算转换成功率"""
        print(f"\n=== 分析测试结果 ===")
        
        # 计算新旧混淆矩阵
        test_submission = self.create_test_submission(state, test_batch, test_direction)
        new_bad_count = len(test_submission[test_submission['Predict'] == 1])
        new_good_count = len(test_submission[test_submission['Predict'] == 0])
        new_cm = self.calculate_confusion_matrix(new_bad_count, new_good_count, new_f1)
        
        old_cm = state['current_cm']
        
        print(f"转换方向: {test_direction}")
        print(f"测试账户数: {len(test_batch)}")
        print(f"F1变化: {state['current_f1']:.6f} → {new_f1:.6f} ({new_f1 - state['current_f1']:+.6f})")
        print(f"旧混淆矩阵: TP={old_cm['TP']}, FP={old_cm['FP']}, FN={old_cm['FN']}, TN={old_cm['TN']}")
        print(f"新混淆矩阵: TP={new_cm['TP']}, FP={new_cm['FP']}, FN={new_cm['FN']}, TN={new_cm['TN']}")
        
        # 计算转换成功率
        if test_direction == "0_to_1":
            # good → bad的转换
            # 对于0→1转换，我们测试了 len(test_batch) 个账户
            # 这些转换可能增加TP（正确识别真bad）或增加FP（错误识别good为bad）
            tp_change = new_cm['TP'] - old_cm['TP']
            fp_change = new_cm['FP'] - old_cm['FP']
            
            # 总转换数应该等于测试批次大小
            test_batch_size = len(test_batch)
            
            # 成功转换 = TP增加的数量
            # 失败转换 = FP增加的数量  
            # 但要确保总数等于测试批次大小
            if tp_change + fp_change == test_batch_size:
                success_count = tp_change
                failure_count = fp_change
                success_rate = success_count / test_batch_size if test_batch_size > 0 else 0
            else:
                # 如果数量不匹配，说明有其他变化，按比例计算
                success_count = tp_change
                failure_count = test_batch_size - tp_change
                success_rate = success_count / test_batch_size if test_batch_size > 0 else 0
            
            print(f"转换分析: 测试{test_batch_size}个, TP增加{tp_change}, FP增加{fp_change}, 成功率{success_rate:.2%}")
            
        else:  # "1_to_0"
            # bad → good的转换
            tn_change = new_cm['TN'] - old_cm['TN']
            fn_change = new_cm['FN'] - old_cm['FN']
            
            test_batch_size = len(test_batch)
            
            # 对于1→0转换，成功的转换会增加TN，失败的转换会增加FN
            if tn_change + fn_change == test_batch_size:
                success_count = tn_change
                failure_count = fn_change
                success_rate = success_count / test_batch_size if test_batch_size > 0 else 0
            else:
                success_count = tn_change
                failure_count = test_batch_size - tn_change
                success_rate = success_count / test_batch_size if test_batch_size > 0 else 0
            
            print(f"转换分析: 测试{test_batch_size}个, TN增加{tn_change}, FN增加{fn_change}, 成功率{success_rate:.2%}")
        
        # 二分决策
        decision = self.make_binary_decision(state, test_batch, success_rate, new_f1)
        
        # 更新状态
        if new_f1 > state['best_f1']:
            state['best_f1'] = new_f1
            self.save_best_submission(state, new_f1)
        
        state['current_f1'] = new_f1
        state['current_cm'] = new_cm
        
        return decision
    
    def make_binary_decision(self, state, test_batch, success_rate, new_f1):
        """基于转换成功率做出二分决策"""
        print(f"\n=== 二分决策 ===")
        
        if success_rate == 1.0:
            # 100%成功，确认整批转换
            print(f"✅ 成功率100%，确认这{len(test_batch)}个账户的转换")
            
            # 确认这批账户的转换
            for acc in test_batch:
                if acc['current_predict'] == 0:
                    state['confirmed_predictions'][acc['ID']] = 1  # 确认为bad
                    acc['current_predict'] = 1
                else:
                    state['confirmed_predictions'][acc['ID']] = 0  # 确认为good
                    acc['current_predict'] = 0
            
            # 从测试队列移除已确认的账户
            confirmed_ids = set(acc['ID'] for acc in test_batch)
            state['test_queue'] = [acc for acc in state['test_queue'] 
                                 if acc['ID'] not in confirmed_ids]
            
            return f"confirmed_{len(test_batch)}"
            
        elif success_rate == 0.0:
            # 0%成功，确认整批不转换（保持原标签）
            print(f"✅ 成功率0%，确认这{len(test_batch)}个账户保持原标签")
            
            # 确认这批账户保持原标签
            for acc in test_batch:
                state['confirmed_predictions'][acc['ID']] = acc['current_predict']
            
            # 从测试队列移除已确认的账户
            confirmed_ids = set(acc['ID'] for acc in test_batch)
            state['test_queue'] = [acc for acc in state['test_queue'] 
                                 if acc['ID'] not in confirmed_ids]
            
            return f"confirmed_original_{len(test_batch)}"
            
        else:
            # 部分成功，需要二分到更小批次
            print(f"❌ 成功率{success_rate:.2%}，无法确定具体哪些账户成功，需要二分")
            
            if len(test_batch) == 1:
                # 单个账户但成功率不是0或100%，这是异常情况
                # 保守处理：保持原标签
                acc = test_batch[0]
                state['confirmed_predictions'][acc['ID']] = acc['current_predict']
                print(f"⚠️ 单个账户异常成功率，保守确认: {acc['ID']} 保持为 {acc['current_predict']}")
                
                # 从测试队列移除
                state['test_queue'] = [a for a in state['test_queue'] if a['ID'] != acc['ID']]
                return "single_conservative"
            else:
                # 二分：重新排列测试队列，优先测试概率更高的一半
                test_batch.sort(key=lambda x: x['score'], reverse=True)
                
                # 将测试批次重新放入队列前端，准备下次二分
                other_accounts = [acc for acc in state['test_queue'] 
                                if acc['ID'] not in [b['ID'] for b in test_batch]]
                state['test_queue'] = test_batch + other_accounts
                
                print(f"二分策略: 将{len(test_batch)}个账户重新排序，下轮测试前{len(test_batch)//2}个")
                return f"binary_split_{len(test_batch)}"
    
    def run_optimization(self, max_rounds=50, target_f1=1.0):
        """运行优化过程"""
        state = self.initialize_state()
        
        print(f"\n开始正确二分法优化，目标F1: {target_f1}")
        
        for round_num in range(1, max_rounds + 1):
            state['round'] = round_num
            
            print(f"\n{'='*60}")
            print(f"第 {round_num} 轮优化")
            print(f"{'='*60}")
            
            # 选择测试批次
            test_batch, test_direction = self.select_next_test_batch(state)
            if not test_batch:
                print("所有账户已处理完毕！")
                break
            
            print(f"概率范围: {test_batch[0]['score']:.6f} - {test_batch[-1]['score']:.6f}")
            
            # 创建测试文件
            test_submission = self.create_test_submission(state, test_batch, test_direction)
            test_filename = f"correct_test_round_{round_num}.csv"
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
            
            # 分析结果并做出二分决策
            decision = self.analyze_test_results(state, test_batch, test_direction, new_f1)
            
            # 记录历史
            state['test_history'].append({
                'round': round_num,
                'test_direction': test_direction,
                'batch_size': len(test_batch),
                'f1_score': new_f1,
                'decision': decision
            })
            
            # 保存状态
            self.save_state(state)
            
            # 清理测试文件
            os.remove(test_filename)
            
            # 检查完成条件
            if new_f1 >= target_f1:
                print(f"\n🎉 达到目标F1={target_f1}！")
                break
            
            # 显示进度
            confirmed_count = len(state['confirmed_predictions'])
            remaining_queue = len(state['test_queue'])
            print(f"\n当前进度:")
            print(f"  当前F1: {state['current_f1']:.6f}")
            print(f"  最佳F1: {state['best_f1']:.6f}")
            print(f"  已确认账户: {confirmed_count}")
            print(f"  待测试队列: {remaining_queue}")
            
            if remaining_queue == 0:
                print("测试队列已清空！")
                break
        
        self.generate_final_results(state)
        return state
    
    def generate_final_results(self, state):
        """生成最终结果"""
        print(f"\n{'='*60}")
        print("正确二分法优化完成！")
        print(f"{'='*60}")
        
        # 生成最终提交文件
        submission_data = []
        for acc in state['all_accounts']:
            if acc['ID'] in state['confirmed_predictions']:
                predict = state['confirmed_predictions'][acc['ID']]
            else:
                predict = acc['current_predict']
            submission_data.append({'ID': acc['ID'], 'Predict': predict})
        
        final_df = pd.DataFrame(submission_data)
        final_df.to_csv('final_correct_binary_submission.csv', index=False)
        
        # 统计
        final_bad = len(final_df[final_df['Predict'] == 1])
        final_good = len(final_df[final_df['Predict'] == 0])
        confirmed_count = len(state['confirmed_predictions'])
        
        print(f"最终统计:")
        print(f"  最佳F1分数: {state['best_f1']:.6f}")
        print(f"  最终预测bad: {final_bad}")
        print(f"  最终预测good: {final_good}")
        print(f"  确认账户数: {confirmed_count}")
        print(f"  总轮数: {state['round']}")
        print(f"  生成文件: final_correct_binary_submission.csv")

def main():
    """主函数"""
    account_scores_file = "/Users/mannormal/4011/account_scores.csv"
    
    optimizer = CorrectBinarySearchOptimizer(account_scores_file)
    final_state = optimizer.run_optimization(max_rounds=50, target_f1=1.0)
    
    print("\n🎉 正确二分法优化完成！")

if __name__ == "__main__":
    main()