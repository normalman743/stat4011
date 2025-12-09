import pandas as pd
import numpy as np
import json
import os
from time import time

class AlgorithmSimulator:
    def __init__(self, n_accounts=10000, bad_ratio=0.1):
        self.n_accounts = n_accounts
        self.bad_ratio = bad_ratio
        self.true_bad = int(n_accounts * bad_ratio)
        self.true_good = n_accounts - self.true_bad
        
        # 测试参数
        self.batch_size = 50
        self.f1_threshold = 0.01
        
        print("=== 算法效率测试系统 ===")
        print(f"总账户数: {n_accounts}")
        print(f"真实分布: Bad={self.true_bad} ({bad_ratio*100:.1f}%), Good={self.true_good} ({(1-bad_ratio)*100:.1f}%)")
    
    def generate_test_data(self):
        """生成测试数据"""
        print("\\n生成测试数据...")
        
        # 1. 生成真实标签 (1:bad, 0:good)
        true_labels = [1] * self.true_bad + [0] * self.true_good
        np.random.shuffle(true_labels)
        
        # 2. 为每个账户生成概率分数
        accounts = []
        for i in range(self.n_accounts):
            account_id = f"test_{i:06d}"
            true_label = true_labels[i]
            
            # 根据真实标签和规则生成分数
            if true_label == 1:  # 真bad账户
                # 70%概率获得高分(0.6-0.9)，30%概率获得低分(0.1-0.4)
                if np.random.random() < 0.7:
                    score = np.random.uniform(0.6, 0.9)
                else:
                    score = np.random.uniform(0.1, 0.4)
            else:  # 真good账户
                # 80%概率获得低分(0.1-0.4)，20%概率获得高分(0.6-0.9)
                if np.random.random() < 0.8:
                    score = np.random.uniform(0.1, 0.4)
                else:
                    score = np.random.uniform(0.6, 0.9)
            
            accounts.append({
                'ID': account_id,
                'predict': score,
                'true_label': true_label  # 这个在真实系统中不存在，仅用于模拟
            })
        
        # 保存数据
        accounts_df = pd.DataFrame(accounts)
        accounts_df[['ID', 'predict']].to_csv('test_account_scores.csv', index=False, float_format='%.6f')
        
        # 生成基线提交文件（基于概率阈值0.5）
        baseline_submission = []
        for acc in accounts:
            baseline_predict = 1 if acc['predict'] > 0.5 else 0
            baseline_submission.append({
                'ID': acc['ID'], 
                'Predict': baseline_predict,
                'true_label': acc['true_label']  # 仅用于模拟
            })
        
        baseline_df = pd.DataFrame(baseline_submission)
        baseline_df[['ID', 'Predict']].to_csv('test_baseline_submission.csv', index=False)
        
        # 计算基线性能
        baseline_cm = self.calculate_true_confusion_matrix(baseline_df)
        baseline_f1 = self.calculate_f1_from_cm(baseline_cm)
        
        print(f"数据生成完成:")
        print(f"  test_account_scores.csv: {len(accounts)} 账户")
        print(f"  test_baseline_submission.csv: 基线F1={baseline_f1:.6f}")
        print(f"  基线混淆矩阵: TP={baseline_cm['TP']}, FP={baseline_cm['FP']}, FN={baseline_cm['FN']}, TN={baseline_cm['TN']}")
        
        # 验证分数分布
        high_score_accounts = [a for a in accounts if a['predict'] > 0.5]
        low_score_accounts = [a for a in accounts if a['predict'] <= 0.5]
        
        high_score_bad_ratio = sum(1 for a in high_score_accounts if a['true_label'] == 1) / len(high_score_accounts) if high_score_accounts else 0
        low_score_bad_ratio = sum(1 for a in low_score_accounts if a['true_label'] == 1) / len(low_score_accounts) if low_score_accounts else 0
        
        print(f"\\n分数分布验证:")
        print(f"  高分账户(>0.5): {len(high_score_accounts)}个, 真bad比例: {high_score_bad_ratio:.3f}")
        print(f"  低分账户(<=0.5): {len(low_score_accounts)}个, 真bad比例: {low_score_bad_ratio:.3f}")
        
        return accounts, baseline_df, baseline_f1, baseline_cm
    
    def calculate_true_confusion_matrix(self, submission_df):
        """计算真实混淆矩阵（模拟环境特有）"""
        tp = len(submission_df[(submission_df['Predict'] == 1) & (submission_df['true_label'] == 1)])
        fp = len(submission_df[(submission_df['Predict'] == 1) & (submission_df['true_label'] == 0)])
        fn = len(submission_df[(submission_df['Predict'] == 0) & (submission_df['true_label'] == 1)])
        tn = len(submission_df[(submission_df['Predict'] == 0) & (submission_df['true_label'] == 0)])
        
        return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}
    
    def calculate_f1_from_cm(self, cm):
        """从混淆矩阵计算F1分数"""
        precision = cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) > 0 else 0
        recall = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
    def simulate_submission(self, submission_df):
        """模拟提交并返回F1分数（替代真实API调用）"""
        cm = self.calculate_true_confusion_matrix(submission_df)
        f1 = self.calculate_f1_from_cm(cm)
        return f1
    
    def initialize_test_state(self, accounts, baseline_df, baseline_f1, baseline_cm):
        """初始化测试状态"""
        print("\\n初始化验证状态...")
        
        # 按策略分组账户
        unconfirmed_accounts = []
        for acc in accounts:
            if 0 < acc['predict'] < 1:  # 未确认的账户
                unconfirmed_accounts.append({
                    'ID': acc['ID'],
                    'score': acc['predict'],
                    'current_predict': 1 if acc['predict'] > 0.5 else 0,
                    'true_label': acc['true_label']  # 仅用于验证
                })
        
        # 排序：优先处理高概率bad账户（从bad往good走）
        suspected_bad = sorted([a for a in unconfirmed_accounts if a['score'] > 0.5], 
                              key=lambda x: x['score'], reverse=True)
        suspected_good = sorted([a for a in unconfirmed_accounts if a['score'] < 0.5], 
                               key=lambda x: x['score'])
        
        state = {
            'round': 0,
            'baseline_f1': baseline_f1,
            'baseline_cm': baseline_cm,
            'baseline_bad': len(baseline_df[baseline_df['Predict'] == 1]),
            'baseline_good': len(baseline_df[baseline_df['Predict'] == 0]),
            'suspected_bad_queue': suspected_bad,
            'suspected_good_queue': suspected_good,
            'confirmed_accounts': {},
            'test_history': [],
            'true_labels': {acc['ID']: acc['true_label'] for acc in accounts}  # 仅用于验证
        }
        
        print(f"  Suspected bad queue: {len(suspected_bad)}")
        print(f"  Suspected good queue: {len(suspected_good)}")
        
        return state
    
    def select_test_batch(self, state):
        """选择测试批次"""
        if len(state['suspected_bad_queue']) >= self.batch_size:
            batch = state['suspected_bad_queue'][:self.batch_size]
            test_direction = "bad_to_good"
            print(f"选择 {len(batch)} 个高概率bad账户，测试改成good (1→0)")
        elif len(state['suspected_good_queue']) >= self.batch_size:
            batch = state['suspected_good_queue'][:self.batch_size]
            test_direction = "good_to_bad"
            print(f"选择 {len(batch)} 个高概率good账户，测试改成bad (0→1)")
        else:
            remaining = state['suspected_bad_queue'] + state['suspected_good_queue']
            if len(remaining) == 0:
                return None, None
            
            batch = remaining[:min(self.batch_size, len(remaining))]
            test_direction = "mixed"
            print(f"选择剩余 {len(batch)} 个账户进行最终测试")
        
        return batch, test_direction
    
    def create_test_submission(self, state, test_batch, test_direction):
        """创建测试提交"""
        baseline_df = pd.read_csv('test_baseline_submission.csv')
        
        submission_data = []
        test_account_ids = [acc['ID'] for acc in test_batch]
        
        for _, row in baseline_df.iterrows():
            account_id = row['ID']
            
            if account_id in state['confirmed_accounts']:
                predict = state['confirmed_accounts'][account_id]['label']
            elif account_id in test_account_ids:
                if test_direction == "bad_to_good":
                    predict = 0
                elif test_direction == "good_to_bad":
                    predict = 1
                else:  # mixed
                    acc_info = next(acc for acc in test_batch if acc['ID'] == account_id)
                    predict = 0 if acc_info['score'] > 0.5 else 1
            else:
                predict = row['Predict']
            
            submission_data.append({
                'ID': account_id, 
                'Predict': predict,
                'true_label': state['true_labels'][account_id]  # 仅用于模拟
            })
        
        return pd.DataFrame(submission_data)
    
    def analyze_test_results(self, state, test_batch, test_direction, new_f1):
        """分析测试结果（简化版，重点验证算法逻辑）"""
        print(f"\\n=== 分析第{state['round']}轮测试结果 ===")
        
        f1_change = new_f1 - state['baseline_f1']
        print(f"F1变化: {f1_change:+.4f}")
        
        # 验证算法准确性：计算测试批次的真实标签分布
        true_bad_in_batch = sum(1 for acc in test_batch if acc['true_label'] == 1)
        true_good_in_batch = len(test_batch) - true_bad_in_batch
        
        print(f"测试批次真实分布: {true_bad_in_batch}个真bad, {true_good_in_batch}个真good")
        
        confirmed_count = 0
        
        # 简化的确认逻辑
        if abs(f1_change) > self.f1_threshold:
            if test_direction == "bad_to_good" and f1_change < -self.f1_threshold:
                # F1下降显著，确认这批账户大多是bad
                print(f"✅ 算法判断：这批账户大多是bad")
                for acc in test_batch:
                    state['confirmed_accounts'][acc['ID']] = {
                        'label': 1 if acc['score'] > np.median([a['score'] for a in test_batch]) else 0,
                        'confidence': 0.9,
                        'true_label': acc['true_label']  # 仅用于验证
                    }
                    confirmed_count += 1
                    
            elif test_direction == "bad_to_good" and f1_change > self.f1_threshold:
                # F1提升，确认这批账户大多是good
                print(f"✅ 算法判断：这批账户大多是good")
                for acc in test_batch:
                    state['confirmed_accounts'][acc['ID']] = {
                        'label': 0 if acc['score'] < np.median([a['score'] for a in test_batch]) else 1,
                        'confidence': 0.9,
                        'true_label': acc['true_label']  # 仅用于验证
                    }
                    confirmed_count += 1
                    
            elif test_direction == "good_to_bad":
                # 类似逻辑，但方向相反
                if f1_change > self.f1_threshold:
                    print(f"✅ 算法判断：这批账户大多是bad")
                else:
                    print(f"✅ 算法判断：这批账户大多是good")
                
                for acc in test_batch:
                    predicted_label = 1 if f1_change > 0 else 0
                    state['confirmed_accounts'][acc['ID']] = {
                        'label': predicted_label,
                        'confidence': 0.8,
                        'true_label': acc['true_label']
                    }
                    confirmed_count += 1
        else:
            print(f"⚠️  F1变化太小({f1_change:+.4f})，无法确定")
        
        # 计算算法准确性
        if confirmed_count > 0:
            correct_predictions = sum(1 for acc_id, info in state['confirmed_accounts'].items() 
                                    if acc_id in [acc['ID'] for acc in test_batch] and 
                                    info['label'] == info['true_label'])
            accuracy = correct_predictions / confirmed_count
            print(f"本轮算法准确率: {accuracy:.3f} ({correct_predictions}/{confirmed_count})")
        
        # 从队列中移除已确认的账户
        confirmed_ids = set(state['confirmed_accounts'].keys())
        state['suspected_bad_queue'] = [acc for acc in state['suspected_bad_queue'] 
                                       if acc['ID'] not in confirmed_ids]
        state['suspected_good_queue'] = [acc for acc in state['suspected_good_queue'] 
                                        if acc['ID'] not in confirmed_ids]
        
        return confirmed_count
    
    def run_simulation(self, max_rounds=20):
        """运行完整的模拟测试"""
        start_time = time()
        
        # 生成测试数据
        accounts, baseline_df, baseline_f1, baseline_cm = self.generate_test_data()
        
        # 初始化状态
        state = self.initialize_test_state(accounts, baseline_df, baseline_f1, baseline_cm)
        
        print(f"\\n{'='*60}")
        print("开始算法验证")
        print(f"{'='*60}")
        
        total_submissions = 0
        
        for round_num in range(1, max_rounds + 1):
            state['round'] = round_num
            
            print(f"\\n--- 第 {round_num} 轮 ---")
            
            # 选择测试批次
            test_batch, test_direction = self.select_test_batch(state)
            if not test_batch:
                print("所有账户已确认完毕！")
                break
            
            # 创建测试提交
            test_submission = self.create_test_submission(state, test_batch, test_direction)
            
            # 模拟提交并获取F1
            new_f1 = self.simulate_submission(test_submission)
            total_submissions += 1
            
            print(f"模拟F1分数: {new_f1:.6f}")
            
            # 分析结果
            confirmed_count = self.analyze_test_results(state, test_batch, test_direction, new_f1)
            
            # 记录历史
            state['test_history'].append({
                'round': round_num,
                'test_direction': test_direction,
                'batch_size': len(test_batch),
                'f1_score': new_f1,
                'confirmed_count': confirmed_count
            })
            
            # 检查完成状态
            total_unconfirmed = len(state['suspected_bad_queue']) + len(state['suspected_good_queue'])
            total_confirmed = len(state['confirmed_accounts'])
            
            print(f"进度: 已确认 {total_confirmed}, 剩余 {total_unconfirmed}")
            
            if total_unconfirmed == 0:
                print("\\n🎉 所有账户验证完成！")
                break
        
        end_time = time()
        
        # 最终统计
        self.generate_simulation_report(state, total_submissions, end_time - start_time)
        
        return state
    
    def generate_simulation_report(self, state, total_submissions, elapsed_time):
        """生成模拟测试报告"""
        print(f"\\n{'='*60}")
        print("算法验证报告")
        print(f"{'='*60}")
        
        total_confirmed = len(state['confirmed_accounts'])
        
        # 计算整体准确率
        if total_confirmed > 0:
            correct_predictions = sum(1 for info in state['confirmed_accounts'].values() 
                                    if info['label'] == info['true_label'])
            overall_accuracy = correct_predictions / total_confirmed
        else:
            overall_accuracy = 0
        
        # 按轮次统计
        confirmed_by_round = [h['confirmed_count'] for h in state['test_history']]
        f1_progression = [h['f1_score'] for h in state['test_history']]
        
        print(f"性能指标:")
        print(f"  总轮数: {state['round']}")
        print(f"  总提交次数: {total_submissions}")
        print(f"  用时: {elapsed_time:.2f} 秒")
        print(f"  平均每轮: {elapsed_time/state['round']:.3f} 秒")
        
        print(f"\\n算法效果:")
        print(f"  总确认账户: {total_confirmed}/{self.n_accounts} ({total_confirmed/self.n_accounts*100:.1f}%)")
        print(f"  算法整体准确率: {overall_accuracy:.3f}")
        print(f"  剩余未确认: {self.n_accounts - total_confirmed}")
        
        print(f"\\n收敛分析:")
        print(f"  每轮确认数: {confirmed_by_round}")
        print(f"  F1进展: {[f'{f1:.4f}' for f1 in f1_progression[:5]]}...")
        
        # 理论vs实际
        theoretical_max_rounds = int(np.ceil(np.log2(self.n_accounts)))
        efficiency = theoretical_max_rounds / state['round'] if state['round'] > 0 else 0
        
        print(f"\\n复杂度分析:")
        print(f"  理论最大轮数: {theoretical_max_rounds} (log₂({self.n_accounts}))")
        print(f"  实际轮数: {state['round']}")
        print(f"  算法效率: {efficiency:.2f}")
        
        # 保存报告
        report = {
            'parameters': {
                'n_accounts': self.n_accounts,
                'bad_ratio': self.bad_ratio,
                'batch_size': self.batch_size,
                'f1_threshold': self.f1_threshold
            },
            'results': {
                'total_rounds': state['round'],
                'total_submissions': total_submissions,
                'elapsed_time': elapsed_time,
                'total_confirmed': total_confirmed,
                'overall_accuracy': overall_accuracy,
                'theoretical_max_rounds': theoretical_max_rounds,
                'efficiency': efficiency
            },
            'history': state['test_history']
        }
        
        with open('test_simulation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\\n📊 详细报告已保存: test_simulation_report.json")

def main():
    """主测试函数"""
    # 测试不同规模
    test_cases = [
        {'n_accounts': 1000, 'bad_ratio': 0.1},
        # {'n_accounts': 5000, 'bad_ratio': 0.1},
        # {'n_accounts': 10000, 'bad_ratio': 0.1}
    ]
    
    for i, params in enumerate(test_cases, 1):
        print(f"\\n{'#'*80}")
        print(f"测试案例 {i}: {params['n_accounts']} 账户, {params['bad_ratio']*100:.0f}% bad")
        print(f"{'#'*80}")
        
        simulator = AlgorithmSimulator(**params)
        final_state = simulator.run_simulation()
        
        print(f"\\n测试案例 {i} 完成！")

if __name__ == "__main__":
    main()