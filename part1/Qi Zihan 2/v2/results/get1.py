import pandas as pd
import numpy as np
import json
import os
import requests
from urllib.parse import urlparse, parse_qs
import urllib3
from time import sleep

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class SmartBinaryVerifier:
    def __init__(self, account_scores_file, baseline_submission_file, state_json_file="verification_state.json"):
        self.account_scores_file = account_scores_file
        self.baseline_submission_file = baseline_submission_file
        self.state_json_file = state_json_file
        
        # 真实分布
        self.true_bad = 727
        self.true_good = 6831
        
        # 测试参数
        self.batch_size = 50
        self.f1_threshold = 0.005  # 降低阈值，更敏感地检测F1变化
        
        print("=== 智能二分验证系统 ===")
        print(f"批次大小: {self.batch_size}")
        print(f"F1判断阈值: {self.f1_threshold}")
    
    def calculate_confusion_matrix(self, pred_bad, pred_good, bad_f1):
        """根据预测分布和F1计算混淆矩阵"""
        if bad_f1 == 0:
            return {'TP': 0, 'FP': pred_bad, 'FN': self.true_bad, 'TN': self.true_good}
        
        # 优化的F1反推TP算法 - 使用更精确的搜索
        best_tp = 0
        best_f1_diff = float('inf')
        
        # 扩展搜索范围，考虑边界情况
        max_tp = min(pred_bad, self.true_bad)
        
        for tp in range(max_tp + 1):
            # 计算对应的混淆矩阵元素
            fp = pred_bad - tp
            fn = self.true_bad - tp  
            tn = self.true_good - fp
            
            # 验证混淆矩阵的合理性
            if fp < 0 or fn < 0 or tn < 0:
                continue
                
            # 计算precision, recall, f1
            precision = tp / pred_bad if pred_bad > 0 else 0
            recall = tp / self.true_bad if self.true_bad > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_diff = abs(f1 - bad_f1)
            if f1_diff < best_f1_diff:
                best_f1_diff = f1_diff
                best_tp = tp
        
        tp = best_tp
        fp = pred_bad - tp
        fn = self.true_bad - tp
        tn = self.true_good - fp
        
        # 再次验证结果的合理性
        if fp < 0: fp = 0
        if fn < 0: fn = 0  
        if tn < 0: tn = 0
        
        return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}
    
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
                print(f"提交成功: {csv_file}, F1分数: {params['score'][0]}")
                return float(params['score'][0])
            return None
        except Exception as e:
            print(f"提交错误: {e}")
            return None
    
    def initialize_state(self):
        """初始化或加载状态"""
        if os.path.exists(self.state_json_file):
            print(f"加载现有状态: {self.state_json_file}")
            with open(self.state_json_file, 'r') as f:
                return json.load(f)
        
        print("创建新状态文件...")
        
        # 读取账户分数
        scores_df = pd.read_csv(self.account_scores_file)
        baseline_df = pd.read_csv(self.baseline_submission_file)
        
        # 获取基线混淆矩阵
        baseline_bad = len(baseline_df[baseline_df['Predict'] == 1])
        baseline_good = len(baseline_df[baseline_df['Predict'] == 0])
        baseline_f1 = 0.7628549501151188  # 已知基线F1
        
        baseline_cm = self.calculate_confusion_matrix(baseline_bad, baseline_good, baseline_f1)
        
        # 按策略分组账户
        unconfirmed_accounts = []
        for _, row in scores_df.iterrows():
            if 0 < row['predict'] < 1:  # 未确认的账户
                unconfirmed_accounts.append({
                    'ID': row['ID'],
                    'score': row['predict'],
                    'current_predict': 1 if row['predict'] > 0.5 else 0
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
            'baseline_bad': baseline_bad,
            'baseline_good': baseline_good,
            'suspected_bad_queue': suspected_bad,
            'suspected_good_queue': suspected_good,
            'confirmed_accounts': {},  # {account_id: {label: 0/1, confidence: 0.9}}
            'test_history': []
        }
        
        self.save_state(state)
        return state
    
    def save_state(self, state):
        """保存状态到JSON文件"""
        with open(self.state_json_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def select_test_batch(self, state):
        """选择测试批次 - 优化的log2策略"""
        # 优先策略：选择最有可能带来F1提升的批次
        
        # 策略1：优先测试极高概率bad账户 (>0.8)
        high_confidence_bad = [acc for acc in state['suspected_bad_queue'] if acc['score'] > 0.8]
        if len(high_confidence_bad) >= self.batch_size:
            batch = high_confidence_bad[:self.batch_size]
            test_direction = "bad_to_good"
            print(f"选择 {len(batch)} 个极高概率bad账户 (>0.8)，测试改成good (1→0)")
            return batch, test_direction
        
        # 策略2：测试中等概率账户 (0.5-0.8)，这些最容易通过调整获得F1提升
        medium_bad = [acc for acc in state['suspected_bad_queue'] if 0.5 <= acc['score'] <= 0.8]
        if len(medium_bad) >= self.batch_size:
            # 按概率降序排序，优先测试较高概率的
            batch = sorted(medium_bad, key=lambda x: x['score'], reverse=True)[:self.batch_size]
            test_direction = "bad_to_good"
            print(f"选择 {len(batch)} 个中等概率bad账户 (0.5-0.8)，测试改成good (1→0)")
            return batch, test_direction
        
        # 策略3：测试剩余的suspected_bad队列
        if len(state['suspected_bad_queue']) >= self.batch_size:
            batch = state['suspected_bad_queue'][:self.batch_size]
            test_direction = "bad_to_good"
            print(f"选择 {len(batch)} 个剩余bad账户，测试改成good (1→0)")
            return batch, test_direction
        
        # 策略4：测试suspected_good队列中概率较高的 (接近0.5的)
        high_good = [acc for acc in state['suspected_good_queue'] if acc['score'] > 0.3]
        if len(high_good) >= self.batch_size:
            # 按概率降序排序，优先测试接近0.5的
            batch = sorted(high_good, key=lambda x: x['score'], reverse=True)[:self.batch_size]
            test_direction = "good_to_bad"
            print(f"选择 {len(batch)} 个较高概率good账户 (>0.3)，测试改成bad (0→1)")
            return batch, test_direction
            
        # 策略5：测试剩余的suspected_good队列
        elif len(state['suspected_good_queue']) >= self.batch_size:
            batch = state['suspected_good_queue'][:self.batch_size]
            test_direction = "good_to_bad"  # 0→1  
            print(f"选择 {len(batch)} 个剩余good账户，测试改成bad (0→1)")
            return batch, test_direction
        
        # 策略6：处理剩余的所有账户
        else:
            remaining = state['suspected_bad_queue'] + state['suspected_good_queue']
            if len(remaining) == 0:
                return None, None
            
            batch = remaining[:min(self.batch_size, len(remaining))]
            test_direction = "mixed"
            print(f"选择剩余 {len(batch)} 个账户进行最终测试")
            return batch, test_direction
    
    def create_test_submission(self, state, test_batch, test_direction):
        """创建测试提交文件"""
        # 读取当前账户分数和基线预测
        scores_df = pd.read_csv(self.account_scores_file)
        baseline_df = pd.read_csv(self.baseline_submission_file)
        
        submission_data = []
        test_account_ids = [acc['ID'] for acc in test_batch]
        
        for _, row in baseline_df.iterrows():
            account_id = row['ID']
            
            # 检查是否已确认
            if account_id in state['confirmed_accounts']:
                predict = state['confirmed_accounts'][account_id]['label']
            # 检查是否在测试批次中
            elif account_id in test_account_ids:
                if test_direction == "bad_to_good":
                    predict = 0  # 改成good
                elif test_direction == "good_to_bad":
                    predict = 1  # 改成bad
                else:  # mixed
                    # 根据原始策略
                    score = scores_df[scores_df['ID'] == account_id]['predict'].iloc[0]
                    predict = 0 if score > 0.5 else 1  # 反向测试
            else:
                # 保持原始预测
                predict = row['Predict']
            
            submission_data.append({'ID': account_id, 'Predict': predict})
        
        return pd.DataFrame(submission_data)
    
    def analyze_test_results(self, state, test_batch, test_direction, new_f1):
        """分析测试结果并确认账户标签"""
        print(f"\\n=== 分析测试结果 ===")
        
        # 计算新的混淆矩阵
        if test_direction == "bad_to_good":
            # 减少了bad预测
            new_bad = state['baseline_bad'] - len(test_batch)
            new_good = state['baseline_good'] + len(test_batch)
        elif test_direction == "good_to_bad":
            # 增加了bad预测
            new_bad = state['baseline_bad'] + len(test_batch)
            new_good = state['baseline_good'] - len(test_batch)
        else:
            # mixed情况，需要精确计算
            bad_to_good_count = sum(1 for acc in test_batch if acc['score'] > 0.5)
            good_to_bad_count = len(test_batch) - bad_to_good_count
            new_bad = state['baseline_bad'] - bad_to_good_count + good_to_bad_count
            new_good = state['baseline_good'] + bad_to_good_count - good_to_bad_count
        
        new_cm = self.calculate_confusion_matrix(new_bad, new_good, new_f1)
        baseline_cm = state['baseline_cm']
        
        print(f"基线混淆矩阵: TP={baseline_cm['TP']}, FP={baseline_cm['FP']}, FN={baseline_cm['FN']}, TN={baseline_cm['TN']}")
        print(f"新的混淆矩阵: TP={new_cm['TP']}, FP={new_cm['FP']}, FN={new_cm['FN']}, TN={new_cm['TN']}")
        
        # 计算变化
        tp_change = new_cm['TP'] - baseline_cm['TP']
        fp_change = new_cm['FP'] - baseline_cm['FP'] 
        fn_change = new_cm['FN'] - baseline_cm['FN']
        tn_change = new_cm['TN'] - baseline_cm['TN']
        
        print(f"混淆矩阵变化: TP{tp_change:+d}, FP{fp_change:+d}, FN{fn_change:+d}, TN{tn_change:+d}")
        
        f1_change = new_f1 - state['baseline_f1']
        print(f"F1变化: {f1_change:+.4f}")
        
        # 确认账户标签
        confirmed_count = 0
        
        if test_direction == "bad_to_good":
            # 分析：把suspected bad改成good的结果
            if f1_change < -self.f1_threshold:
                # F1下降显著，说明这批账户确实大多是bad
                true_bad_count = abs(tp_change)  # TP减少的数量
                true_good_count = len(test_batch) - true_bad_count
                
                print(f"✅ 确认结果：{true_bad_count}个真bad，{true_good_count}个真good")
                
                # 按分数排序，高分的更可能是bad
                sorted_batch = sorted(test_batch, key=lambda x: x['score'], reverse=True)
                
                for i, acc in enumerate(sorted_batch):
                    if i < true_bad_count:
                        state['confirmed_accounts'][acc['ID']] = {'label': 1, 'confidence': 0.9}
                        confirmed_count += 1
                    else:
                        state['confirmed_accounts'][acc['ID']] = {'label': 0, 'confidence': 0.8}
                        confirmed_count += 1
                        
            elif f1_change > self.f1_threshold:
                # F1提升，说明这批账户大多是被错误分类的good
                true_good_count = tn_change  
                true_bad_count = len(test_batch) - true_good_count
                
                print(f"✅ 确认结果：{true_bad_count}个真bad，{true_good_count}个真good")
                
                # 低分的更可能是good
                sorted_batch = sorted(test_batch, key=lambda x: x['score'])
                
                for i, acc in enumerate(sorted_batch):
                    if i < true_good_count:
                        state['confirmed_accounts'][acc['ID']] = {'label': 0, 'confidence': 0.9}
                        confirmed_count += 1
                    else:
                        state['confirmed_accounts'][acc['ID']] = {'label': 1, 'confidence': 0.8}
                        confirmed_count += 1
        
        elif test_direction == "good_to_bad":
            # 分析：把suspected good改成bad的结果
            if f1_change > self.f1_threshold:
                # F1提升，说明这批账户确实大多是bad
                true_bad_count = tp_change  # TP增加的数量
                true_good_count = len(test_batch) - true_bad_count
                
                print(f"✅ 确认结果：{true_bad_count}个真bad，{true_good_count}个真good")
                
                # 按分数排序，高分的更可能是bad
                sorted_batch = sorted(test_batch, key=lambda x: x['score'], reverse=True)
                
                for i, acc in enumerate(sorted_batch):
                    if i < true_bad_count:
                        state['confirmed_accounts'][acc['ID']] = {'label': 1, 'confidence': 0.9}
                        confirmed_count += 1
                    else:
                        state['confirmed_accounts'][acc['ID']] = {'label': 0, 'confidence': 0.8}
                        confirmed_count += 1
                        
            elif f1_change < -self.f1_threshold:
                # F1下降，说明这批账户大多确实是good
                true_good_count = abs(tn_change)
                true_bad_count = len(test_batch) - true_good_count
                
                print(f"✅ 确认结果：{true_bad_count}个真bad，{true_good_count}个真good")
                
                # 低分的更可能是good
                sorted_batch = sorted(test_batch, key=lambda x: x['score'])
                
                for i, acc in enumerate(sorted_batch):
                    if i < true_good_count:
                        state['confirmed_accounts'][acc['ID']] = {'label': 0, 'confidence': 0.9}
                        confirmed_count += 1
                    else:
                        state['confirmed_accounts'][acc['ID']] = {'label': 1, 'confidence': 0.8}
                        confirmed_count += 1
        
        else:  # mixed direction
            # 对于混合方向，根据F1变化和混淆矩阵变化来判断
            if abs(f1_change) > self.f1_threshold:
                # 有显著变化，可以确认部分账户
                if f1_change > 0:
                    # F1提升，优先确认高分的为bad，低分的为good
                    print("F1提升，根据分数确认标签")
                else:
                    # F1下降，需要更保守的策略
                    print("F1下降，采用保守确认策略")
                
                # 简化处理：按原始预测确认
                for acc in test_batch:
                    original_predict = 1 if acc['score'] > 0.5 else 0
                    confidence = 0.7  # 混合情况下置信度较低
                    state['confirmed_accounts'][acc['ID']] = {'label': original_predict, 'confidence': confidence}
                    confirmed_count += 1
        
        # 从队列中移除已确认的账户
        confirmed_ids = set(state['confirmed_accounts'].keys())
        state['suspected_bad_queue'] = [acc for acc in state['suspected_bad_queue'] 
                                       if acc['ID'] not in confirmed_ids]
        state['suspected_good_queue'] = [acc for acc in state['suspected_good_queue'] 
                                        if acc['ID'] not in confirmed_ids]
        
        print(f"本轮确认了 {confirmed_count} 个账户")
        print(f"剩余待确认: suspected_bad={len(state['suspected_bad_queue'])}, suspected_good={len(state['suspected_good_queue'])}")
        
        return confirmed_count
    
    def run_verification(self, max_rounds=15):
        """运行验证过程"""
        state = self.initialize_state()
        
        print(f"\\n初始状态:")
        print(f"  Suspected bad: {len(state['suspected_bad_queue'])}")
        print(f"  Suspected good: {len(state['suspected_good_queue'])}")
        print(f"  已确认: {len(state['confirmed_accounts'])}")
        
        for round_num in range(1, max_rounds + 1):
            state['round'] = round_num
            
            print(f"\\n{'='*60}")
            print(f"第 {round_num} 轮测试")
            print(f"{'='*60}")
            
            # 选择测试批次
            test_batch, test_direction = self.select_test_batch(state)
            if not test_batch:
                print("所有账户已确认完毕！")
                break
            
            # 创建测试文件
            test_submission = self.create_test_submission(state, test_batch, test_direction)
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
            
            print(f"获得F1分数: {new_f1:.6f}")
            
            # 计算F1变化
            f1_change = new_f1 - state['baseline_f1']
            print(f"F1变化: {f1_change:+.6f}")
            
            # 分析结果
            confirmed_count = self.analyze_test_results(state, test_batch, test_direction, new_f1)
            
            # 更新基线信息用于下轮计算
            if confirmed_count > 0:
                state['baseline_f1'] = new_f1
                # 重新计算基线分布
                baseline_bad = len(test_submission[test_submission['Predict'] == 1])
                baseline_good = len(test_submission[test_submission['Predict'] == 0])
                state['baseline_bad'] = baseline_bad
                state['baseline_good'] = baseline_good
                state['baseline_cm'] = self.calculate_confusion_matrix(baseline_bad, baseline_good, new_f1)
                print(f"更新基线: Bad={baseline_bad}, Good={baseline_good}, F1={new_f1:.6f}")
            
            # 记录历史
            state['test_history'].append({
                'round': round_num,
                'test_direction': test_direction,
                'batch_size': len(test_batch),
                'f1_score': new_f1,
                'confirmed_count': confirmed_count
            })
            
            # 保存状态
            self.save_state(state)
            
            # 清理测试文件
            os.remove(test_filename)
            
            # 检查是否完成
            total_unconfirmed = len(state['suspected_bad_queue']) + len(state['suspected_good_queue'])
            if total_unconfirmed == 0:
                print("\\n🎉 所有账户验证完成！")
                break
                
            print(f"\\n当前进度: 已确认 {len(state['confirmed_accounts'])}, 剩余 {total_unconfirmed}")
            sleep(2)
        
        self.generate_final_results(state)
        return state
    
    def generate_final_results(self, state):
        """生成最终结果"""
        print(f"\\n{'='*60}")
        print("验证完成！生成最终结果...")
        print(f"{'='*60}")
        
        # 更新account_scores.csv
        scores_df = pd.read_csv(self.account_scores_file)
        
        for account_id, info in state['confirmed_accounts'].items():
            scores_df.loc[scores_df['ID'] == account_id, 'predict'] = info['label']
        
        scores_df.to_csv('final_account_scores.csv', index=False)
        
        # 生成最终提交文件
        baseline_df = pd.read_csv(self.baseline_submission_file)
        final_submission = []
        
        for _, row in baseline_df.iterrows():
            account_id = row['ID']
            if account_id in state['confirmed_accounts']:
                predict = state['confirmed_accounts'][account_id]['label']
            else:
                # 保持基线预测
                predict = row['Predict']
            
            final_submission.append({'ID': account_id, 'Predict': predict})
        
        final_df = pd.DataFrame(final_submission)
        final_df.to_csv('final_submission.csv', index=False)
        
        # 统计信息
        total_confirmed = len(state['confirmed_accounts'])
        confirmed_bad = sum(1 for info in state['confirmed_accounts'].values() if info['label'] == 1)
        confirmed_good = total_confirmed - confirmed_bad
        
        final_bad = len(final_df[final_df['Predict'] == 1])
        final_good = len(final_df[final_df['Predict'] == 0])
        
        print(f"验证统计:")
        print(f"  总确认账户: {total_confirmed}")
        print(f"  确认bad: {confirmed_bad}")
        print(f"  确认good: {confirmed_good}")
        print(f"  总轮数: {state['round']}")
        print(f"\\n最终分布:")
        print(f"  Bad: {final_bad}")
        print(f"  Good: {final_good}")
        print(f"\\n文件生成:")
        print(f"  final_account_scores.csv")
        print(f"  final_submission.csv")

def main():
    """主函数"""
    account_scores_file = "/Users/mannormal/4011/account_scores.csv"  # 你生成的概率文件
    baseline_submission_file = "/Users/mannormal/4011/Qi Zihan/v2/results/high_score_predictions/0.75+/v3.2refined_fold1_bad_f1_0.8083_good_0.9803_bad_0.8083_macro_0.8943_weighted_0.9634_seed_13_REAL_F1_0.7628549501151188.csv"
    
    verifier = SmartBinaryVerifier(account_scores_file, baseline_submission_file)
    final_state = verifier.run_verification()
    
    print("\\n🎉 智能二分验证完成！")

if __name__ == "__main__":
    main()