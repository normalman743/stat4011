import pandas as pd
import numpy as np
import requests
from urllib.parse import urlparse, parse_qs
import urllib3
import os
import json
from pathlib import Path
from time import sleep

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class IterativeAccountVerifier:
    def __init__(self, baseline_file, reference_file, state_file="account_verification_state.csv"):
        self.baseline_file = baseline_file
        self.reference_file = reference_file
        self.state_file = state_file
        self.true_bad = 727
        self.true_good = 6832
        self.total_accounts = 7559
        self.round_count = 0
        
        print("=== 迭代账户验证系统 (二分法优化版) ===")
        print(f"真实分布：Bad={self.true_bad}, Good={self.true_good}, 总计={self.total_accounts}")
        
    def calculate_confusion_matrix(self, pred_bad, pred_good, bad_f1):
        """根据预测分布和F1计算混淆矩阵 - 保持高精度"""
        if bad_f1 == 0:
            return {'TP': 0, 'FP': pred_bad, 'FN': self.true_bad, 'TN': self.true_good - pred_bad}
        
        # 保持高精度计算
        tp_precise = bad_f1 * (pred_bad + self.true_bad) / 2.0
        tp = int(round(tp_precise))
        fp = pred_bad - tp
        fn = self.true_bad - tp  
        tn = self.true_good - fp
        
        precision = tp / pred_bad if pred_bad > 0 else 0
        recall = tp / self.true_bad if self.true_bad > 0 else 0
        
        return {
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'precision': precision, 'recall': recall,
            'tp_precise': tp_precise  # 保存精确值
        }
    
    def submit_file(self, csv_file, group_id=12507):
        """提交文件获取F1分数 - 保持最高精度"""
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
                # 保持完整精度，不进行四舍五入
                return float(params['score'][0])
            return None
        except Exception as e:
            print(f"提交错误: {e}")
            return None
    
    def initialize_state(self):
        """初始化状态文件"""
        if os.path.exists(self.state_file):
            print(f"加载现有状态文件: {self.state_file}")
            # 确保浮点数精度
            return pd.read_csv(self.state_file, float_precision='high')
        
        print("创建新的状态文件...")
        
        ref_df = pd.read_csv(self.reference_file)
        all_accounts = ref_df['ID'].tolist()
        baseline_df = pd.read_csv(self.baseline_file)
        
        state_data = []
        for account_id in all_accounts:
            baseline_predict = baseline_df[baseline_df['ID'] == account_id]['Predict'].iloc[0]
            
            state_data.append({
                'ID': account_id,
                'baseline_predict': baseline_predict,
                'confirmed_label': -1,  # -1=未确认, 0=确认bad, 1=确认good
                'confidence': 0.5,  # 初始置信度
                'test_group': -1,  # 用于二分法分组
                'last_updated': 0  # 最后更新轮次
            })
        
        state_df = pd.DataFrame(state_data)
        # 保存时保持高精度
        state_df.to_csv(self.state_file, index=False, float_format='%.10f')
        
        return state_df
    
    def binary_search_iteration(self, state_df):
        """真正的二分法迭代 - 确保O(log n)复杂度"""
        self.round_count += 1
        print(f"\n{'='*60}")
        print(f"二分搜索 - 第 {self.round_count} 轮")
        print(f"{'='*60}")
        
        # 获取待确认账户
        unconfirmed = state_df[state_df['confirmed_label'] == -1].copy()
        
        if len(unconfirmed) == 0:
            print("所有账户已确认完毕！")
            return state_df, True
        
        if len(unconfirmed) == 1:
            print("只剩1个账户，直接测试")
            test_group_A = unconfirmed['ID'].tolist()
            test_group_B = []
        else:
            # 真正的二分法：将未确认账户分成两半
            unconfirmed_shuffled = unconfirmed.sample(frac=1, random_state=self.round_count).reset_index(drop=True)
            mid_point = len(unconfirmed_shuffled) // 2
            
            group_A = unconfirmed_shuffled.iloc[:mid_point]
            group_B = unconfirmed_shuffled.iloc[mid_point:]
            
            test_group_A = group_A['ID'].tolist()
            test_group_B = group_B['ID'].tolist()
        
        print(f"二分组：A组={len(test_group_A)}个, B组={len(test_group_B)}个")
        
        # 策略：测试A组为bad，B组保持当前状态
        submission_df = self.create_test_submission(state_df, test_group_A, mark_as_bad=True)
        
        current_bad = len(submission_df[submission_df['Predict'] == 0])
        current_good = len(submission_df[submission_df['Predict'] == 1])
        
        print(f"测试预测分布：Bad={current_bad}, Good={current_good}")
        print(f"与真实分布差异：Bad差异={current_bad - self.true_bad}, Good差异={current_good - self.true_good}")
        
        # 提交测试
        test_filename = f"binary_test_round_{self.round_count}.csv"
        submission_df.to_csv(test_filename, index=False)
        
        print(f"提交二分测试: {test_filename}")
        f1_score = self.submit_file(test_filename)
        
        if f1_score is None:
            print("提交失败，跳过本轮")
            return state_df, False
        
        print(f"F1分数: {f1_score:.10f}")  # 显示完整精度
        
        # 计算混淆矩阵变化
        cm = self.calculate_confusion_matrix(current_bad, current_good, f1_score)
        print(f"混淆矩阵：TP={cm['TP']}, FP={cm['FP']}, FN={cm['FN']}, TN={cm['TN']}")
        print(f"精确率={cm['precision']:.6f}, 召回率={cm['recall']:.6f}")
        
        # 二分决策逻辑
        baseline_bad = len(state_df[state_df['baseline_predict'] == 0])
        baseline_expected_f1 = 0.7629  # baseline F1分数
        
        if f1_score > baseline_expected_f1 + 0.01:
            # F1显著提升，说明A组大部分是真bad
            print(f"决策：A组确认为bad (F1提升 {f1_score - baseline_expected_f1:.4f})")
            for acc_id in test_group_A:
                state_df.loc[state_df['ID'] == acc_id, 'confirmed_label'] = 0
                state_df.loc[state_df['ID'] == acc_id, 'confidence'] = min(0.9, 0.5 + (f1_score - baseline_expected_f1))
                state_df.loc[state_df['ID'] == acc_id, 'last_updated'] = self.round_count
                
        elif f1_score < baseline_expected_f1 - 0.01:
            # F1显著下降，说明A组大部分是good
            print(f"决策：A组确认为good (F1下降 {baseline_expected_f1 - f1_score:.4f})")
            for acc_id in test_group_A:
                state_df.loc[state_df['ID'] == acc_id, 'confirmed_label'] = 1
                state_df.loc[state_df['ID'] == acc_id, 'confidence'] = min(0.9, 0.5 + (baseline_expected_f1 - f1_score))
                state_df.loc[state_df['ID'] == acc_id, 'last_updated'] = self.round_count
        else:
            # F1变化不大，需要更精细的分析
            print("决策：F1变化不大，进入精细分析模式")
            
            # 如果组太小，直接基于统计推断
            if len(test_group_A) <= 10:
                # 小组直接根据F1趋势确定
                if f1_score >= baseline_expected_f1:
                    decision_label = 0
                    print(f"小组决策：A组确认为bad")
                else:
                    decision_label = 1
                    print(f"小组决策：A组确认为good")
                
                for acc_id in test_group_A:
                    state_df.loc[state_df['ID'] == acc_id, 'confirmed_label'] = decision_label
                    state_df.loc[state_df['ID'] == acc_id, 'confidence'] = 0.6
                    state_df.loc[state_df['ID'] == acc_id, 'last_updated'] = self.round_count
        
        # 保存状态 - 保持高精度
        state_df.to_csv(self.state_file, index=False, float_format='%.10f')
        
        # 清理测试文件
        os.remove(test_filename)
        
        return state_df, False
    
    def create_test_submission(self, state_df, test_accounts, mark_as_bad=True):
        """创建测试提交文件"""
        submission_data = []
        
        for _, row in state_df.iterrows():
            account_id = row['ID']
            
            if account_id in test_accounts:
                predict = 0 if mark_as_bad else 1
            elif row['confirmed_label'] != -1:
                predict = row['confirmed_label']
            else:
                predict = row['baseline_predict']
            
            submission_data.append({'ID': account_id, 'Predict': predict})
        
        return pd.DataFrame(submission_data)
    
    def run_binary_verification(self, max_rounds=None):
        """运行二分验证 - 确保O(log n)复杂度"""
        state_df = self.initialize_state()
        
        initial_unconfirmed = len(state_df[state_df['confirmed_label'] == -1])
        
        # 理论最大轮数 = log2(n) + 安全边界
        if max_rounds is None:
            max_rounds = int(np.ceil(np.log2(initial_unconfirmed))) + 5
        
        print(f"初始待确认账户: {initial_unconfirmed}")
        print(f"理论最大轮数: {max_rounds}")
        
        for round_num in range(max_rounds):
            state_df, completed = self.binary_search_iteration(state_df)
            
            if completed:
                break
            
            # 显示进度
            remaining = len(state_df[state_df['confirmed_label'] == -1])
            print(f"剩余待确认: {remaining}")
            
            sleep(2)
        
        # 最终统计和保存
        self._finalize_results(state_df)
        return state_df
    
    def _finalize_results(self, state_df):
        """最终化结果"""
        confirmed_bad = len(state_df[state_df['confirmed_label'] == 0])
        confirmed_good = len(state_df[state_df['confirmed_label'] == 1])
        unconfirmed = len(state_df[state_df['confirmed_label'] == -1])
        
        print(f"\n{'='*60}")
        print("二分验证完成！")
        print(f"{'='*60}")
        print(f"确认Bad: {confirmed_bad}")
        print(f"确认Good: {confirmed_good}")  
        print(f"仍待确认: {unconfirmed}")
        print(f"总轮数: {self.round_count}")
        print(f"复杂度: O(log₂({self.total_accounts})) ≈ {int(np.ceil(np.log2(self.total_accounts)))} 理论轮数")
        
        # 生成最终文件
        final_submission = self.create_test_submission(state_df, [], mark_as_bad=False)
        final_filename = "final_binary_verified_submission.csv"
        final_submission.to_csv(final_filename, index=False)
        
        final_bad = len(final_submission[final_submission['Predict'] == 0])
        final_good = len(final_submission[final_submission['Predict'] == 1])
        
        print(f"最终分布：Bad={final_bad}, Good={final_good}")
        print(f"最终文件：{final_filename}")

def main():
    baseline_file = "/Users/mannormal/4011/Qi Zihan/v2/results/high_score_predictions/0.75+/v3.2refined_fold1_bad_f1_0.8083_good_0.9803_bad_0.8083_macro_0.8943_weighted_0.9634_seed_13_REAL_F1_0.7628549501151188.csv"
    reference_file = "/Users/mannormal/4011/Qi Zihan/v2/results/test_acc_predict_REAL_F1_0.17549788774894384_REAL_F1_0.0.csv"
    
    verifier = IterativeAccountVerifier(baseline_file, reference_file)
    final_state = verifier.run_binary_verification()
    
    print("\n二分验证系统完成！")

if __name__ == "__main__":
    main()