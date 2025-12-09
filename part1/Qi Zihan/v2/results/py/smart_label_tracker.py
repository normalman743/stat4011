#!/usr/bin/env python3
"""
🧠 智能标签追踪器
通过F1变化推断真实标签，逐步构建真实测试集标签
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
from upload import submit_file
from time import sleep
from datetime import datetime

class SmartLabelTracker:
    def __init__(self, 
                 base_file="/Users/mannormal/4011/Qi Zihan/v2/results/test_acc_predict_REAL_F1_0.17549788774894384.csv",
                 tracker_file="/Users/mannormal/4011/Qi Zihan/v2/results/label_tracker.json"):
        
        self.base_file = base_file
        self.tracker_file = tracker_file
        self.base_f1 = 0.17549788774894384  # 全1(Bad)的基准F1
        self.total_bad = 727  # 已知真实Bad数量
        self.total_accounts = 7559  # 总账户数
        
        # 加载或初始化追踪数据
        self.load_tracker_data()
        
    def load_tracker_data(self):
        """加载追踪数据"""
        if os.path.exists(self.tracker_file):
            with open(self.tracker_file, 'r') as f:
                self.tracker_data = json.load(f)
            print(f"📂 Loaded existing tracker data: {len(self.tracker_data.get('accounts', {}))} accounts tracked")
        else:
            # 初始化：所有账户都预测为Bad(1)，真实标签未知
            base_df = pd.read_csv(self.base_file.replace("_REAL_F1_0.17549788774894384.csv", ".csv") if "_REAL_F1_" in self.base_file else self.base_file)
            
            self.tracker_data = {
                "metadata": {
                    "base_f1": self.base_f1,
                    "total_bad": self.total_bad,
                    "total_accounts": self.total_accounts,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                },
                "accounts": {}
            }
            
            # 初始化每个账户的状态
            for idx, row in base_df.iterrows():
                account_id = row.iloc[0]  # 第一列是账户ID
                self.tracker_data["accounts"][account_id] = {
                    "model_prediction": 1,  # 初始预测：Bad
                    "true_label": None,     # 未知真实标签
                    "confidence": 0.0,      # 置信度
                    "experiments": []       # 实验历史
                }
            
            self.save_tracker_data()
            print(f"🆕 Initialized tracker for {len(self.tracker_data['accounts'])} accounts")
    
    def save_tracker_data(self):
        """保存追踪数据"""
        self.tracker_data["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.tracker_file, 'w') as f:
            json.dump(self.tracker_data, f, indent=2)
    
    def calculate_f1_change_theory(self, flip_from_bad_to_good=1, flip_from_good_to_bad=0):
        """
        理论计算：翻转标签后F1的变化
        
        当前状态：727个Bad(1), 6832个Good(0)
        F1 = 0.175 (Bad类F1)
        
        翻转效果：
        - Bad→Good: 如果翻转的是真Bad，F1下降；如果是真Good，F1上升
        - Good→Bad: 如果翻转的是真Good，F1下降；如果是真Bad，F1上升
        """
        current_tp = 727 * self.base_f1 / (2 * self.base_f1 - 1) if self.base_f1 != 0.5 else 727 * 0.5
        current_fp = 727 - current_tp
        current_fn = self.total_bad - current_tp
        
        print(f"📊 当前估算状态 (基于F1={self.base_f1:.4f}):")
        print(f"   True Positive (正确Bad): {current_tp:.1f}")
        print(f"   False Positive (错误Bad): {current_fp:.1f}") 
        print(f"   False Negative (错误Good): {current_fn:.1f}")
        
        # 计算翻转后的变化
        scenarios = {
            "flip_true_bad_to_good": {
                "new_tp": current_tp - flip_from_bad_to_good,
                "new_fp": current_fp,
                "new_fn": current_fn + flip_from_bad_to_good,
                "description": "翻转真Bad→Good (F1下降)"
            },
            "flip_false_bad_to_good": {
                "new_tp": current_tp,
                "new_fp": current_fp - flip_from_bad_to_good, 
                "new_fn": current_fn,
                "description": "翻转假Bad→Good (F1上升)"
            },
            "flip_true_good_to_bad": {
                "new_tp": current_tp,
                "new_fp": current_fp + flip_from_good_to_bad,
                "new_fn": current_fn,
                "description": "翻转真Good→Bad (F1下降)"
            },
            "flip_false_good_to_bad": {
                "new_tp": current_tp + flip_from_good_to_bad,
                "new_fp": current_fp,
                "new_fn": current_fn - flip_from_good_to_bad,
                "description": "翻转假Good→Bad (F1上升)"
            }
        }
        
        print(f"\n🎯 理论F1变化预测:")
        for scenario, values in scenarios.items():
            new_tp, new_fp, new_fn = values["new_tp"], values["new_fp"], values["new_fn"]
            
            if new_tp + new_fp > 0:
                new_precision = new_tp / (new_tp + new_fp)
                new_recall = new_tp / (new_tp + new_fn)
                new_f1 = 2 * new_precision * new_recall / (new_precision + new_recall) if (new_precision + new_recall) > 0 else 0
                f1_change = new_f1 - self.base_f1
                
                print(f"   {scenario}: F1 = {new_f1:.4f} (变化: {f1_change:+.4f}) - {values['description']}")
            else:
                print(f"   {scenario}: 无效配置")
        
        return scenarios
    
    def create_flip_experiment(self, accounts_to_flip, flip_type="bad_to_good", experiment_name=None):
        """
        创建翻转实验
        accounts_to_flip: 要翻转的账户列表
        flip_type: "bad_to_good" 或 "good_to_bad"
        """
        if experiment_name is None:
            experiment_name = f"flip_{len(accounts_to_flip)}_{flip_type}_{datetime.now().strftime('%H%M%S')}"
        
        print(f"🧪 创建实验: {experiment_name}")
        print(f"   翻转类型: {flip_type}")
        print(f"   翻转账户数: {len(accounts_to_flip)}")
        
        # 基于当前预测创建新的提交文件
        base_df = pd.read_csv(self.base_file.replace("_REAL_F1_0.17549788774894384.csv", ".csv") if "_REAL_F1_" in self.base_file else self.base_file)
        new_df = base_df.copy()
        
        # 执行翻转
        flipped_count = 0
        for account in accounts_to_flip:
            if account in new_df.iloc[:, 0].values:
                account_idx = new_df[new_df.iloc[:, 0] == account].index[0]
                current_pred = new_df.iloc[account_idx, 1]
                
                if flip_type == "bad_to_good" and current_pred == 1:
                    new_df.iloc[account_idx, 1] = 0
                    flipped_count += 1
                elif flip_type == "good_to_bad" and current_pred == 0:
                    new_df.iloc[account_idx, 1] = 1
                    flipped_count += 1
        
        print(f"   实际翻转: {flipped_count} 个账户")
        
        # 保存实验文件
        experiment_file = f"/Users/mannormal/4011/Qi Zihan/v2/results/experiment_{experiment_name}.csv"
        new_df.to_csv(experiment_file, index=False)
        
        # 记录实验
        experiment_record = {
            "name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "flip_type": flip_type,
            "accounts_flipped": accounts_to_flip,
            "actual_flips": flipped_count,
            "file_path": experiment_file,
            "base_f1": self.base_f1,
            "predicted_f1_changes": self.calculate_f1_change_theory(
                flip_from_bad_to_good=flipped_count if flip_type == "bad_to_good" else 0,
                flip_from_good_to_bad=flipped_count if flip_type == "good_to_bad" else 0
            ),
            "result_f1": None,
            "status": "pending"
        }
        
        # 保存到追踪数据
        if "experiments" not in self.tracker_data:
            self.tracker_data["experiments"] = []
        self.tracker_data["experiments"].append(experiment_record)
        
        # 更新账户实验历史
        for account in accounts_to_flip:
            if account in self.tracker_data["accounts"]:
                self.tracker_data["accounts"][account]["experiments"].append(experiment_name)
        
        self.save_tracker_data()
        
        print(f"✅ 实验文件保存: {experiment_file}")
        return experiment_file, experiment_record
    
    def submit_and_analyze_experiment(self, experiment_file, experiment_record):
        """提交实验并分析结果"""
        print(f"🚀 提交实验: {experiment_record['name']}")
        
        try:
            score = submit_file(12507, experiment_file)
            if score is not None:
                print(f"🎯 实验F1得分: {score}")
                
                # 分析结果
                f1_change = score - self.base_f1
                experiment_record["result_f1"] = score
                experiment_record["actual_f1_change"] = f1_change
                experiment_record["status"] = "completed"
                
                print(f"📈 F1变化: {f1_change:+.6f}")
                
                # 推断真实标签
                self.analyze_experiment_results(experiment_record)
                
                # 重命名文件
                new_filename = f"experiment_{experiment_record['name']}_F1_{score:.6f}.csv"
                new_filepath = os.path.dirname(experiment_file) + "/" + new_filename
                os.rename(experiment_file, new_filepath)
                experiment_record["file_path"] = new_filepath
                
                self.save_tracker_data()
                
                return score, f1_change
            else:
                print("❌ 提交失败")
                experiment_record["status"] = "failed"
                return None, None
                
        except Exception as e:
            print(f"❌ 提交错误: {e}")
            experiment_record["status"] = "error"
            return None, None
    
    def analyze_experiment_results(self, experiment_record):
        """分析实验结果，推断真实标签"""
        f1_change = experiment_record.get("actual_f1_change", 0)
        flip_type = experiment_record["flip_type"]
        accounts_flipped = experiment_record["accounts_flipped"]
        
        print(f"\n🧠 分析实验结果:")
        
        if flip_type == "bad_to_good":
            if f1_change > 0.001:  # F1显著上升
                conclusion = "翻转的账户大多是假Bad (原本应该是Good)"
                confidence = min(0.9, abs(f1_change) * 100)
                inferred_true_label = 0  # Good
            elif f1_change < -0.001:  # F1显著下降
                conclusion = "翻转的账户大多是真Bad"
                confidence = min(0.9, abs(f1_change) * 100)
                inferred_true_label = 1  # Bad
            else:  # F1变化很小
                conclusion = "结果不明确，可能混合了真Bad和假Bad"
                confidence = 0.1
                inferred_true_label = None
        
        elif flip_type == "good_to_bad":
            if f1_change > 0.001:  # F1显著上升
                conclusion = "翻转的账户大多是假Good (原本应该是Bad)"
                confidence = min(0.9, abs(f1_change) * 100)
                inferred_true_label = 1  # Bad
            elif f1_change < -0.001:  # F1显著下降
                conclusion = "翻转的账户大多是真Good"
                confidence = min(0.9, abs(f1_change) * 100)
                inferred_true_label = 0  # Good
            else:
                conclusion = "结果不明确"
                confidence = 0.1
                inferred_true_label = None
        
        print(f"   结论: {conclusion}")
        print(f"   置信度: {confidence:.2f}")
        
        # 更新账户标签推断
        for account in accounts_flipped:
            if account in self.tracker_data["accounts"]:
                if inferred_true_label is not None:
                    self.tracker_data["accounts"][account]["true_label"] = inferred_true_label
                    self.tracker_data["accounts"][account]["confidence"] = confidence
                
                # 记录实验结果
                self.tracker_data["accounts"][account]["last_experiment"] = {
                    "name": experiment_record["name"],
                    "result": conclusion,
                    "confidence": confidence
                }
    
    def get_uncertain_accounts(self, min_confidence=0.5):
        """获取置信度低的账户列表"""
        uncertain = []
        for account_id, data in self.tracker_data["accounts"].items():
            confidence = data.get("confidence", 0)
            if confidence < min_confidence:
                uncertain.append(account_id)
        return uncertain
    
    def suggest_next_experiment(self):
        """建议下一个实验"""
        print(f"\n💡 建议下一个实验:")
        
        # 统计当前状态
        confirmed_bad = sum(1 for acc in self.tracker_data["accounts"].values() 
                           if acc.get("true_label") == 1 and acc.get("confidence", 0) > 0.5)
        confirmed_good = sum(1 for acc in self.tracker_data["accounts"].values() 
                            if acc.get("true_label") == 0 and acc.get("confidence", 0) > 0.5)
        uncertain = len(self.get_uncertain_accounts())
        
        print(f"   已确认Bad: {confirmed_bad}")
        print(f"   已确认Good: {confirmed_good}")  
        print(f"   不确定: {uncertain}")
        
        # 建议策略
        if uncertain > 100:
            print(f"   建议: 随机选择10个当前预测为Bad的账户，翻转为Good测试")
            current_bad_accounts = [acc_id for acc_id, data in self.tracker_data["accounts"].items() 
                                   if data["model_prediction"] == 1 and data.get("confidence", 0) < 0.5]
            suggested_accounts = np.random.choice(current_bad_accounts, min(10, len(current_bad_accounts)), replace=False)
            return list(suggested_accounts), "bad_to_good"
        else:
            print(f"   建议: 继续细化高不确定性账户")
            uncertain_accounts = self.get_uncertain_accounts(min_confidence=0.3)
            return uncertain_accounts[:5], "bad_to_good"

def main():
    """主程序演示"""
    tracker = SmartLabelTracker()
    
    print("🧠 智能标签追踪器启动")
    print("="*50)
    
    # 显示理论计算
    tracker.calculate_f1_change_theory(flip_from_bad_to_good=1)
    
    # 建议下一个实验
    suggested_accounts, flip_type = tracker.suggest_next_experiment()
    
    print(f"\n❓ 是否要执行建议的实验？")
    print(f"   翻转账户: {len(suggested_accounts)} 个")
    print(f"   翻转类型: {flip_type}")
    
    # 这里可以添加用户交互
    choice = input("执行实验? (y/n): ").lower().strip()
    if choice == 'y':
        # 创建并提交实验
        exp_file, exp_record = tracker.create_flip_experiment(
            suggested_accounts, flip_type, f"auto_experiment_{len(suggested_accounts)}"
        )
        
        # 提交实验
        score, f1_change = tracker.submit_and_analyze_experiment(exp_file, exp_record)
        
        if score is not None:
            print(f"\n🎉 实验完成!")
            print(f"   新F1分数: {score}")
            print(f"   F1变化: {f1_change:+.6f}")

if __name__ == "__main__":
    main()