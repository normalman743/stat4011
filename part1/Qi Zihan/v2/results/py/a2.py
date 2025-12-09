import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class VoteThresholdAnalyzer:
    def __init__(self):
        # 真实分布
        self.real_bad = 727
        self.real_good = 6831
        self.total = 7558
        
        # 存储各阈值的数据（只使用正向策略，因为效果更好）
        self.threshold_data = self._initialize_data()
    
    def _initialize_data(self) -> Dict:
        """初始化各阈值的混淆矩阵数据（基于正向策略）"""
        data = {
            0: {'pred_bad': 1665, 'pred_good': 5893, 'tp_bad': 652, 'fn_bad': 75, 'fp_bad': 1013, 'tn_good': 5818},
            1: {'pred_bad': 1496, 'pred_good': 6062, 'tp_bad': 642, 'fn_bad': 85, 'fp_bad': 854, 'tn_good': 5977},
            2: {'pred_bad': 1407, 'pred_good': 6151, 'tp_bad': 637, 'fn_bad': 90, 'fp_bad': 770, 'tn_good': 6061},
            3: {'pred_bad': 1331, 'pred_good': 6227, 'tp_bad': 628, 'fn_bad': 99, 'fp_bad': 703, 'tn_good': 6128},
            4: {'pred_bad': 1274, 'pred_good': 6284, 'tp_bad': 625, 'fn_bad': 102, 'fp_bad': 649, 'tn_good': 6182},
            5: {'pred_bad': 1225, 'pred_good': 6333, 'tp_bad': 618, 'fn_bad': 109, 'fp_bad': 607, 'tn_good': 6224},
            6: {'pred_bad': 1180, 'pred_good': 6378, 'tp_bad': 614, 'fn_bad': 113, 'fp_bad': 566, 'tn_good': 6265},
            7: {'pred_bad': 1153, 'pred_good': 6405, 'tp_bad': 613, 'fn_bad': 114, 'fp_bad': 540, 'tn_good': 6291},
            8: {'pred_bad': 1119, 'pred_good': 6439, 'tp_bad': 605, 'fn_bad': 122, 'fp_bad': 514, 'tn_good': 6317},
            9: {'pred_bad': 1089, 'pred_good': 6469, 'tp_bad': 602, 'fn_bad': 125, 'fp_bad': 487, 'tn_good': 6344},
            10: {'pred_bad': 1070, 'pred_good': 6488, 'tp_bad': 600, 'fn_bad': 127, 'fp_bad': 470, 'tn_good': 6361},
            11: {'pred_bad': 1045, 'pred_good': 6513, 'tp_bad': 597, 'fn_bad': 130, 'fp_bad': 448, 'tn_good': 6383},
            12: {'pred_bad': 1016, 'pred_good': 6542, 'tp_bad': 595, 'fn_bad': 132, 'fp_bad': 421, 'tn_good': 6410},
            13: {'pred_bad': 979, 'pred_good': 6579, 'tp_bad': 590, 'fn_bad': 137, 'fp_bad': 389, 'tn_good': 6442},
            14: {'pred_bad': 958, 'pred_good': 6600, 'tp_bad': 588, 'fn_bad': 139, 'fp_bad': 370, 'tn_good': 6461},
            15: {'pred_bad': 924, 'pred_good': 6634, 'tp_bad': 583, 'fn_bad': 144, 'fp_bad': 341, 'tn_good': 6490},
            16: {'pred_bad': 885, 'pred_good': 6673, 'tp_bad': 576, 'fn_bad': 151, 'fp_bad': 309, 'tn_good': 6522},
            17: {'pred_bad': 781, 'pred_good': 6777, 'tp_bad': 559, 'fn_bad': 168, 'fp_bad': 222, 'tn_good': 6609},
            18: {'pred_bad': 763, 'pred_good': 6795, 'tp_bad': 553, 'fn_bad': 174, 'fp_bad': 210, 'tn_good': 6621},
            19: {'pred_bad': 753, 'pred_good': 6805, 'tp_bad': 550, 'fn_bad': 177, 'fp_bad': 203, 'tn_good': 6628},
            20: {'pred_bad': 738, 'pred_good': 6820, 'tp_bad': 542, 'fn_bad': 185, 'fp_bad': 196, 'tn_good': 6635},
            21: {'pred_bad': 728, 'pred_good': 6830, 'tp_bad': 538, 'fn_bad': 189, 'fp_bad': 190, 'tn_good': 6641},
            22: {'pred_bad': 707, 'pred_good': 6851, 'tp_bad': 534, 'fn_bad': 193, 'fp_bad': 173, 'tn_good': 6658},
            23: {'pred_bad': 696, 'pred_good': 6862, 'tp_bad': 528, 'fn_bad': 199, 'fp_bad': 168, 'tn_good': 6663},
            24: {'pred_bad': 685, 'pred_good': 6873, 'tp_bad': 523, 'fn_bad': 204, 'fp_bad': 162, 'tn_good': 6669},
            25: {'pred_bad': 667, 'pred_good': 6891, 'tp_bad': 518, 'fn_bad': 209, 'fp_bad': 149, 'tn_good': 6682},
            26: {'pred_bad': 662, 'pred_good': 6896, 'tp_bad': 517, 'fn_bad': 210, 'fp_bad': 145, 'tn_good': 6686},
            27: {'pred_bad': 655, 'pred_good': 6903, 'tp_bad': 516, 'fn_bad': 211, 'fp_bad': 139, 'tn_good': 6692},
            28: {'pred_bad': 655, 'pred_good': 6903, 'tp_bad': 516, 'fn_bad': 211, 'fp_bad': 139, 'tn_good': 6692},
            29: {'pred_bad': 646, 'pred_good': 6912, 'tp_bad': 512, 'fn_bad': 215, 'fp_bad': 134, 'tn_good': 6697},
            30: {'pred_bad': 629, 'pred_good': 6929, 'tp_bad': 504, 'fn_bad': 223, 'fp_bad': 125, 'tn_good': 6706},
            31: {'pred_bad': 607, 'pred_good': 6951, 'tp_bad': 494, 'fn_bad': 233, 'fp_bad': 113, 'tn_good': 6718},
            32: {'pred_bad': 584, 'pred_good': 6974, 'tp_bad': 486, 'fn_bad': 241, 'fp_bad': 98, 'tn_good': 6733},
            33: {'pred_bad': 560, 'pred_good': 6998, 'tp_bad': 481, 'fn_bad': 246, 'fp_bad': 79, 'tn_good': 6752},
            34: {'pred_bad': 499, 'pred_good': 7059, 'tp_bad': 458, 'fn_bad': 269, 'fp_bad': 41, 'tn_good': 6790},
            35: {'pred_bad': 457, 'pred_good': 7101, 'tp_bad': 431, 'fn_bad': 296, 'fp_bad': 26, 'tn_good': 6805},
            36: {'pred_bad': 420, 'pred_good': 7138, 'tp_bad': 406, 'fn_bad': 321, 'fp_bad': 14, 'tn_good': 6817}
        }
        return data
    
    def analyze_threshold_changes(self) -> pd.DataFrame:
        """分析阈值变化对预测结果的影响"""
        results = []
        
        for i in range(len(self.threshold_data) - 1):
            curr = self.threshold_data[i]
            next_data = self.threshold_data[i + 1]
            
            # 预测分布的变化
            pred_bad_change = next_data['pred_bad'] - curr['pred_bad']
            pred_good_change = next_data['pred_good'] - curr['pred_good']
            
            # 实际正确识别的变化
            tp_bad_change = next_data['tp_bad'] - curr['tp_bad']
            tn_good_change = next_data['tn_good'] - curr['tn_good']
            
            # 错误的变化
            fp_bad_change = next_data['fp_bad'] - curr['fp_bad']
            fn_bad_change = next_data['fn_bad'] - curr['fn_bad']
            
            # 计算变化的准确率
            # 当预测为bad减少时，看减少的部分中有多少是正确的（原本是fp）
            if pred_bad_change < 0:
                # 预测bad减少，意味着有些样本从bad改为good
                # 这些样本中原本就是good的比例
                correct_change_ratio = -fp_bad_change / -pred_bad_change if pred_bad_change != 0 else 0
            else:
                correct_change_ratio = 0
            
            # 计算各种指标
            precision_bad = curr['tp_bad'] / curr['pred_bad'] if curr['pred_bad'] > 0 else 0
            recall_bad = curr['tp_bad'] / self.real_bad
            f1_bad = 2 * precision_bad * recall_bad / (precision_bad + recall_bad) if (precision_bad + recall_bad) > 0 else 0
            
            results.append({
                '阈值': i,
                '预测Bad数': curr['pred_bad'],
                '预测Good数': curr['pred_good'],
                '预测Bad变化': pred_bad_change,
                '预测Good变化': pred_good_change,
                'TP_Bad': curr['tp_bad'],
                'FP_Bad': curr['fp_bad'],
                'TP_Bad变化': tp_bad_change,
                'FP_Bad变化': fp_bad_change,
                'Bad精确率': precision_bad,
                'Bad召回率': recall_bad,
                'Bad_F1': f1_bad,
                '变化正确率': correct_change_ratio * 100
            })
        
        return pd.DataFrame(results)
    
    def greedy_optimization(self, target_f1: float = 1.0) -> Dict:
        """
        贪心算法：从概率最高的样本开始逐个分类，优化Bad F1
        
        基于投票数从高到低排序，逐步调整阈值找到最优解
        """
        print("开始贪心优化算法...")
        print(f"目标: Bad F1 = {target_f1}")
        print(f"真实分布: Bad={self.real_bad}, Good={self.real_good}")
        print("-" * 60)
        
        best_result = {
            'threshold': None,
            'f1': 0,
            'precision': 0,
            'recall': 0,
            'tp_bad': 0,
            'fp_bad': 0,
            'fn_bad': self.real_bad,
            'strategy': []
        }
        
        # 分析每个阈值的边际效益
        marginal_benefits = []
        
        for threshold in sorted(self.threshold_data.keys(), reverse=True):
            data = self.threshold_data[threshold]
            
            precision = data['tp_bad'] / data['pred_bad'] if data['pred_bad'] > 0 else 0
            recall = data['tp_bad'] / self.real_bad
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 计算边际效益：每预测一个bad的正确率
            marginal_precision = precision
            
            marginal_benefits.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp_bad': data['tp_bad'],
                'fp_bad': data['fp_bad'],
                'pred_bad': data['pred_bad'],
                'marginal_precision': marginal_precision,
                'cost_per_tp': data['pred_bad'] / data['tp_bad'] if data['tp_bad'] > 0 else float('inf')
            })
            
            if f1 > best_result['f1']:
                best_result = {
                    'threshold': threshold,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'tp_bad': data['tp_bad'],
                    'fp_bad': data['fp_bad'],
                    'fn_bad': data['fn_bad'],
                    'pred_bad': data['pred_bad']
                }
        
        # 输出优化策略
        print("\n优化策略（按精确率从高到低）:")
        print("-" * 80)
        print(f"{'阈值':^6} | {'预测Bad':^8} | {'TP_Bad':^7} | {'FP_Bad':^7} | {'精确率':^8} | {'召回率':^8} | {'F1分数':^8} | {'每TP成本':^10}")
        print("-" * 80)
        
        # 按精确率排序
        marginal_benefits.sort(key=lambda x: x['precision'], reverse=True)
        
        cumulative_tp = 0
        cumulative_fp = 0
        
        for benefit in marginal_benefits[:10]:  # 显示前10个最优策略
            print(f"{benefit['threshold']:^6} | {benefit['pred_bad']:^8} | {benefit['tp_bad']:^7} | "
                  f"{benefit['fp_bad']:^7} | {benefit['precision']:^8.3f} | {benefit['recall']:^8.3f} | "
                  f"{benefit['f1']:^8.3f} | {benefit['cost_per_tp']:^10.2f}")
        
        print("-" * 80)
        print(f"\n最佳方案: 阈值={best_result['threshold']}")
        print(f"  - Bad F1: {best_result['f1']:.4f}")
        print(f"  - 精确率: {best_result['precision']:.4f}")
        print(f"  - 召回率: {best_result['recall']:.4f}")
        print(f"  - 预测Bad数: {best_result['pred_bad']}")
        print(f"  - 正确识别Bad: {best_result['tp_bad']}/{self.real_bad}")
        print(f"  - 错误识别为Bad: {best_result['fp_bad']}/{self.real_good}")
        
        # 理论最优解分析
        print("\n理论分析:")
        print("-" * 60)
        print(f"要达到Bad F1=1.0，需要:")
        print(f"  1. 精确率=1.0: 所有预测为Bad的都是真的Bad (FP=0)")
        print(f"  2. 召回率=1.0: 所有真的Bad都被识别出来 (FN=0)")
        print(f"  即: 需要正确识别所有{self.real_bad}个Bad，且不误判任何Good")
        print(f"\n当前最佳方案距离目标:")
        print(f"  - 还有 {self.real_bad - best_result['tp_bad']} 个Bad未识别")
        print(f"  - 误判了 {best_result['fp_bad']} 个Good为Bad")
        
        return best_result
    
    def plot_analysis(self, df: pd.DataFrame):
        """绘制分析图表"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. 预测分布变化
        ax = axes[0, 0]
        ax.plot(df['阈值'], df['预测Bad数'], 'b-', label='预测Bad数', marker='o')
        ax.plot(df['阈值'], df['预测Good数'], 'g-', label='预测Good数', marker='s')
        ax.set_xlabel('阈值')
        ax.set_ylabel('预测数量')
        ax.set_title('预测分布随阈值变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. TP和FP变化
        ax = axes[0, 1]
        ax.plot(df['阈值'], df['TP_Bad'], 'r-', label='TP (正确识别Bad)', marker='o')
        ax.plot(df['阈值'], df['FP_Bad'], 'orange', label='FP (误判为Bad)', marker='s')
        ax.set_xlabel('阈值')
        ax.set_ylabel('数量')
        ax.set_title('正确识别vs误判')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 精确率、召回率和F1
        ax = axes[0, 2]
        ax.plot(df['阈值'], df['Bad精确率'], 'b-', label='精确率', marker='o')
        ax.plot(df['阈值'], df['Bad召回率'], 'g-', label='召回率', marker='s')
        ax.plot(df['阈值'], df['Bad_F1'], 'r-', label='F1分数', marker='^', linewidth=2)
        ax.set_xlabel('阈值')
        ax.set_ylabel('分数')
        ax.set_title('Bad类性能指标')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 边际变化
        ax = axes[1, 0]
        ax.bar(df['阈值'], df['预测Bad变化'], color=['red' if x < 0 else 'blue' for x in df['预测Bad变化']])
        ax.set_xlabel('阈值')
        ax.set_ylabel('预测Bad数变化')
        ax.set_title('阈值增加时预测Bad数的变化')
        ax.grid(True, alpha=0.3)
        
        # 5. 变化正确率
        ax = axes[1, 1]
        ax.plot(df['阈值'], df['变化正确率'], 'purple', marker='o')
        ax.set_xlabel('阈值')
        ax.set_ylabel('正确率 (%)')
        ax.set_title('阈值变化的正确率')
        ax.grid(True, alpha=0.3)
        
        # 6. 精确率vs召回率权衡
        ax = axes[1, 2]
        ax.scatter(df['Bad召回率'], df['Bad精确率'], c=df['阈值'], cmap='viridis', s=50)
        for i, txt in enumerate(df['阈值']):
            if i % 5 == 0:  # 每5个标注一个
                ax.annotate(txt, (df['Bad召回率'].iloc[i], df['Bad精确率'].iloc[i]))
        ax.set_xlabel('召回率')
        ax.set_ylabel('精确率')
        ax.set_title('精确率-召回率权衡曲线')
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(ax.scatter(df['Bad召回率'], df['Bad精确率'], c=df['阈值'], cmap='viridis'), ax=ax, label='阈值')
        
        plt.tight_layout()
        plt.show()

# 使用示例
def main():
    analyzer = VoteThresholdAnalyzer()
    
    # 1. 分析阈值变化
    print("=" * 80)
    print("阈值变化分析")
    print("=" * 80)
    df = analyzer.analyze_threshold_changes()
    
    # 显示关键统计
    print("\n关键发现:")
    print("-" * 60)
    print(f"1. 阈值从0到36，预测Bad数从{df['预测Bad数'].iloc[0]}减少到{df['预测Bad数'].iloc[-1]}")
    print(f"2. 最高Bad F1分数: {df['Bad_F1'].max():.4f} (阈值={df.loc[df['Bad_F1'].idxmax(), '阈值']})")
    print(f"3. 最高精确率: {df['Bad精确率'].max():.4f} (阈值={df.loc[df['Bad精确率'].idxmax(), '阈值']})")
    print(f"4. 最高召回率: {df['Bad召回率'].max():.4f} (阈值={df.loc[df['Bad召回率'].idxmax(), '阈值']})")
    
    # 显示详细表格
    print("\n详细分析表:")
    print(df[['阈值', '预测Bad数', 'TP_Bad', 'FP_Bad', 'Bad精确率', 'Bad召回率', 'Bad_F1']].to_string(index=False))
    
    # 2. 贪心优化
    print("\n" + "=" * 80)
    print("贪心优化算法")
    print("=" * 80)
    best_solution = analyzer.greedy_optimization()
    
    # 3. 绘制图表
    analyzer.plot_analysis(df)

if __name__ == "__main__":
    main()