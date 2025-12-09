import sys
sys.path.append('/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v5')
from simulator import get_confusion_matrix
import pandas as pd

# 文件路径列表
file_paths = [
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/AGGRESSIVE_AGGRESSIVE_VOTING_REAL_F1_0.7521489971346705.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/GRADIENT_TUNE_10PCT_REAL_F1_0.7611336032388665.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/GRADIENT_TUNE_7PCT_REAL_F1_0.7531847133757962.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/GRADIENT_TUNE_8PCT_REAL_F1_0.7528174305033809.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/GRADIENT_TUNE_9PCT_REAL_F1_0.7533759772565743.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/result.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/submit.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold1_bad_f1_0.7778_good_0.9765_bad_0.7778_macro_0.8771_weighted_0.9570_seed_13_REAL_F1_0.7549378200438918.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold1_bad_f1_0.8083_good_0.9803_bad_0.8083_macro_0.8943_weighted_0.9634_seed_13_REAL_F1_0.7628549501151188.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold4_bad_f1_0.8250_good_0.9814_bad_0.8250_macro_0.9032_weighted_0.9661_seed_13_REAL_F1_0.7525325615050651_REAL_F1_0.7525325615050651.csv",
    "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/v3.2refined_fold5_bad_f1_0.8401_good_0.9838_bad_0.8401_macro_0.9119_weighted_0.9697_seed_13_REAL_F1_0.7579273008507347.csv"
]

# 真实标签文件路径
real_flag_path = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/融合二分模型_最终版 copy.csv"

print("=" * 100)
print("混淆矩阵详细分析")
print("=" * 100)
print()

results = []

for i, file_path in enumerate(file_paths, 1):
    file_name = file_path.split('/')[-1]
    
    print(f"{i}. {file_name}")
    print("-" * 100)
    
    try:
        # 获取混淆矩阵
        confusion = get_confusion_matrix(file_path, real_flag_path)
        
        if confusion:
            # 显示混淆矩阵
            print(f"   混淆矩阵:")
            print(f"      TP (真正例): {confusion['TP']:4d}  |  FN (假负例): {confusion['FN']:4d}")
            print(f"      FP (假正例): {confusion['FP']:4d}  |  TN (真负例): {confusion['TN']:4d}")
            print()
            print(f"   评估指标:")
            print(f"      Accuracy  (准确率):   {confusion['accuracy']:.4f} ({confusion['accuracy']*100:.2f}%)")
            print(f"      Precision (精确率):   {confusion['precision']:.4f} ({confusion['precision']*100:.2f}%)")
            print(f"      Recall    (召回率):   {confusion['recall']:.4f} ({confusion['recall']*100:.2f}%)")
            print(f"      Specificity (特异度): {confusion['specificity']:.4f} ({confusion['specificity']*100:.2f}%)")
            print(f"      F1 Score  (F1分数):   {confusion['f1_score']:.4f} ({confusion['f1_score']*100:.2f}%)")
            print()
            
            # 保存结果
            results.append({
                'file_name': file_name,
                'TP': confusion['TP'],
                'FP': confusion['FP'],
                'FN': confusion['FN'],
                'TN': confusion['TN'],
                'accuracy': confusion['accuracy'],
                'precision': confusion['precision'],
                'recall': confusion['recall'],
                'specificity': confusion['specificity'],
                'f1_score': confusion['f1_score']
            })
        else:
            print(f"   ❌ 无法计算混淆矩阵")
            print()
            
    except Exception as e:
        print(f"   ❌ 处理文件时出错: {e}")
        print()
        continue

print("=" * 100)
print("总结对比 (按F1分数排序)")
print("=" * 100)
print()

# 按F1分数排序
results.sort(key=lambda x: x['f1_score'], reverse=True)

print(f"{'排名':<4} {'文件名':<60} {'F1分数':<10} {'准确率':<10} {'精确率':<10} {'召回率':<10}")
print("-" * 100)

for rank, result in enumerate(results, 1):
    print(f"{rank:<4} {result['file_name'][:60]:<60} "
          f"{result['f1_score']:.4f}    "
          f"{result['accuracy']:.4f}    "
          f"{result['precision']:.4f}    "
          f"{result['recall']:.4f}")

# 保存到CSV
df = pd.DataFrame(results)
df = df[['file_name', 'TP', 'FP', 'FN', 'TN', 'accuracy', 'precision', 'recall', 'specificity', 'f1_score']]
df.columns = ['文件名', 'TP', 'FP', 'FN', 'TN', '准确率', '精确率', '召回率', '特异度', 'F1分数']

output_path = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/Qi Zihan/v7/confusion_matrices_analysis.csv"
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print()
print(f"详细结果已保存到: {output_path}")

# 额外分析：找出F1最高的文件
print()
print("=" * 100)
print("最佳模型 (F1分数最高)")
print("=" * 100)
best_model = results[0]
print(f"文件名: {best_model['file_name']}")
print(f"F1分数: {best_model['f1_score']:.6f}")
print(f"TP={best_model['TP']}, FP={best_model['FP']}, FN={best_model['FN']}, TN={best_model['TN']}")
print(f"精确率: {best_model['precision']:.4f}, 召回率: {best_model['recall']:.4f}")
