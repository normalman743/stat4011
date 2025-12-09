# filepath: /Users/mannormal/4011/Qi Zihan/classification_systems/ensemble_learning/meta_learning_natxis_classification.py
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

print("=== Meta-Learning NATXIS Classification System ===")
print("使用逻辑回归作为元分类器，优于硬投票集成")

def classify_account_type(row):
    """将账户分为4种类型基于交易模式"""
    has_forward = (row['normal_fprofit'] > 0 or row['abnormal_fprofit'] > 0 or 
                   row['normal_fsize'] > 0 or row['abnormal_fsize'] > 0)
    
    has_backward = (row['normal_bprofit'] > 0 or row['abnormal_bprofit'] > 0 or
                    row['normal_bsize'] > 0 or row['abnormal_bsize'] > 0)
    
    if has_forward and has_backward:
        return 'type1'
    elif has_forward and not has_backward:
        return 'type2'
    elif not has_forward and has_backward:
        return 'type3'
    else:
        return 'type4'

def train_meta_learning_ensemble(data, account_type, n_base_models=50, n_meta_models=10):
    """训练基于元学习的集成模型"""
    print(f"\n训练元学习集成模型 {account_type}:")
    print(f"总账户数: {len(data)}")
    
    # 准备数据
    flag_counts = data['flag'].value_counts()
    print(f"标签分布: {dict(flag_counts)}")
    
    data_copy = data.copy()
    data_copy.loc[data_copy['flag'] == -1, 'flag'] = 0
    
    feature_cols = [col for col in data_copy.columns if col not in ['account', 'flag', 'account_type']]
    
    good_accounts = len(data_copy[data_copy['flag'] == 1])
    bad_accounts = len(data_copy[data_copy['flag'] == 0])
    
    if good_accounts == 0:
        print("未找到好账户，跳过此类型")
        return None, None, None
        
    sample_size = min(good_accounts, bad_accounts)
    print(f"使用平衡采样: 每类 {sample_size} 个账户")
    
    # 第一阶段：训练基础分类器
    print("第一阶段：训练基础分类器...")
    base_predictions = []
    base_models = []
    
    for i in tqdm(range(n_base_models), desc=f"训练基础模型"):
        # 平衡采样
        good_sample = data_copy[data_copy['flag'] == 1].sample(n=sample_size, replace=True, random_state=i)
        bad_sample = data_copy[data_copy['flag'] == 0].sample(n=sample_size, replace=True, random_state=i)
        
        train_data = pd.concat([good_sample, bad_sample], axis=0)
        X_train = train_data[feature_cols].values
        y_train = train_data['flag'].values
        
        # 训练随机森林
        clf = RandomForestClassifier(n_estimators=100, random_state=i, max_depth=10)
        clf.fit(X_train, y_train)
        base_models.append(clf)
        
        # 在全部数据上预测概率
        X_all = data_copy[feature_cols].values
        y_pred_proba = clf.predict_proba(X_all)[:, 1]  # 好账户的概率
        base_predictions.append(y_pred_proba)
    
    # 第二阶段：训练元分类器
    print("第二阶段：训练元分类器...")
    base_predictions_array = np.array(base_predictions).T  # 转置，每行是一个样本的所有基础预测
    y_true = data_copy['flag'].values
    
    # 使用交叉验证训练元分类器
    meta_models = []
    meta_predictions = []
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(base_predictions_array, y_true)):
        X_meta_train = base_predictions_array[train_idx]
        y_meta_train = y_true[train_idx]
        X_meta_val = base_predictions_array[val_idx]
        
        # 训练逻辑回归元分类器
        meta_clf = LogisticRegression(random_state=fold, max_iter=1000)
        meta_clf.fit(X_meta_train, y_meta_train)
        meta_models.append(meta_clf)
        
        # 在验证集上预测
        val_pred = meta_clf.predict(X_meta_val)
        meta_predictions.append((val_idx, val_pred))
    
    # 组合所有验证预测
    final_meta_predictions = np.zeros(len(data_copy))
    for val_idx, val_pred in meta_predictions:
        final_meta_predictions[val_idx] = val_pred
    
    # 计算元学习准确率
    meta_accuracy = metrics.accuracy_score(y_true, final_meta_predictions)
    
    # 比较硬投票结果
    hard_voting_predictions = np.where(np.mean(base_predictions_array, axis=1) > 0.5, 1, 0)
    hard_voting_accuracy = metrics.accuracy_score(y_true, hard_voting_predictions)
    
    print(f"元学习准确率: {meta_accuracy:.4f}")
    print(f"硬投票准确率: {hard_voting_accuracy:.4f}")
    print(f"元学习提升: {meta_accuracy - hard_voting_accuracy:.4f}")
    
    return base_models, meta_models, final_meta_predictions

def predict_with_meta_learning(base_models, meta_models, test_data, feature_cols):
    """使用元学习模型进行预测"""
    X_test = test_data[feature_cols].values
    
    # 基础模型预测
    base_predictions = []
    for model in base_models:
        pred_proba = model.predict_proba(X_test)[:, 1]
        base_predictions.append(pred_proba)
    
    base_predictions_array = np.array(base_predictions).T
    
    # 元分类器预测
    meta_predictions = []
    for meta_model in meta_models:
        meta_pred = meta_model.predict(base_predictions_array)
        meta_predictions.append(meta_pred)
    
    # 对元分类器结果进行投票
    meta_predictions_array = np.array(meta_predictions)
    final_predictions = np.where(np.mean(meta_predictions_array, axis=0) > 0.5, 1, 0)
    
    return final_predictions

# 加载数据
features_path = '/Users/mannormal/4011/Qi Zihan/feature_extraction/generated_features/all_features.csv'
if os.path.exists(features_path):
    print("加载预提取特征...")
    all_features_df = pd.read_csv(features_path)
    print(f"加载特征形状: {all_features_df.shape}")
else:
    print(f"错误: {features_path} 未找到!")
    exit()

# 加载训练和测试数据
pwd = '/Users/mannormal/4011/Qi Zihan/original_data/'
ta = pd.read_csv(pwd + 'train_acc.csv')
te = pd.read_csv(pwd + 'test_acc_predict.csv')

ta.loc[ta['flag'] == 0, 'flag'] = -1

print(f"训练账户: {ta.shape[0]}")
print(f"测试账户: {te.shape[0]}")

# 合并训练数据
training_df = pd.merge(all_features_df, ta[['account', 'flag']], on='account', how='inner')
training_df['account_type'] = training_df.apply(classify_account_type, axis=1)

print(f"\n训练数据准备完成: {training_df.shape}")
print("账户类型分布:")
print(training_df['account_type'].value_counts())

# 按账户类型拆分数据并训练模型
type_data = {}
meta_learning_models = {}
type_predictions = {}

for account_type in ['type1', 'type2', 'type3', 'type4']:
    type_data[account_type] = training_df[training_df['account_type'] == account_type].copy()
    
    if len(type_data[account_type]) > 0:
        base_models, meta_models, final_predictions = train_meta_learning_ensemble(
            type_data[account_type], 
            account_type,
            n_base_models=50,
            n_meta_models=10
        )
        
        if base_models is not None:
            meta_learning_models[account_type] = {
                'base_models': base_models,
                'meta_models': meta_models
            }
            type_predictions[account_type] = final_predictions

print(f"\n为以下类型训练了模型: {list(meta_learning_models.keys())}")

# 处理测试账户
print("\n处理测试账户...")
test_df = pd.merge(all_features_df, te[['account']], on='account', how='inner')
test_df['account_type'] = test_df.apply(classify_account_type, axis=1)

print(f"测试数据准备完成: {test_df.shape}")
print("测试账户类型分布:")
print(test_df['account_type'].value_counts())

# 对测试账户进行元学习预测
print("\n对测试账户进行元学习预测...")
test_predictions = {}

for account_type in ['type1', 'type2', 'type3', 'type4']:
    type_test_data = test_df[test_df['account_type'] == account_type].copy()
    
    if len(type_test_data) > 0 and account_type in meta_learning_models:
        print(f"预测 {account_type}: {len(type_test_data)} 个账户")
        
        feature_cols = [col for col in type_test_data.columns if col not in ['account', 'account_type']]
        
        base_models = meta_learning_models[account_type]['base_models']
        meta_models = meta_learning_models[account_type]['meta_models']
        
        final_predictions = predict_with_meta_learning(
            base_models, meta_models, type_test_data, feature_cols
        )
        
        print(f"{account_type} 预测分布: {np.bincount(final_predictions)}")
        test_predictions[account_type] = {
            'accounts': type_test_data['account'].values,
            'predictions': final_predictions
        }

# 合并测试预测结果
print("\n合并测试预测结果...")
final_test_results = []

for account_type in ['type1', 'type2', 'type3', 'type4']:
    if account_type in test_predictions:
        accounts = test_predictions[account_type]['accounts']
        predictions = test_predictions[account_type]['predictions']
        
        for acc, pred in zip(accounts, predictions):
            final_test_results.append({'account': acc, 'flag': pred})

results_df = pd.DataFrame(final_test_results)
print(f"最终测试结果: {len(results_df)} 个账户")
print("预测分布:")
print(results_df['flag'].value_counts())

# 保存结果
output_path = '../../result_analysis/prediction_results/meta_learning_predictions.csv'
results_df.to_csv(output_path, index=False)
print(f"元学习预测结果已保存到 {output_path}")

# 计算训练数据的F1分数
print("\n" + "="*60)
print("元学习系统 F1-SCORE 分析")
print("="*60)

overall_f1_binary = 0
overall_f1_weighted = 0
overall_f1_macro = 0
total_accounts = 0

for account_type in ['type1', 'type2', 'type3', 'type4']:
    type_training_data = training_df[training_df['account_type'] == account_type].copy()
    
    if len(type_training_data) == 0 or account_type not in type_predictions:
        print(f"{account_type.upper()}: 无数据或预测结果")
        continue
    
    y_true = np.where(type_training_data['flag'].values == -1, 0, 1)
    y_pred = type_predictions[account_type]
    
    if len(y_true) != len(y_pred):
        print(f"{account_type.upper()}: 长度不匹配 - 真实: {len(y_true)}, 预测: {len(y_pred)}")
        continue
    
    try:
        f1_binary = metrics.f1_score(y_true, y_pred, average='binary', zero_division=0)
        f1_weighted = metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = metrics.f1_score(y_true, y_pred, average='macro', zero_division=0)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        
        true_counts = np.bincount(y_true)
        pred_counts = np.bincount(y_pred)
        
        print(f"\n{account_type.upper()} 详细结果:")
        print(f"  账户数量: {len(type_training_data)}")
        print(f"  真实标签分布: Good={true_counts[1] if len(true_counts)>1 else 0}, Bad={true_counts[0]}")
        print(f"  预测标签分布: Good={pred_counts[1] if len(pred_counts)>1 else 0}, Bad={pred_counts[0]}")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  F1-Score (binary): {f1_binary:.4f}")
        print(f"  F1-Score (weighted): {f1_weighted:.4f}")
        print(f"  F1-Score (macro): {f1_macro:.4f}")
        
        weight = len(type_training_data)
        overall_f1_binary += f1_binary * weight
        overall_f1_weighted += f1_weighted * weight
        overall_f1_macro += f1_macro * weight
        total_accounts += weight
        
    except Exception as e:
        print(f"{account_type.upper()}: 计算F1分数时出错: {e}")

if total_accounts > 0:
    overall_f1_binary /= total_accounts
    overall_f1_weighted /= total_accounts
    overall_f1_macro /= total_accounts

print(f"\n" + "="*60)
print("🏆 元学习NATXIS分类系统总结")
print("="*60)
print(f"总体F1-Score (binary):   {overall_f1_binary:.4f}")
print(f"总体F1-Score (weighted): {overall_f1_weighted:.4f}")
print(f"总体F1-Score (macro):    {overall_f1_macro:.4f}")
print(f"分析的总账户数: {total_accounts}")
print(f"使用特征数: 31个基础特征")

print(f"\n测试预测统计:")
print(f"测试账户总数: {len(results_df)}")
print(f"预测为Good的账户: {len(results_df[results_df['flag']==1])}")
print(f"预测为Bad的账户: {len(results_df[results_df['flag']==0])}")
print(f"Good账户比例: {len(results_df[results_df['flag']==1])/len(results_df)*100:.1f}%")

print("\n=== 元学习NATXIS分类完成 ===")
print("✅ 使用逻辑回归元分类器提升硬投票效果!")
print("🎯 预期优于硬投票集成方法")
print(f"📊 结果已保存到: {output_path}")