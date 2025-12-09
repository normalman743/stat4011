import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

print("=== 使用已提取特征的 NATXIS BASELINE 系统 ===")

# 1. 加载原始标签数据
ta = pd.read_csv('../../original_data/train_acc.csv')
te = pd.read_csv('../../original_data/test_acc_predict.csv')

# 2. 加载已提取的特征 (使用增强版44特征)
features_df = pd.read_csv('../../feature_extraction/generated_features/all_features_with_categories.csv')

print(f"训练账户数: {len(ta)}")
print(f"测试账户数: {len(te)}")
print(f"特征数据: {features_df.shape}")
print(f"特征列数: {len(features_df.columns)}")

# 3. 数据预处理
ta.loc[ta['flag'] == 0, 'flag'] = -1  # 0标签转换为-1

# 4. 准备训练数据
train_accounts = set(ta['account'].tolist())
test_accounts = set(te['account'].tolist())

# 筛选训练数据
train_features = features_df[features_df['account'].isin(train_accounts)].copy()
train_features = train_features.merge(ta[['account', 'flag']], on='account', how='inner')

print(f"有效训练数据: {len(train_features)} 个账户")
print(f"标签分布: {train_features['flag'].value_counts().to_dict()}")

# 5. 选择数值特征 (排除分类特征和account列)
categorical_cols = ['account', 'flag', 'traditional_category', 'volume_category', 
                   'profit_category', 'interaction_category', 'behavior_category']
feature_cols = [col for col in train_features.columns if col not in categorical_cols]

print(f"使用特征数量: {len(feature_cols)}")
print("特征列表:", feature_cols[:10], "...")  # 显示前10个特征

# 6. 准备特征矩阵
X = train_features[feature_cols].fillna(0)  # 填充缺失值
y = train_features['flag']

# 7. 特征标准化 (可选)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8. 划分训练验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")

# 9. 训练模型
print("训练RandomForest模型...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 10. 验证集评估
y_val_pred = model.predict(X_val)
y_val_binary = np.where(y_val == -1, 0, 1)
y_pred_binary = np.where(y_val_pred == -1, 0, 1)

# 计算指标
accuracy = accuracy_score(y_val, y_val_pred)
f1_binary = f1_score(y_val_binary, y_pred_binary, average='binary')
f1_weighted = f1_score(y_val_binary, y_pred_binary, average='weighted')
f1_macro = f1_score(y_val_binary, y_pred_binary, average='macro')

print("\n" + "="*60)
print("改进的 NATXIS BASELINE 系统结果")
print("="*60)
print(f"验证集准确率: {accuracy:.4f}")
print(f"F1-Score (binary):   {f1_binary:.4f}")
print(f"F1-Score (weighted): {f1_weighted:.4f}")
print(f"F1-Score (macro):    {f1_macro:.4f}")

# 11. 交叉验证
print("\n进行5折交叉验证...")
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='f1_weighted', n_jobs=-1)
print(f"交叉验证F1分数: {cv_scores}")
print(f"平均CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 12. 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n前10个重要特征:")
print(feature_importance.head(10).to_string(index=False))

# 13. 详细分类报告
print("\n分类报告:")
print(classification_report(y_val, y_val_pred))

# 14. 测试集预测
print("\n预测测试集...")
test_features = features_df[features_df['account'].isin(test_accounts)].copy()
X_test = test_features[feature_cols].fillna(0)
X_test_scaled = scaler.transform(X_test)
y_test_pred = model.predict(X_test_scaled)

# 保存预测结果
test_predictions = pd.DataFrame({
    'account': test_features['account'],
    'Predict': y_test_pred
})

output_path = '../../result_analysis/prediction_results/natxis_baseline_improved_predictions.csv'
test_predictions.to_csv(output_path, index=False)

print(f"\n🏆 改进后的系统总结:")
print(f"真实验证F1分数: {f1_binary:.4f}")
print(f"交叉验证平均F1: {cv_scores.mean():.4f}")
print(f"测试预测已保存到: {output_path}")
print(f"测试预测分布: {pd.Series(y_test_pred).value_counts().to_dict()}")

print("\n=== NATXIS Baseline 改进版完成 ===")