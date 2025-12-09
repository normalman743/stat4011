import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve
)
from sklearn import metrics

# Import the training accounts
ta = pd.read_csv('../../original_data/train_acc.csv')

# Import the testing accounts
te = pd.read_csv('../../original_data/test_acc_predict.csv')

# Import the transaction details
trans = pd.read_csv('../../original_data/transactions.csv')

print("Training accounts shape:", ta.shape)
print("Testing accounts shape:", te.shape)
print("Transactions shape:", trans.shape)

# Check data
print("\nTraining accounts sample:")
print(ta.head())
print("\nTesting accounts sample:")
print(te.head())
print("\nTransactions sample:")
print(trans.head())

# Calculate profit for transactions
trans['pprofit'] = (trans.value.astype('float') - trans.gas*trans.gas_price)/100000

# An indicator variable to show positive profit
trans['profit'] = np.where(trans['pprofit'] > 0, trans['pprofit'], 0)

print("\nTransactions with profit calculation:")
print(trans.tail())

# Replace zero flag by -1 flag in training data
ta.loc[ta[ta.flag==0].index,'flag'] = -1

# Merge training and testing accounts
tdf = pd.merge(ta, te[['account']], left_on=['account'], right_on=['account'], how='outer')
print("\nMerged accounts shape:", tdf.shape)

# Since the testing account flags are missing, replace them with zero flags
tdf.replace(np.nan, 0, inplace=True)

# Use account as index
tdf.set_index('account', inplace=True)

print("\nMerged dataframe sample:")
print(tdf.head(10))

print("\n=== NATXIS BASELINE CLASSIFICATION SYSTEM ===")

# Feature extraction functions (simplified baseline)
def find_to_nei(acc):
    """Find outgoing neighbors for account"""
    to_neighbors = trans[trans['from_account'] == acc]['to_account'].tolist()
    return list(set(to_neighbors))

def find_from_nei(acc):
    """Find incoming neighbors for account"""  
    from_neighbors = trans[trans['to_account'] == acc]['from_account'].tolist()
    return list(set(from_neighbors))

def extract_baseline_features(account):
    """Extract basic features for baseline system"""
    features = {'account': account}
    
    # Get outgoing transactions
    out_trans = trans[trans['from_account'] == account]
    # Get incoming transactions  
    in_trans = trans[trans['to_account'] == account]
    
    # Basic feature extraction
    features['out_count'] = len(out_trans)
    features['in_count'] = len(in_trans)
    features['total_count'] = features['out_count'] + features['in_count']
    
    features['out_profit'] = out_trans['profit'].sum() if len(out_trans) > 0 else 0
    features['in_profit'] = in_trans['profit'].sum() if len(in_trans) > 0 else 0
    features['total_profit'] = features['out_profit'] + features['in_profit']
    
    features['out_neighbors'] = len(find_to_nei(account))
    features['in_neighbors'] = len(find_from_nei(account))
    features['total_neighbors'] = features['out_neighbors'] + features['in_neighbors']
    
    return features

# Extract features for all accounts
print("Extracting baseline features...")
all_accounts = tdf.index.tolist()
feature_list = []

for account in tqdm(all_accounts[:1000]):  # Limit to first 1000 for baseline demo
    features = extract_baseline_features(account)
    feature_list.append(features)

features_df = pd.DataFrame(feature_list)
features_df.set_index('account', inplace=True)

print(f"Features extracted for {len(features_df)} accounts")
print(f"Feature columns: {features_df.columns.tolist()}")

# Prepare training data
training_accounts = ta['account'].tolist()
training_features = features_df.loc[features_df.index.isin(training_accounts)]
training_labels = ta.set_index('account')['flag']

# Align training data
common_accounts = training_features.index.intersection(training_labels.index)
X_train = training_features.loc[common_accounts]
y_train = training_labels.loc[common_accounts]

print(f"Training data: {len(X_train)} accounts")
print(f"Training label distribution: {y_train.value_counts().to_dict()}")

# Train baseline model
print("Training baseline RandomForest model...")
baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)

# Predict on training data for F1-score calculation
y_train_pred = baseline_model.predict(X_train)

# Convert labels for binary classification metrics
y_true_binary = np.where(y_train.values == -1, 0, 1)
y_pred_binary = np.where(y_train_pred == -1, 0, 1)

# Calculate F1 scores
f1_binary = f1_score(y_true_binary, y_pred_binary, average='binary')
f1_weighted = f1_score(y_true_binary, y_pred_binary, average='weighted')
f1_macro = f1_score(y_true_binary, y_pred_binary, average='macro')
accuracy = accuracy_score(y_true_binary, y_pred_binary)

print("\n" + "="*60)
print("NATXIS BASELINE SYSTEM F1-SCORE RESULTS")
print("="*60)
print(f"Training Accuracy: {accuracy:.4f}")
print(f"F1-Score (binary):   {f1_binary:.4f}")
print(f"F1-Score (weighted): {f1_weighted:.4f}")  
print(f"F1-Score (macro):    {f1_macro:.4f}")
print(f"Training accounts: {len(y_train)}")

# Predict on test data
test_accounts = te['account'].tolist()
test_features = features_df.loc[features_df.index.isin(test_accounts)]
y_test_pred = baseline_model.predict(test_features)

# Save predictions
test_predictions = pd.DataFrame({
    'account': test_features.index,
    'Predict': y_test_pred
})

output_path = '../../result_analysis/prediction_results/natxis_baseline_predictions.csv'
test_predictions.to_csv(output_path, index=False)

print(f"\n🏆 NATXIS BASELINE SYSTEM SUMMARY:")
print(f"Overall F1-Score (binary): {f1_binary:.4f}")
print(f"Test predictions saved to: {output_path}")
print(f"Test prediction distribution: {pd.Series(y_test_pred).value_counts().to_dict()}")

print("\n=== NATXIS Baseline Classification Complete ===")