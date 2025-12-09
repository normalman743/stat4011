import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

print("=== Decision Tree Training by Account Type with K-Fold CV ===")

# Load data
features_path = '/Users/mannormal/4011/Qi Zihan/feature_extraction/generated_features/all_features.csv'
all_features_df = pd.read_csv(features_path)

pwd = '/Users/mannormal/4011/Qi Zihan/original_data/'
ta = pd.read_csv(pwd + 'train_acc.csv')
te = pd.read_csv(pwd + 'test_acc_predict.csv')
ta.loc[ta['flag'] == 0, 'flag'] = -1

training_df = pd.merge(all_features_df, ta[['account', 'flag']], on='account', how='inner')

def classify_account_type(row):
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

training_df['account_type'] = training_df.apply(classify_account_type, axis=1)
print("Account type distribution:")
print(training_df['account_type'].value_counts())

feature_cols = [col for col in training_df.columns if col not in ['account', 'flag', 'account_type']]

for account_type in ['type1', 'type2', 'type3', 'type4']:
    type_data = training_df[training_df['account_type'] == account_type].copy()
    if len(type_data) == 0:
        print(f"No data for {account_type}")
        continue
    
    type_data.loc[type_data['flag'] == -1, 'flag'] = 0
    X = type_data[feature_cols].values
    y = type_data['flag'].values
    
    print(f"\n=== Training for {account_type} ({len(type_data)} accounts) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    train_f1_scores = []
    test_f1_scores = []
    fold_num = 1
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = DecisionTreeClassifier(random_state=42, max_depth=10)  # Limited depth to reduce overfit
        clf.fit(X_train, y_train)
        
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        train_f1 = f1_score(y_train, y_train_pred, average='binary', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='binary', zero_division=0)
        
        print(f"Fold {fold_num}: Train F1 = {train_f1:.4f}, Test F1 = {test_f1:.4f}")
        
        train_f1_scores.append(train_f1)
        test_f1_scores.append(test_f1)
        fold_num += 1
    
    mean_train_f1 = np.mean(train_f1_scores)
    mean_test_f1 = np.mean(test_f1_scores)
    print(f"Mean Train F1: {mean_train_f1:.4f}")
    print(f"Mean Test F1: {mean_test_f1:.4f}")

# Test predictions (similar structure, but without writing file)
test_df = pd.merge(all_features_df, te[['account']], on='account', how='inner')
test_df['account_type'] = test_df.apply(classify_account_type, axis=1)

test_predictions = []
for account_type in ['type1', 'type2', 'type3', 'type4']:
    type_test_data = test_df[test_df['account_type'] == account_type].copy()
    if len(type_test_data) == 0:
        continue
    
    X_test = type_test_data[feature_cols].values
    
    # Retrain on full type training data
    type_train_data = training_df[training_df['account_type'] == account_type].copy()
    type_train_data.loc[type_train_data['flag'] == -1, 'flag'] = 0
    X_train = type_train_data[feature_cols].values
    y_train = type_train_data['flag'].values
    
    clf = DecisionTreeClassifier(random_state=42, max_depth=10)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    for acc, pred in zip(type_test_data['account'], y_pred):
        test_predictions.append({'account': acc, 'flag': pred})

results_df = pd.DataFrame(test_predictions)
print("\nTest Prediction Distribution:")
print(results_df['flag'].value_counts())

print("=== Training Complete (No file written) ===")