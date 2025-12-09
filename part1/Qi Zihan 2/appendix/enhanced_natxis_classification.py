import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

print("=== Enhanced NATXIS Classification System ===")
print("Using ensemble learning with 100 models per account type")

def classify_account_type(row):
    """Classify accounts into 4 types based on transaction patterns"""
    # Check if forward transactions exist (has outgoing transactions ending with 'a' accounts)
    has_forward = (row['normal_fprofit'] > 0 or row['abnormal_fprofit'] > 0 or 
                   row['normal_fsize'] > 0 or row['abnormal_fsize'] > 0)
    
    # Check if backward transactions exist (has incoming transactions from 'a' accounts) 
    has_backward = (row['normal_bprofit'] > 0 or row['abnormal_bprofit'] > 0 or
                    row['normal_bsize'] > 0 or row['abnormal_bsize'] > 0)
    
    if has_forward and has_backward:
        return 'type1'  # Both directions
    elif has_forward and not has_backward:
        return 'type2'  # Forward only
    elif not has_forward and has_backward:
        return 'type3'  # Backward only
    else:
        return 'type4'  # Neither direction

def train_ensemble_model(data, account_type, n_estimators=100, n_models=100):
    """Train ensemble of RandomForest models with balanced sampling"""
    print(f"\nTraining ensemble for {account_type}:")
    print(f"Total accounts: {len(data)}")
    
    # Prepare data
    flag_counts = data['flag'].value_counts()
    print(f"Flag distribution: {dict(flag_counts)}")
    
    # Convert -1 flags to 0 for binary classification
    data_copy = data.copy()
    data_copy.loc[data_copy['flag'] == -1, 'flag'] = 0
    
    # Get feature columns (exclude account, flag, and account_type)
    feature_cols = [col for col in data_copy.columns if col not in ['account', 'flag', 'account_type']]
    
    # Count good and bad accounts
    good_accounts = len(data_copy[data_copy['flag'] == 1])
    bad_accounts = len(data_copy[data_copy['flag'] == 0])
    
    if good_accounts == 0:
        print("No good accounts found, skipping this type")
        return None, None
        
    # Use minimum count for balanced sampling
    sample_size = min(good_accounts, bad_accounts)
    print(f"Using balanced sampling: {sample_size} accounts per class")
    
    # Train ensemble of models
    predictions = []
    
    for i in tqdm(range(n_models), desc=f"Training {account_type} models"):
        # Balanced sampling
        good_sample = data_copy[data_copy['flag'] == 1].sample(n=sample_size, replace=True)
        bad_sample = data_copy[data_copy['flag'] == 0].sample(n=sample_size, replace=True)
        
        # Combine samples
        train_data = pd.concat([good_sample, bad_sample], axis=0)
        
        # Prepare training data
        X_train = train_data[feature_cols].values
        y_train = train_data['flag'].values
        
        # Train RandomForest
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=i)
        clf.fit(X_train, y_train)
        
        # Predict on all data
        X_all = data_copy[feature_cols].values
        y_pred = clf.predict(X_all)
        predictions.append(y_pred)
    
    # Ensemble voting (>93 out of 100 models vote for good account)
    predictions_array = np.array(predictions)
    ensemble_votes = np.sum(predictions_array, axis=0)
    final_predictions = np.where(ensemble_votes > 93, 1, 0)
    
    # Calculate accuracy
    y_true = data_copy['flag'].values
    accuracy = metrics.accuracy_score(y_true, final_predictions)
    print(f"Ensemble accuracy: {accuracy:.4f}")
    
    return predictions_array, final_predictions

# Load data
if os.path.exists('all_features.csv'):
    print("Loading pre-extracted features...")
    all_features_df = pd.read_csv('all_features.csv')
    print(f"Loaded features shape: {all_features_df.shape}")
else:
    print("Error: all_features.csv not found!")
    exit()

# Load training and testing data
pwd = '/Users/mannormal/4011/Qi Zihan/'
ta = pd.read_csv(pwd + 'train_acc.csv')
te = pd.read_csv(pwd + 'test_acc_predict.csv')

# Replace zero flag by -1 flag in training data
ta.loc[ta['flag'] == 0, 'flag'] = -1

print(f"Training accounts: {ta.shape[0]}")
print(f"Test accounts: {te.shape[0]}")

# Merge with training data to get flags
training_df = pd.merge(
    all_features_df, 
    ta[['account', 'flag']], 
    on='account', 
    how='inner'
)

# Classify accounts into 4 types
training_df['account_type'] = training_df.apply(classify_account_type, axis=1)

print(f"\nTraining data ready: {training_df.shape}")
print("Account type distribution:")
print(training_df['account_type'].value_counts())

# Split data by account type
type_data = {}
ensemble_models = {}
type_predictions = {}

for account_type in ['type1', 'type2', 'type3', 'type4']:
    type_data[account_type] = training_df[training_df['account_type'] == account_type].copy()
    
    if len(type_data[account_type]) > 0:
        # Train ensemble model
        predictions_array, final_predictions = train_ensemble_model(
            type_data[account_type], 
            account_type,
            n_estimators=100,
            n_models=100
        )
        
        if predictions_array is not None:
            ensemble_models[account_type] = predictions_array
            type_predictions[account_type] = final_predictions

print(f"\nModels trained for types: {list(ensemble_models.keys())}")

# Now process test accounts
print("\nProcessing test accounts...")
test_df = pd.merge(
    all_features_df, 
    te[['account']], 
    on='account', 
    how='inner'
)

test_df['account_type'] = test_df.apply(classify_account_type, axis=1)
print(f"Test data ready: {test_df.shape}")
print("Test account type distribution:")
print(test_df['account_type'].value_counts())

# Make predictions for test accounts
print("\nMaking ensemble predictions for test accounts...")
test_predictions = {}

for account_type in ['type1', 'type2', 'type3', 'type4']:
    type_test_data = test_df[test_df['account_type'] == account_type].copy()
    
    if len(type_test_data) > 0 and account_type in ensemble_models:
        print(f"Predicting for {account_type}: {len(type_test_data)} accounts")
        
        # Get feature columns
        feature_cols = [col for col in type_test_data.columns if col not in ['account', 'account_type']]
        X_test = type_test_data[feature_cols].values
        
        # Apply each model in the ensemble
        predictions = []
        for i in range(100):  # 100 models per type
            # We need to retrain the models on test data or save them
            # For now, let's use a simplified approach
            
            # Sample from training data
            train_type_data = type_data[account_type].copy()
            train_type_data.loc[train_type_data['flag'] == -1, 'flag'] = 0
            
            good_accounts = len(train_type_data[train_type_data['flag'] == 1])
            bad_accounts = len(train_type_data[train_type_data['flag'] == 0])
            sample_size = min(good_accounts, bad_accounts)
            
            if sample_size > 0:
                good_sample = train_type_data[train_type_data['flag'] == 1].sample(n=sample_size, replace=True)
                bad_sample = train_type_data[train_type_data['flag'] == 0].sample(n=sample_size, replace=True)
                train_data = pd.concat([good_sample, bad_sample], axis=0)
                
                train_feature_cols = [col for col in train_data.columns if col not in ['account', 'flag', 'account_type']]
                X_train = train_data[train_feature_cols].values
                y_train = train_data['flag'].values
                
                clf = RandomForestClassifier(n_estimators=100, random_state=i)
                clf.fit(X_train, y_train)
                
                y_pred = clf.predict(X_test)
                predictions.append(y_pred)
        
        if predictions:
            # Ensemble voting
            predictions_array = np.array(predictions)
            ensemble_votes = np.sum(predictions_array, axis=0)
            final_predictions = np.where(ensemble_votes > 93, 1, 0)
            
            test_predictions[account_type] = {
                'accounts': type_test_data['account'].values,
                'predictions': final_predictions
            }

# Combine all test predictions
print("\nCombining test predictions...")
final_test_results = []

for account_type in ['type1', 'type2', 'type3', 'type4']:
    if account_type in test_predictions:
        accounts = test_predictions[account_type]['accounts']
        predictions = test_predictions[account_type]['predictions']
        
        for acc, pred in zip(accounts, predictions):
            final_test_results.append({'account': acc, 'flag': pred})

# Create results DataFrame
results_df = pd.DataFrame(final_test_results)
print(f"Final test results: {len(results_df)} accounts")
print("Prediction distribution:")
print(results_df['flag'].value_counts())

# Save results
results_df.to_csv('enhanced_test_predictions.csv', index=False)
print("Enhanced predictions saved to enhanced_test_predictions.csv")

print("\n=== Enhanced NATXIS Classification Complete ===")
print("Using ensemble learning with balanced sampling should significantly improve accuracy!")