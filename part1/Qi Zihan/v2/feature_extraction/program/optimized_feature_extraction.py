import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configuration
BATCH_SIZE = 100000
N_CORES = 4
CACHE_DIR = '/Users/mannormal/4011/Qi Zihan/v2/feature_extraction/cache'
RESULT_DIR = '/Users/mannormal/4011/Qi Zihan/v2/feature_extraction/result'

# Create directories if they don't exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def extract_features_for_account(account_info):
    """Extract features for a single account"""
    account, from_data, to_data = account_info
    
    feature_dict = {'account': account}
    
    # Basic transaction counts
    feature_dict['outgoing_count'] = len(from_data)
    feature_dict['incoming_count'] = len(to_data)
    feature_dict['total_transactions'] = len(from_data) + len(to_data)
    
    # Value statistics for outgoing transactions
    if len(from_data) > 0:
        values = from_data['value'].astype(float).values
        feature_dict['outgoing_total_value'] = float(np.sum(values))
        feature_dict['outgoing_avg_value'] = float(np.mean(values))
        feature_dict['outgoing_max_value'] = float(np.max(values))
        feature_dict['outgoing_min_value'] = float(np.min(values))
        feature_dict['outgoing_std_value'] = float(np.std(values)) if len(values) > 1 else 0.0
    else:
        feature_dict['outgoing_total_value'] = 0.0
        feature_dict['outgoing_avg_value'] = 0.0
        feature_dict['outgoing_max_value'] = 0.0
        feature_dict['outgoing_min_value'] = 0.0
        feature_dict['outgoing_std_value'] = 0.0
    
    # Value statistics for incoming transactions
    if len(to_data) > 0:
        values = to_data['value'].astype(float).values
        feature_dict['incoming_total_value'] = float(np.sum(values))
        feature_dict['incoming_avg_value'] = float(np.mean(values))
        feature_dict['incoming_max_value'] = float(np.max(values))
        feature_dict['incoming_min_value'] = float(np.min(values))
        feature_dict['incoming_std_value'] = float(np.std(values)) if len(values) > 1 else 0.0
    else:
        feature_dict['incoming_total_value'] = 0.0
        feature_dict['incoming_avg_value'] = 0.0
        feature_dict['incoming_max_value'] = 0.0
        feature_dict['incoming_min_value'] = 0.0
        feature_dict['incoming_std_value'] = 0.0
    
    # Gas statistics for outgoing transactions
    if len(from_data) > 0:
        gas_values = from_data['gas'].astype(float).values
        gas_price_values = from_data['gas_price'].astype(float).values
        feature_dict['outgoing_total_gas'] = float(np.sum(gas_values))
        feature_dict['outgoing_avg_gas'] = float(np.mean(gas_values))
        feature_dict['outgoing_avg_gas_price'] = float(np.mean(gas_price_values))
    else:
        feature_dict['outgoing_total_gas'] = 0.0
        feature_dict['outgoing_avg_gas'] = 0.0
        feature_dict['outgoing_avg_gas_price'] = 0.0
    
    # Ratio features - 确保类型安全的除法
    outgoing_count = max(float(feature_dict['outgoing_count']), 1.0)
    outgoing_total_value = max(float(feature_dict['outgoing_total_value']), 1.0)
    
    feature_dict['in_out_ratio'] = float(feature_dict['incoming_count']) / outgoing_count
    feature_dict['value_ratio'] = float(feature_dict['incoming_total_value']) / outgoing_total_value
    
    # Unique counterparties
    feature_dict['unique_recipients'] = len(set(from_data['to_account'])) if len(from_data) > 0 else 0
    feature_dict['unique_senders'] = len(set(to_data['from_account'])) if len(to_data) > 0 else 0
    
    return feature_dict

def process_batch(batch_data):
    """Process a batch of accounts"""
    return [extract_features_for_account(account_info) for account_info in batch_data]

if __name__ == '__main__':
    print("Loading data...")

    # Load all datasets
    train_labels = pd.read_csv('/Users/mannormal/4011/Qi Zihan/original_data/train_acc.csv')
    test_accounts = pd.read_csv('/Users/mannormal/4011/Qi Zihan/original_data/test_acc_predict.csv')

    print(f"Train labels: {len(train_labels)} accounts")
    print(f"Test accounts: {len(test_accounts)} accounts")

    # Get all accounts we need features for
    all_accounts = list(set(train_labels['account'].tolist() + test_accounts['account'].tolist()))
    print(f"Total accounts to process: {len(all_accounts)}")

    # Check if cached features exist
    cache_file = os.path.join(CACHE_DIR, 'features_cache.pkl')
    if os.path.exists(cache_file):
        print("Loading cached features...")
        with open(cache_file, 'rb') as f:
            features_df = pickle.load(f)
        print("Cached features loaded successfully!")
    else:
        print("Loading transaction data...")
        transactions = pd.read_csv('/Users/mannormal/4011/Qi Zihan/original_data/transactions.csv')
        print(f"Transactions: {len(transactions)} records")
        
        # 确保数值列的数据类型正确
        numeric_columns = ['value', 'gas', 'gas_price']
        for col in numeric_columns:
            if col in transactions.columns:
                transactions[col] = pd.to_numeric(transactions[col], errors='coerce').fillna(0)
        
        print("Building indexes for faster lookup...")
        # Create indexes for faster lookup
        from_index = transactions.groupby('from_account')
        to_index = transactions.groupby('to_account')
        
        print("Preparing account data for parallel processing...")
        # Prepare data for each account
        account_data = []
        for account in tqdm(all_accounts, desc="Preparing data"):
            from_data = from_index.get_group(account) if account in from_index.groups else pd.DataFrame()
            to_data = to_index.get_group(account) if account in to_index.groups else pd.DataFrame()
            account_data.append((account, from_data, to_data))
        
        print("Extracting features with parallel processing...")
        
        # Process in batches with multiprocessing
        all_features = []
        
        # Split into batches
        batches = []
        for i in range(0, len(account_data), BATCH_SIZE):
            batch = account_data[i:i + BATCH_SIZE]
            batches.append(batch)
        
        print(f"Processing {len(batches)} batches with {N_CORES} cores...")
        
        # Process each batch
        for i, batch in enumerate(tqdm(batches, desc="Processing batches")):
            print(f"Processing batch {i+1}/{len(batches)} with {len(batch)} accounts")
            
            # Use multiprocessing for this batch
            with Pool(processes=N_CORES) as pool:
                # Split batch into smaller chunks for each process
                chunk_size = max(1, len(batch) // N_CORES)
                chunks = [batch[j:j + chunk_size] for j in range(0, len(batch), chunk_size)]
                
                # Process chunks in parallel
                batch_results = pool.map(process_batch, chunks)
                
                # Flatten results
                for chunk_result in batch_results:
                    all_features.extend(chunk_result)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Fill any remaining NaN values with 0
        features_df = features_df.fillna(0)
        
        print("Features extracted successfully!")
        print(f"Feature matrix shape: {features_df.shape}")
        
        # Cache the features
        print("Caching features...")
        with open(cache_file, 'wb') as f:
            pickle.dump(features_df, f)
        print("Features cached successfully!")

    print(f"Features: {[col for col in features_df.columns if col != 'account']}")

    # Prepare training data
    train_features = features_df[features_df['account'].isin(train_labels['account'])]
    train_features = train_features.merge(train_labels, on='account', how='left')

    # Prepare test data
    test_features = features_df[features_df['account'].isin(test_accounts['account'])]

    print(f"Training data shape: {train_features.shape}")
    print(f"Test data shape: {test_features.shape}")

    # Separate features and labels
    X_train = train_features.drop(['account', 'flag'], axis=1)
    y_train = train_features['flag']
    X_test = test_features.drop(['account'], axis=1)

    print(f"Label distribution: {y_train.value_counts().to_dict()}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training model with cross-validation...")

    # Train Random Forest with cross-validation
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=N_CORES
    )

    # Perform cross-validation
    cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='f1')
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train final model
    rf_model.fit(X_train_scaled, y_train)

    # Make predictions on training set to check F1
    train_pred = rf_model.predict(X_train_scaled)
    train_f1 = f1_score(y_train, train_pred)
    print(f"Training F1 score: {train_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_train, train_pred))

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Feature Importances:")
    print(feature_importance.head(10))

    # Make predictions on test set
    test_pred = rf_model.predict(X_test_scaled)

    # Create submission file
    submission = pd.DataFrame({
        'account': test_features['account'],
        'Predict': test_pred
    })

    # Save results to the correct directory
    result_file = os.path.join(RESULT_DIR, 'test_predictions.csv')
    submission.to_csv(result_file, index=False)
    print(f"\nPredictions saved to {result_file}")
    print(f"Test predictions distribution: {pd.Series(test_pred).value_counts().to_dict()}")

    # Save feature importance
    importance_file = os.path.join(RESULT_DIR, 'feature_importance.csv')
    feature_importance.to_csv(importance_file, index=False)
    print(f"Feature importance saved to {importance_file}")

    # Save model
    model_file = os.path.join(RESULT_DIR, 'trained_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump({'model': rf_model, 'scaler': scaler}, f)
    print(f"Model saved to {model_file}")

    print("\nModel training completed successfully!")