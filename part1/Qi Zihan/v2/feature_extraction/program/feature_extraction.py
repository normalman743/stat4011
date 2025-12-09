import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")

# Load all datasets
train_labels = pd.read_csv('/Users/mannormal/4011/Qi Zihan/original_data/train_acc.csv')
test_accounts = pd.read_csv('/Users/mannormal/4011/Qi Zihan/original_data/test_acc_predict.csv')
transactions = pd.read_csv('/Users/mannormal/4011/Qi Zihan/original_data/transactions.csv')

print(f"Train labels: {len(train_labels)} accounts")
print(f"Test accounts: {len(test_accounts)} accounts") 
print(f"Transactions: {len(transactions)} records")

# Get all accounts we need features for
all_accounts = set(train_labels['account'].tolist() + test_accounts['account'].tolist())
print(f"Total accounts to process: {len(all_accounts)}")

print("Extracting features...")

# Extract simple features for each account
features = []

for i, account in enumerate(all_accounts):
    if i % 1000 == 0:
        print(f"Processing account {i+1}/{len(all_accounts)}")
    
    # Transactions where this account is sender
    outgoing = transactions[transactions['from_account'] == account]
    # Transactions where this account is receiver  
    incoming = transactions[transactions['to_account'] == account]
    
    feature_dict = {'account': account}
    
    # Basic transaction counts
    feature_dict['outgoing_count'] = len(outgoing)
    feature_dict['incoming_count'] = len(incoming)
    feature_dict['total_transactions'] = len(outgoing) + len(incoming)
    
    # Value statistics for outgoing transactions
    if len(outgoing) > 0:
        feature_dict['outgoing_total_value'] = outgoing['value'].sum()
        feature_dict['outgoing_avg_value'] = outgoing['value'].mean()
        feature_dict['outgoing_max_value'] = outgoing['value'].max()
        feature_dict['outgoing_min_value'] = outgoing['value'].min()
        feature_dict['outgoing_std_value'] = outgoing['value'].std()
    else:
        feature_dict['outgoing_total_value'] = 0
        feature_dict['outgoing_avg_value'] = 0
        feature_dict['outgoing_max_value'] = 0
        feature_dict['outgoing_min_value'] = 0
        feature_dict['outgoing_std_value'] = 0
    
    # Value statistics for incoming transactions
    if len(incoming) > 0:
        feature_dict['incoming_total_value'] = incoming['value'].sum()
        feature_dict['incoming_avg_value'] = incoming['value'].mean()
        feature_dict['incoming_max_value'] = incoming['value'].max()
        feature_dict['incoming_min_value'] = incoming['value'].min()
        feature_dict['incoming_std_value'] = incoming['value'].std()
    else:
        feature_dict['incoming_total_value'] = 0
        feature_dict['incoming_avg_value'] = 0
        feature_dict['incoming_max_value'] = 0
        feature_dict['incoming_min_value'] = 0
        feature_dict['incoming_std_value'] = 0
    
    # Gas statistics for outgoing transactions
    if len(outgoing) > 0:
        feature_dict['outgoing_total_gas'] = outgoing['gas'].sum()
        feature_dict['outgoing_avg_gas'] = outgoing['gas'].mean()
        feature_dict['outgoing_avg_gas_price'] = outgoing['gas_price'].mean()
    else:
        feature_dict['outgoing_total_gas'] = 0
        feature_dict['outgoing_avg_gas'] = 0
        feature_dict['outgoing_avg_gas_price'] = 0
    
    # Ratio features
    feature_dict['in_out_ratio'] = feature_dict['incoming_count'] / max(feature_dict['outgoing_count'], 1)
    feature_dict['value_ratio'] = feature_dict['incoming_total_value'] / max(feature_dict['outgoing_total_value'], 1)
    
    # Unique counterparties
    feature_dict['unique_recipients'] = len(outgoing['to_account'].unique()) if len(outgoing) > 0 else 0
    feature_dict['unique_senders'] = len(incoming['from_account'].unique()) if len(incoming) > 0 else 0
    
    features.append(feature_dict)

# Convert to DataFrame
features_df = pd.DataFrame(features)

# Fill any remaining NaN values with 0
features_df = features_df.fillna(0)

print("Features extracted successfully!")
print(f"Feature matrix shape: {features_df.shape}")
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
    class_weight='balanced'
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

# Save results
submission.to_csv('/Users/mannormal/4011/test_predictions.csv', index=False)
print(f"\nPredictions saved to test_predictions.csv")
print(f"Test predictions distribution: {pd.Series(test_pred).value_counts().to_dict()}")

print("\nModel training completed successfully!")