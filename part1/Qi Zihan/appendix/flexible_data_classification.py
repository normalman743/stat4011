import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== Flexible Data Classification System ===")

# Load the complete features file
print("Loading features...")
if not pd.io.common.file_exists('all_features.csv'):
    print("Error: all_features.csv not found. Please run parallel_feature_extraction.py first.")
    exit()

features_df = pd.read_csv('all_features.csv')
print(f"Total features loaded: {features_df.shape[0]}")

def analyze_feature_patterns(df):
    """Analyze different feature patterns in the data"""
    patterns = {}
    
    # Pattern 1: Forward/Backward transaction analysis
    df['has_forward_cnt'] = (df['normal_fprofit'] > 0) | (df['normal_fsize'] > 0) | \
                           (df['abnormal_fprofit'] > 0) | (df['abnormal_fsize'] > 0) | \
                           (df['bad_fprofit'] > 0) | (df['bad_fsize'] > 0)
    
    df['has_backward_cnt'] = (df['normal_bprofit'] > 0) | (df['normal_bsize'] > 0) | \
                            (df['abnormal_bprofit'] > 0) | (df['abnormal_bsize'] > 0) | \
                            (df['bad_bprofit'] > 0) | (df['bad_bsize'] > 0)
    
    # Pattern 2: Transaction volume analysis
    df['total_forward_transactions'] = df['normal_fsize'] + df['abnormal_fsize'] + df['bad_fsize']
    df['total_backward_transactions'] = df['normal_bsize'] + df['abnormal_bsize'] + df['bad_bsize']
    df['total_transactions'] = df['total_forward_transactions'] + df['total_backward_transactions']
    
    # Pattern 3: Profit analysis
    df['total_forward_profit'] = df['normal_fprofit'] + df['abnormal_fprofit'] + df['bad_fprofit']
    df['total_backward_profit'] = df['normal_bprofit'] + df['abnormal_bprofit'] + df['bad_bprofit']
    df['total_profit'] = df['total_forward_profit'] + df['total_backward_profit']
    
    # Pattern 4: Account type interaction
    df['has_A_forward'] = (df['A_fprofit'] > 0) | (df['A_fsize'] > 0)
    df['has_B_forward'] = (df['B_fprofit'] > 0) | (df['B_fsize'] > 0)
    df['has_A_backward'] = (df['A_bprofit'] > 0) | (df['A_bsize'] > 0)
    df['has_B_backward'] = (df['B_bprofit'] > 0) | (df['B_bsize'] > 0)
    
    return df

def create_advanced_categories(df):
    """Create multiple categorization schemes"""
    categories = {}
    
    # Traditional 4-category system (from PDF)
    def traditional_category(row):
        forward_cnt = row['has_forward_cnt']
        backward_cnt = row['has_backward_cnt']
        
        if forward_cnt and backward_cnt:
            return 'both_directions'
        elif forward_cnt and not backward_cnt:
            return 'forward_only'
        elif not forward_cnt and backward_cnt:
            return 'backward_only'
        else:
            return 'isolated'
    
    # Transaction volume based categories
    def volume_category(row):
        total_txn = row['total_transactions']
        if total_txn == 0:
            return 'no_transactions'
        elif total_txn <= 5:
            return 'low_volume'
        elif total_txn <= 50:
            return 'medium_volume'
        elif total_txn <= 500:
            return 'high_volume'
        else:
            return 'very_high_volume'
    
    # Profit-based categories
    def profit_category(row):
        total_profit = row['total_profit']
        if total_profit <= 0:
            return 'loss_or_zero'
        elif total_profit <= 1:
            return 'low_profit'
        elif total_profit <= 100:
            return 'medium_profit'
        elif total_profit <= 10000:
            return 'high_profit'
        else:
            return 'very_high_profit'
    
    # Account interaction patterns
    def interaction_category(row):
        patterns = []
        if row['has_A_forward']: patterns.append('A_out')
        if row['has_A_backward']: patterns.append('A_in')
        if row['has_B_forward']: patterns.append('B_out')
        if row['has_B_backward']: patterns.append('B_in')
        
        if not patterns:
            return 'no_interaction'
        else:
            return '_'.join(sorted(patterns))
    
    # Behavioral patterns (combining multiple factors)
    def behavior_category(row):
        # High activity accounts
        if row['total_transactions'] > 100 and row['total_profit'] > 100:
            return 'high_activity_profitable'
        elif row['total_transactions'] > 100 and row['total_profit'] <= 0:
            return 'high_activity_unprofitable'
        
        # Medium activity accounts
        elif row['total_transactions'] > 10:
            if row['has_forward_cnt'] and row['has_backward_cnt']:
                return 'medium_activity_bidirectional'
            else:
                return 'medium_activity_unidirectional'
        
        # Low activity accounts
        elif row['total_transactions'] > 0:
            return 'low_activity'
        else:
            return 'inactive'
    
    # Apply all categorizations
    df['traditional_category'] = df.apply(traditional_category, axis=1)
    df['volume_category'] = df.apply(volume_category, axis=1)
    df['profit_category'] = df.apply(profit_category, axis=1)
    df['interaction_category'] = df.apply(interaction_category, axis=1)
    df['behavior_category'] = df.apply(behavior_category, axis=1)
    
    return df

def save_category_data(df, category_name, feature_set=None):
    """Save data for each category within a categorization scheme"""
    if feature_set is None:
        # Default feature set (all features except categorical ones)
        feature_set = [col for col in df.columns if col.startswith(('account', 'normal_', 'abnormal_', 'bad_', 'A_', 'B_'))]
    
    print(f"\n=== {category_name.upper()} CATEGORIZATION ===")
    category_counts = df[category_name].value_counts()
    print(f"Category distribution:")
    print(category_counts)
    
    # Save each category as separate file
    for category_value in category_counts.index:
        category_df = df[df[category_name] == category_value][feature_set].copy()
        filename = f'{category_name}_{category_value}.csv'
        category_df.to_csv(filename, index=False)
        print(f"Saved {len(category_df)} records to {filename}")
    
    # Also save a mapping file
    mapping_df = df[['account', category_name]].copy()
    mapping_df.to_csv(f'{category_name}_mapping.csv', index=False)
    print(f"Saved category mapping to {category_name}_mapping.csv")

# Main processing
print("Analyzing feature patterns...")
features_df = analyze_feature_patterns(features_df)

print("Creating advanced categories...")
features_df = create_advanced_categories(features_df)

# Define feature sets for different purposes
full_feature_set = [col for col in features_df.columns if col.startswith(('account', 'normal_', 'abnormal_', 'bad_', 'A_', 'B_'))]

forward_feature_set = ['account'] + [col for col in features_df.columns if 'f' in col and any(x in col for x in ['profit', 'size'])]

backward_feature_set = ['account'] + [col for col in features_df.columns if 'b' in col and any(x in col for x in ['profit', 'size'])]

# Save data using different categorization schemes
print("\n" + "="*50)
print("GENERATING DATA FILES")
print("="*50)

# Traditional 4-category system (compatible with original approach)
save_category_data(features_df, 'traditional_category', full_feature_set)

# Volume-based categories
save_category_data(features_df, 'volume_category', full_feature_set)

# Profit-based categories
save_category_data(features_df, 'profit_category', full_feature_set)

# Interaction-based categories
save_category_data(features_df, 'interaction_category', full_feature_set)

# Behavioral categories
save_category_data(features_df, 'behavior_category', full_feature_set)

# Create summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)

summary_stats = {
    'Total Accounts': len(features_df),
    'Accounts with Forward Transactions': features_df['has_forward_cnt'].sum(),
    'Accounts with Backward Transactions': features_df['has_backward_cnt'].sum(),
    'Accounts with Both Directions': (features_df['has_forward_cnt'] & features_df['has_backward_cnt']).sum(),
    'Isolated Accounts': (~features_df['has_forward_cnt'] & ~features_df['has_backward_cnt']).sum(),
    'Average Transactions per Account': features_df['total_transactions'].mean(),
    'Average Profit per Account': features_df['total_profit'].mean(),
    'High Activity Accounts (>100 txns)': (features_df['total_transactions'] > 100).sum(),
    'High Profit Accounts (>10000)': (features_df['total_profit'] > 10000).sum(),
}

for key, value in summary_stats.items():
    print(f"{key}: {value:.2f}")

# Save comprehensive summary
summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv('feature_summary_statistics.csv', index=False)
print("\nSaved summary statistics to feature_summary_statistics.csv")

# Save the enhanced feature dataset with all categories
features_df.to_csv('all_features_with_categories.csv', index=False)
print("Saved enhanced feature dataset to all_features_with_categories.csv")

print("\n" + "="*50)
print("DATA CLASSIFICATION COMPLETED!")
print("="*50)
print(f"Generated {len(features_df['traditional_category'].unique())} traditional categories")
print(f"Generated {len(features_df['volume_category'].unique())} volume categories")
print(f"Generated {len(features_df['profit_category'].unique())} profit categories")
print(f"Generated {len(features_df['interaction_category'].unique())} interaction categories")
print(f"Generated {len(features_df['behavior_category'].unique())} behavior categories")
print("Use any of these categorization schemes for training different classification models!")