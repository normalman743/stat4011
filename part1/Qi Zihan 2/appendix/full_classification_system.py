import pandas as pd
import numpy as np

# Load data
print("Loading data...")
ta = pd.read_csv('../train_acc.csv')
te = pd.read_csv('../test_acc_predict.csv')
trans = pd.read_csv('../transactions.csv')

print(f"Training accounts: {ta.shape[0]}")
print(f"Test accounts: {te.shape[0]}")
print(f"Transactions: {trans.shape[0]}")

# Calculate profit for transactions
trans['pprofit'] = (trans.value.astype('float') - trans.gas*trans.gas_price)/100000
trans['profit'] = np.where(trans['pprofit'] > 0, trans['pprofit'], 0)

# Replace zero flag by -1 flag in training data
ta.loc[ta[ta.flag==0].index,'flag'] = -1

# Merge training and testing accounts
tdf = pd.merge(ta, te[['account']], left_on=['account'], right_on=['account'], how='outer')
tdf.replace(np.nan, 0, inplace=True)
tdf.set_index('account', inplace=True)

# Graph analysis functions (same as before)
def find_to_nei(acc):
    """Find following transaction accounts"""
    neis = []
    s = trans[trans.from_account==acc]
    if len(s) > 0:
        neis.extend([(x,y,z) for x,y,z in zip(s.to_account.tolist(),s.profit.tolist(),s.profit.tolist())])
    return neis

def find_from_nei(acc):
    """Find prior transaction accounts"""
    neis = []
    s = trans[trans.to_account==acc]
    if len(s) > 0:
        neis.extend([(x,y,z) for x,y,z in zip(s.from_account.tolist(),s.profit.tolist(),s.profit.tolist())])
    return neis

def find_forward_paths(acc, h):
    """Find a forward transaction path of depth h"""
    paths = [[(acc,0,0)]]
    if h > 0:
        stop = 0
        iteration = 0
        max_iterations = h + 2  # Prevent infinite loops
        while stop == 0 and iteration < max_iterations:
            newpaths = []
            for i in range(len(paths)):
                a_path = paths[i]
                u = a_path[-1][0]
                y = find_to_nei(u)
                if len(y) > 0:
                    for s,t,a in y[:5]:  # Limit to first 5 to prevent explosion
                        if len(a_path) < h:  # Only extend if not at max depth
                            x = a_path.copy()
                            x.append((s,t,a))
                            newpaths.append(x)
                else:
                    newpaths.append(a_path)
            
            if not newpaths or max(len(x) for x in newpaths) >= h:
                stop = 1
            elif max(len(x) for x in newpaths) == max(len(x) for x in paths):
                stop = 1
            paths = newpaths[:100]  # Limit number of paths
            iteration += 1
    return paths

def find_backward_paths(acc, h):
    """Find a backward transaction path of depth h"""
    paths = [[(acc,0,0)]]
    if h > 0:
        stop = 0
        iteration = 0
        max_iterations = h + 2
        while stop == 0 and iteration < max_iterations:
            newpaths = []
            for i in range(len(paths)):
                a_path = paths[i]
                u = a_path[-1][0]
                y = find_from_nei(u)
                if len(y) > 0:
                    for s,t,a in y[:5]:  # Limit to first 5
                        if len(a_path) < h:
                            x = a_path.copy()
                            x.append((s,t,a))
                            newpaths.append(x)
                else:
                    newpaths.append(a_path)
            
            if not newpaths or max(len(x) for x in newpaths) >= h:
                stop = 1
            elif max(len(x) for x in newpaths) == max(len(x) for x in paths):
                stop = 1
            paths = newpaths[:100]  # Limit number of paths
            iteration += 1
    return paths

def find_weights(paths):
    """Process paths and extract features"""
    cnt_data = []
    cnt2_data = []
    
    for path in paths:
        for node in path[1:]:  # Skip the starting node
            account = node[0]
            profit = node[1] 
            pprofit = node[2]
            
            # Add to cnt2 (all transactions)
            cnt2_data.append({
                'account': account,
                'profit': profit,
                'pprofit': pprofit,
                'size': 1
            })
            
            # Add to cnt only if it's an 'a' account
            if account.startswith('a') and account in tdf.index:
                flag = tdf.loc[account, 'flag']
                cnt_data.append({
                    'flag': flag,
                    'profit': profit,
                    'pprofit': pprofit,
                    'size': 1
                })
    
    result = {}
    if cnt_data:
        cnt_df = pd.DataFrame(cnt_data)
        result['cnt'] = cnt_df.groupby('flag').agg({
            'profit': 'sum',
            'pprofit': 'sum', 
            'size': 'sum'
        })
    
    if cnt2_data:
        cnt2_df = pd.DataFrame(cnt2_data)
        result['cnt2'] = cnt2_df.groupby('account').agg({
            'profit': 'sum',
            'pprofit': 'sum',
            'size': 'sum'
        })
    
    return result

def extract_features_for_account(acc):
    """Extract graph features for a single account"""
    features = {}
    
    # Forward path analysis
    forward_paths = find_forward_paths(acc, 3)
    if forward_paths:
        forward_weights = find_weights(forward_paths)
        
        # Features from forward cnt (transactions ending with 'a' accounts)
        if 'cnt' in forward_weights:
            cnt = forward_weights['cnt']
            if -1.0 in cnt.index:
                features['normal_fprofit'] = cnt.loc[-1.0, 'profit']
                features['normal_fpprofit'] = cnt.loc[-1.0, 'pprofit'] 
                features['normal_fsize'] = cnt.loc[-1.0, 'size']
            else:
                features['normal_fprofit'] = features['normal_fpprofit'] = features['normal_fsize'] = 0
                
            if 0.0 in cnt.index:
                features['abnormal_fprofit'] = cnt.loc[0.0, 'profit']
                features['abnormal_fpprofit'] = cnt.loc[0.0, 'pprofit']
                features['abnormal_fsize'] = cnt.loc[0.0, 'size']
            else:
                features['abnormal_fprofit'] = features['abnormal_fpprofit'] = features['abnormal_fsize'] = 0
                
            if 1.0 in cnt.index:
                features['bad_fprofit'] = cnt.loc[1.0, 'profit']
                features['bad_fpprofit'] = cnt.loc[1.0, 'pprofit']
                features['bad_fsize'] = cnt.loc[1.0, 'size']
            else:
                features['bad_fprofit'] = features['bad_fpprofit'] = features['bad_fsize'] = 0
        else:
            features.update({
                'normal_fprofit': 0, 'normal_fpprofit': 0, 'normal_fsize': 0,
                'abnormal_fprofit': 0, 'abnormal_fpprofit': 0, 'abnormal_fsize': 0,
                'bad_fprofit': 0, 'bad_fpprofit': 0, 'bad_fsize': 0
            })
        
        # Features from forward cnt2 (all transactions)
        if 'cnt2' in forward_weights:
            cnt2 = forward_weights['cnt2']
            a_accounts = cnt2[cnt2.index.str.startswith('a', na=False)]
            b_accounts = cnt2[cnt2.index.str.startswith('b', na=False)]
            
            features['A_fprofit'] = a_accounts['profit'].sum() if not a_accounts.empty else 0
            features['A_fpprofit'] = a_accounts['pprofit'].sum() if not a_accounts.empty else 0
            features['A_fsize'] = a_accounts['size'].sum() if not a_accounts.empty else 0
            
            features['B_fprofit'] = b_accounts['profit'].sum() if not b_accounts.empty else 0
            features['B_fpprofit'] = b_accounts['pprofit'].sum() if not b_accounts.empty else 0
            features['B_fsize'] = b_accounts['size'].sum() if not b_accounts.empty else 0
        else:
            features.update({
                'A_fprofit': 0, 'A_fpprofit': 0, 'A_fsize': 0,
                'B_fprofit': 0, 'B_fpprofit': 0, 'B_fsize': 0
            })
    else:
        features.update({
            'normal_fprofit': 0, 'normal_fpprofit': 0, 'normal_fsize': 0,
            'abnormal_fprofit': 0, 'abnormal_fpprofit': 0, 'abnormal_fsize': 0,
            'bad_fprofit': 0, 'bad_fpprofit': 0, 'bad_fsize': 0,
            'A_fprofit': 0, 'A_fpprofit': 0, 'A_fsize': 0,
            'B_fprofit': 0, 'B_fpprofit': 0, 'B_fsize': 0
        })
    
    # Backward path analysis
    backward_paths = find_backward_paths(acc, 3)
    if backward_paths:
        backward_weights = find_weights(backward_paths)
        
        # Similar processing for backward paths
        if 'cnt' in backward_weights:
            cnt = backward_weights['cnt']
            if -1.0 in cnt.index:
                features['normal_bprofit'] = cnt.loc[-1.0, 'profit']
                features['normal_bpprofit'] = cnt.loc[-1.0, 'pprofit']
                features['normal_bsize'] = cnt.loc[-1.0, 'size']
            else:
                features['normal_bprofit'] = features['normal_bpprofit'] = features['normal_bsize'] = 0
                
            if 0.0 in cnt.index:
                features['abnormal_bprofit'] = cnt.loc[0.0, 'profit']
                features['abnormal_bpprofit'] = cnt.loc[0.0, 'pprofit']
                features['abnormal_bsize'] = cnt.loc[0.0, 'size']
            else:
                features['abnormal_bprofit'] = features['abnormal_bpprofit'] = features['abnormal_bsize'] = 0
                
            if 1.0 in cnt.index:
                features['bad_bprofit'] = cnt.loc[1.0, 'profit']
                features['bad_bpprofit'] = cnt.loc[1.0, 'pprofit']
                features['bad_bsize'] = cnt.loc[1.0, 'size']
            else:
                features['bad_bprofit'] = features['bad_bpprofit'] = features['bad_bsize'] = 0
        else:
            features.update({
                'normal_bprofit': 0, 'normal_bpprofit': 0, 'normal_bsize': 0,
                'abnormal_bprofit': 0, 'abnormal_bpprofit': 0, 'abnormal_bsize': 0,
                'bad_bprofit': 0, 'bad_bpprofit': 0, 'bad_bsize': 0
            })
            
        if 'cnt2' in backward_weights:
            cnt2 = backward_weights['cnt2']
            a_accounts = cnt2[cnt2.index.str.startswith('a', na=False)]
            b_accounts = cnt2[cnt2.index.str.startswith('b', na=False)]
            
            features['A_bprofit'] = a_accounts['profit'].sum() if not a_accounts.empty else 0
            features['A_bpprofit'] = a_accounts['pprofit'].sum() if not a_accounts.empty else 0
            features['A_bsize'] = a_accounts['size'].sum() if not a_accounts.empty else 0
            
            features['B_bprofit'] = b_accounts['profit'].sum() if not b_accounts.empty else 0
            features['B_bpprofit'] = b_accounts['pprofit'].sum() if not b_accounts.empty else 0
            features['B_bsize'] = b_accounts['size'].sum() if not b_accounts.empty else 0
        else:
            features.update({
                'A_bprofit': 0, 'A_bpprofit': 0, 'A_bsize': 0,
                'B_bprofit': 0, 'B_bpprofit': 0, 'B_bsize': 0
            })
    else:
        features.update({
            'normal_bprofit': 0, 'normal_bpprofit': 0, 'normal_bsize': 0,
            'abnormal_bprofit': 0, 'abnormal_bpprofit': 0, 'abnormal_bsize': 0,
            'bad_bprofit': 0, 'bad_bpprofit': 0, 'bad_bsize': 0,
            'A_bprofit': 0, 'A_bpprofit': 0, 'A_bsize': 0,
            'B_bprofit': 0, 'B_bpprofit': 0, 'B_bsize': 0
        })
    
    return features

print("Feature extraction functions defined!")

# Test with a few accounts
print("\nTesting feature extraction with sample accounts...")
sample_accounts = ta.head(3)['account'].tolist()

for acc in sample_accounts:
    print(f"\nExtracting features for {acc}...")
    features = extract_features_for_account(acc)
    print(f"Features extracted: {len(features)} features")
    # Show first few features
    feature_items = list(features.items())[:5]
    print(f"Sample features: {feature_items}")

print("\nFeature extraction test completed!")
print("Ready to process all accounts and build classification models...")