import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Load data
ta = pd.read_csv('../train_acc.csv')
te = pd.read_csv('../test_acc_predict.csv')
trans = pd.read_csv('../transactions.csv')

print("Data loaded successfully!")
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

print("Data preprocessing completed!")

# Functions to find transaction neighbors
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
        while stop == 0:
            newpaths = []
            for i in range(len(paths)):
                a_path = paths[i]
                u = a_path[-1][0]
                y = find_to_nei(u)
                if len(y) > 0:
                    for s,t,a in y:
                        x = a_path.copy()
                        x.append((s,t,a))
                        newpaths.append(x)
                else:
                    newpaths.append(a_path)
            if max(len(x) for x in newpaths) == h:
                stop = 1
            elif max(len(x) for x in newpaths) == max(len(x) for x in paths):
                stop = 1
            paths = newpaths
    return paths

def find_backward_paths(acc, h):
    """Find a backward transaction path of depth h"""
    paths = [[(acc,0,0)]]
    if h > 0:
        stop = 0
        while stop == 0:
            newpaths = []
            for i in range(len(paths)):
                a_path = paths[i]
                u = a_path[-1][0]
                y = find_from_nei(u)
                if len(y) > 0:
                    for s,t,a in y:
                        x = a_path.copy()
                        x.append((s,t,a))
                        newpaths.append(x)
                else:
                    newpaths.append(a_path)
            if max(len(x) for x in newpaths) == h:
                stop = 1
            elif max(len(x) for x in newpaths) == max(len(x) for x in paths):
                stop = 1
            paths = newpaths
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
        }).reset_index()
    
    if cnt2_data:
        cnt2_df = pd.DataFrame(cnt2_data)
        result['cnt2'] = cnt2_df.groupby('account').agg({
            'profit': 'sum',
            'pprofit': 'sum',
            'size': 'sum'
        }).reset_index()
    
    return result

print("Graph analysis functions defined!")

# Test with a sample account
sample_acc = ta.iloc[0]['account']
print(f"\nTesting with sample account: {sample_acc}")

forward_paths = find_forward_paths(sample_acc, 2)
backward_paths = find_backward_paths(sample_acc, 2)

print(f"Forward paths found: {len(forward_paths)}")
print(f"Backward paths found: {len(backward_paths)}")

if forward_paths:
    forward_weights = find_weights(forward_paths)
    print("Forward path analysis completed!")

if backward_paths:
    backward_weights = find_weights(backward_paths)
    print("Backward path analysis completed!")

print("\nGraph network analysis setup complete!")
print("Ready to implement full feature extraction and classification...")