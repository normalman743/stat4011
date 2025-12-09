import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Queue, Value, Lock
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and preprocess data once"""
    print("=== Parallel Feature Extraction System ===")
    print("Loading data...")
    
    # Load data
    pwd = '/Users/mannormal/4011/Qi Zihan/'
    ta = pd.read_csv(pwd + 'train_acc.csv')
    te = pd.read_csv(pwd + 'test_acc_predict.csv')
    trans = pd.read_csv(pwd + 'transactions.csv')
    
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
    
    return ta, te, trans, tdf

# Graph analysis functions
def find_to_nei(acc, trans_data):
    """Find following transaction accounts"""
    neis = []
    s = trans_data[trans_data.from_account==acc]
    if len(s) > 0:
        neis.extend([(x,y,z) for x,y,z in zip(s.to_account.tolist(),s.profit.tolist(),s.profit.tolist())])
    return neis

def find_from_nei(acc, trans_data):
    """Find prior transaction accounts"""
    neis = []
    s = trans_data[trans_data.to_account==acc]
    if len(s) > 0:
        neis.extend([(x,y,z) for x,y,z in zip(s.from_account.tolist(),s.profit.tolist(),s.profit.tolist())])
    return neis

def find_forward_paths(acc, h, trans_data):
    """Find a forward transaction path of depth h"""
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
                y = find_to_nei(u, trans_data)
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

def find_backward_paths(acc, h, trans_data):
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
                y = find_from_nei(u, trans_data)
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

def find_weights(paths, tdf_data):
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
            if account.startswith('a') and account in tdf_data.index:
                flag = tdf_data.loc[account, 'flag']
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

def extract_features_for_account(acc, trans_data, tdf_data):
    """Extract graph features for a single account"""
    features = {'account': acc}
    
    # Forward path analysis
    forward_paths = find_forward_paths(acc, 3, trans_data)
    if forward_paths:
        forward_weights = find_weights(forward_paths, tdf_data)
        
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
    backward_paths = find_backward_paths(acc, 3, trans_data)
    if backward_paths:
        backward_weights = find_weights(backward_paths, tdf_data)
        
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

def worker_process(task_queue, result_queue, trans_data, tdf_data, process_id):
    """Worker process that consumes tasks from queue and produces results"""
    processed_count = 0
    
    while True:
        try:
            # Get a task from queue
            task = task_queue.get(timeout=30)
            
            # Check for poison pill (shutdown signal)
            if task is None:
                break
                
            acc = task
            
            try:
                # Process single account
                features = extract_features_for_account(acc, trans_data, tdf_data)
                result_queue.put(('success', acc, features, process_id))
                processed_count += 1
                
            except Exception as e:
                # Send error result with default features
                default_features = {'account': acc}
                feature_names = ['normal_fprofit', 'normal_fpprofit', 'normal_fsize',
                               'abnormal_fprofit', 'abnormal_fpprofit', 'abnormal_fsize',
                               'bad_fprofit', 'bad_fpprofit', 'bad_fsize',
                               'A_fprofit', 'A_fpprofit', 'A_fsize',
                               'B_fprofit', 'B_fpprofit', 'B_fsize',
                               'normal_bprofit', 'normal_bpprofit', 'normal_bsize',
                               'abnormal_bprofit', 'abnormal_bpprofit', 'abnormal_bsize',
                               'bad_bprofit', 'bad_bpprofit', 'bad_bsize',
                               'A_bprofit', 'A_bpprofit', 'A_bsize',
                               'B_bprofit', 'B_bpprofit', 'B_bsize']
                for fname in feature_names:
                    default_features[fname] = 0
                result_queue.put(('error', acc, default_features, process_id))
                processed_count += 1
                print(f"Worker {process_id}: Error processing {acc}: {e}")
                
        except:
            # Queue timeout or other error, exit gracefully
            break

# Removed save_batch_to_disk function - now handled directly in batch_saver_process

def batch_saver_process(result_queue, progress_queue, total_accounts, batch_size=100):
    """Process that collects results and saves them in batches"""
    current_batch = []
    batch_id = 0
    processed_count = 0
    
    try:
        while processed_count < total_accounts:
            try:
                # Get result from worker
                result = result_queue.get(timeout=60)
                
                if result is None:  # Shutdown signal
                    break
                    
                _, acc, features, _ = result
                current_batch.append(features)
                processed_count += 1
                
                # Notify progress immediately for each account
                progress_queue.put(1)
                
                # Save batch when it reaches batch_size
                if len(current_batch) >= batch_size:
                    filename = f'features_batch_{batch_id:03d}.csv'
                    pd.DataFrame(current_batch).to_csv(filename, index=False)
                    print(f"Saved batch {batch_id}: {len(current_batch)} accounts → {filename}")
                    batch_id += 1
                    current_batch = []
                    
            except Exception as e:
                # For timeout, keep trying
                if "timed out" not in str(e).lower():
                    print(f"Batch saver error: {e}")
                continue
        
    except Exception as e:
        print(f"Batch saver fatal error: {e}")
    
    # Save remaining batch
    if current_batch:
        filename = f'features_batch_{batch_id:03d}.csv'
        pd.DataFrame(current_batch).to_csv(filename, index=False)
        print(f"Saved final batch {batch_id}: {len(current_batch)} accounts → {filename}")
    
    print(f"Feature extraction completed: {processed_count} accounts processed")

def load_progress():
    """Load processing progress from checkpoint file"""
    if os.path.exists('extraction_progress.json'):
        with open('extraction_progress.json', 'r') as f:
            progress_data = json.load(f)
            # Ensure consistency: total_processed should match completed_accounts length
            if 'completed_accounts' in progress_data:
                progress_data['total_processed'] = len(progress_data['completed_accounts'])
            return progress_data
    return {'completed_accounts': [], 'completed_batches': [], 'total_processed': 0}

def save_progress(progress):
    """Save processing progress to checkpoint file"""
    with open('extraction_progress.json', 'w') as f:
        json.dump(progress, f)

def save_batch_features(features, batch_id):
    """Save features for a batch to individual file"""
    filename = f'features_batch_{batch_id:03d}.csv'
    pd.DataFrame(features).to_csv(filename, index=False)
    print(f"Saved batch {batch_id} features to {filename}")

def load_all_batch_features():
    """Load and combine all batch feature files"""
    all_features = []
    batch_files = [f for f in os.listdir('.') if f.startswith('features_batch_') and f.endswith('.csv')]
    batch_files.sort()  # Ensure consistent order
    
    print(f"Found {len(batch_files)} batch files to combine")
    for filename in batch_files:
        batch_df = pd.read_csv(filename)
        all_features.extend(batch_df.to_dict('records'))
        print(f"Loaded {len(batch_df)} features from {filename}")
    
    return all_features

if __name__ == '__main__':
    # Load data only once in main process
    ta, te, trans, tdf = load_data()
    
    # Combine all accounts first
    all_accounts = list(ta['account']) + list(te['account'])
    all_accounts = list(set(all_accounts))  # Remove duplicates
    total_accounts = len(all_accounts)
    
    print(f"Total accounts: {total_accounts}")
    
    # Check if complete features already exist
    if os.path.exists('all_features.csv'):
        print("Found existing all_features.csv file!")
        existing_df = pd.read_csv('all_features.csv')
        print(f"Existing features shape: {existing_df.shape}")
        
        if len(existing_df) == total_accounts:
            print("✅ Complete feature extraction already done!")
            print("Sample features:")
            print(existing_df.head())
            exit()
        else:
            print(f"⚠️  Existing file has {len(existing_df)} accounts, but need {total_accounts}")
            print("Continuing with feature extraction...")
    
    # Load existing progress from batch files
    progress = load_progress()
    completed_accounts = set(progress['completed_accounts'])
    
    # Filter out already completed accounts
    remaining_accounts = [acc for acc in all_accounts if acc not in completed_accounts]
    
    print(f"Already processed: {len(completed_accounts)} accounts")
    print(f"Remaining to process: {len(remaining_accounts)} accounts")
    
    if not remaining_accounts:
        print("All accounts already processed in batches! Combining features...")
        all_features = load_all_batch_features()
        features_df = pd.DataFrame(all_features)
        print(f"Combined features shape: {features_df.shape}")
        
        # Save final combined file
        features_df.to_csv('all_features.csv', index=False)
        print("Saved combined features to all_features.csv")
        exit()
    
    # Task Queue Mode - Real-time Progress Updates  
    num_processes = 14  # Use 16 processes as requested
    batch_size = 100  # Save every 100 completed accounts
    
    print(f"Using task queue mode with {num_processes} worker processes")
    print(f"Real-time progress updates, saving every {batch_size} accounts")
    
    # Create multiprocessing queues
    task_queue = Queue()
    result_queue = Queue()
    progress_queue = Queue()
    
    # Add all remaining accounts to task queue
    for acc in remaining_accounts:
        task_queue.put(acc)
    
    # Add poison pills to signal workers to stop (one per worker)
    for _ in range(num_processes):
        task_queue.put(None)
    
    print(f"Added {len(remaining_accounts)} accounts to task queue")
    
    # Start batch saver process
    saver = mp.Process(
        target=batch_saver_process,
        args=(result_queue, progress_queue, len(remaining_accounts), batch_size)
    )
    saver.start()
    
    # Start worker processes
    workers = []
    for i in range(num_processes):
        worker = mp.Process(
            target=worker_process,
            args=(task_queue, result_queue, trans, tdf, i)
        )
        worker.start()
        workers.append(worker)
    
    print(f"Started {num_processes} workers and batch saver")
    
    # Real-time progress tracking
    processed_count = 0
    last_update_time = time.time()
    
    with tqdm(total=len(remaining_accounts), desc="Processing accounts") as pbar:
        
        while processed_count < len(remaining_accounts):
            try:
                # Wait for progress update (each account completion)
                progress_update = progress_queue.get(timeout=10)
                processed_count += progress_update
                pbar.update(progress_update)
                last_update_time = time.time()
                
                # Update progress file every 10 accounts
                if processed_count % 10 == 0:
                    progress['total_processed'] = len(all_accounts) - len(remaining_accounts) + processed_count
                    save_progress(progress)
                    
            except:
                # Check if processes are still alive
                alive_workers = [w for w in workers if w.is_alive()]
                saver_alive = saver.is_alive()
                
                if not alive_workers and not saver_alive:
                    break
                    
                # Continue waiting
                continue
                    
    print("\nWaiting for all processes to finish...")
    
    # Wait for all workers to complete
    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            print(f"Force terminating worker {worker.pid}")
            worker.terminate()
    
    # Signal batch saver to stop and wait
    result_queue.put(None)
    saver.join(timeout=10)
    if saver.is_alive():
        print("Force terminating batch saver")
        saver.terminate()
    
    print("All processes completed!")
    
    # Final progress update
    progress['total_processed'] = len(all_accounts)
    progress['completed_accounts'] = all_accounts
    save_progress(progress)
    
    print(f"Total processed: {len(all_accounts)} accounts")
    
    # Combine all batch files into final result
    print("Combining all batch features...")
    all_features = load_all_batch_features()
    features_df = pd.DataFrame(all_features)
    
    print(f"Final features shape: {features_df.shape}")
    print("Sample features:")
    print(features_df.head())
    
    # Save the complete feature dataset
    features_df.to_csv('all_features.csv', index=False)
    print("Saved complete features to all_features.csv")
    
    # Clean up batch files (optional)
    cleanup = input("Delete individual batch files? (y/n): ")
    if cleanup.lower() == 'y':
        batch_files = [f for f in os.listdir('.') if f.startswith('features_batch_') and f.endswith('.csv')]
        for filename in batch_files:
            os.remove(filename)
        os.remove('extraction_progress.json')
        print(f"Cleaned up {len(batch_files)} batch files and progress file")