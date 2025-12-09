import os, math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

def classify_account_type(row):
    has_forward = (row['normal_fprofit'] > 0 or row['abnormal_fprofit'] > 0 or 
                   row['normal_fsize'] > 0 or row['abnormal_fsize'] > 0)
    has_backward = (row['normal_bprofit'] > 0 or row['abnormal_bprofit'] > 0 or
                    row['normal_bsize'] > 0 or row['abnormal_bsize'] > 0)
    if has_forward and has_backward: return 'type1'
    if has_forward and not has_backward: return 'type2'
    if not has_forward and has_backward: return 'type3'
    return 'type4'

def eval_one_setting_type(
    df_type, feature_cols, n_estimators=100, n_models=100, threshold_ratio=0.7,
    use_balanced_weight=True, random_state=42, n_splits=5
):
    """对某个 account_type 做 K 折评估，返回 binary-F1 / macro / weighted"""
    if len(df_type) < 2: 
        return None, None, None

    # 标签：-1→0
    y = np.where(df_type['flag'].values == -1, 0, 1)
    X = df_type[feature_cols].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    f1_bin_list, f1_mac_list, f1_wgt_list = [], [], []

    for fold_idx, (tr, va) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X[tr], y[tr]
        X_va, y_va = X[va], y[va]

        # 外层集成：重复训练 n_models 个 RF
        preds = []
        for m in range(n_models):
            if use_balanced_weight:
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=1000 + m,
                    class_weight='balanced',
                    n_jobs=-1
                )
                clf.fit(X_tr, y_tr)
            else:
                # 你的“对折采样”版本（可替换）
                # 这里简单示例随机下采样等量
                pos_idx = np.where(y_tr==1)[0]
                neg_idx = np.where(y_tr==0)[0]
                if len(pos_idx)==0 or len(neg_idx)==0:
                    continue
                ss = min(len(pos_idx), len(neg_idx))
                np.random.seed(1000 + m)
                pick_pos = np.random.choice(pos_idx, ss, replace=True)
                pick_neg = np.random.choice(neg_idx, ss, replace=True)
                pick = np.concatenate([pick_pos, pick_neg])
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=1000 + m,
                    n_jobs=-1
                )
                clf.fit(X_tr[pick], y_tr[pick])

            preds.append(clf.predict(X_va))

        if not preds:
            # 这一折训练失败（极端不平衡），跳过
            continue

        preds = np.array(preds)              # [n_models, n_va]
        votes = np.sum(preds, axis=0)        # 每个样本的“赞成1”的票数
        thr = int(math.ceil(threshold_ratio * preds.shape[0]))  # 比例→票数
        y_hat = (votes >= thr).astype(int)

        f1_bin = metrics.f1_score(y_va, y_hat, average='binary', zero_division=0)
        f1_mac = metrics.f1_score(y_va, y_hat, average='macro', zero_division=0)
        f1_wgt = metrics.f1_score(y_va, y_hat, average='weighted', zero_division=0)
        f1_bin_list.append(f1_bin); f1_mac_list.append(f1_mac); f1_wgt_list.append(f1_wgt)

    if not f1_bin_list:
        return 0.0, 0.0, 0.0

    return np.mean(f1_bin_list), np.mean(f1_mac_list), np.mean(f1_wgt_list)

if __name__ == "__main__":
    print("=== 爬山式参数调优系统 (Enhanced Ensemble / CV, ratio-threshold) ===")

    # 路径
    features_path = '/Users/mannormal/4011/Qi Zihan/feature_extraction/generated_features/all_features.csv'
    pwd = '/Users/mannormal/4011/Qi Zihan/original_data/'

    # 数据
    all_features_df = pd.read_csv(features_path)
    ta = pd.read_csv(pwd + 'train_acc.csv')
    ta.loc[ta['flag']==0, 'flag'] = -1

    training_df = pd.merge(all_features_df, ta[['account','flag']], on='account', how='inner')
    training_df['account_type'] = training_df.apply(classify_account_type, axis=1)

    # 特征列
    ignore_cols = ['account', 'flag', 'account_type']
    feature_cols = [c for c in training_df.columns if c not in ignore_cols]

    # 粗搜网格（注意：阈值用比例）
    n_estimators_list = [100, 150]
    n_models_list     = [100, 150]
    thr_ratio_list    = [0.93]

    best = {}

    for t in ['type1','type2','type3','type4']:
        df_t = training_df[training_df['account_type']==t].copy()
        if df_t.empty:
            print(f">>> {t}: 无数据，跳过")
            continue

        print(f"\n>>> 调参: {t} | 样本={len(df_t)} | 正类={int((df_t['flag']==1).sum())} 负类={int((df_t['flag']==-1).sum())}")
        best_key = None
        best_scores = (-1, -1, -1)

        # 粗搜
        for ne in n_estimators_list:
            for nm in n_models_list:
                for trr in thr_ratio_list:
                    f1b, f1m, f1w = eval_one_setting_type(
                        df_t, feature_cols,
                        n_estimators=ne, n_models=nm, threshold_ratio=trr,
                        use_balanced_weight=True
                    )
                    print(f"  尝试: F1bin={f1b:.4f} (macro={f1m:.4f}, weighted={f1w:.4f})  params={{'n_estimators':{ne}, 'n_models':{nm}, 'thr_ratio':{trr}}}")

                    if f1b > best_scores[0]:
                        best_scores = (f1b, f1m, f1w)
                        best_key = {'n_estimators': ne, 'n_models': nm, 'thr_ratio': trr}

        # 局部爬山（在最优点 ±5 搜）
        # n_estimators 与 n_models 使用 ±5，阈值比例用 ±0.05
        if best_key:
            ne0, nm0, tr0 = best_key['n_estimators'], best_key['n_models'], best_key['thr_ratio']

            def clamp(x, lo, hi): return max(lo, min(hi, x))

            for _ in range(2):  # 小迭代两轮
                # 调 n_estimators
                for ne in [clamp(ne0-5,1,500), ne0, clamp(ne0+5,1,500)]:
                    f1b, f1m, f1w = eval_one_setting_type(df_t, feature_cols, ne, nm0, tr0, True)
                    if f1b > best_scores[0]:
                        best_scores = (f1b, f1m, f1w); ne0 = ne

                # 调 n_models
                for nm in [clamp(nm0-5,10,500), nm0, clamp(nm0+5,10,500)]:
                    f1b, f1m, f1w = eval_one_setting_type(df_t, feature_cols, ne0, nm, tr0, True)
                    if f1b > best_scores[0]:
                        best_scores = (f1b, f1m, f1w); nm0 = nm

                # 调 阈值比例
                for trr in [clamp(tr0-0.05,0.1,0.99), tr0, clamp(tr0+0.05,0.1,0.99)]:
                    f1b, f1m, f1w = eval_one_setting_type(df_t, feature_cols, ne0, nm0, trr, True)
                    if f1b > best_scores[0]:
                        best_scores = (f1b, f1m, f1w); tr0 = trr

            best_key = {'n_estimators': ne0, 'n_models': nm0, 'thr_ratio': tr0}

        best[t] = {'f1_binary': best_scores[0], 'f1_macro': best_scores[1], 'f1_weighted': best_scores[2], 'params': best_key}
        print(f"  最优结果: F1bin={best_scores[0]:.4f}  (macro={best_scores[1]:.4f}, weighted={best_scores[2]:.4f})  params={best_key}")

    print("\n=== 各类型最佳 ===")
    for t, res in best.items():
        print(f"{t}: F1bin={res['f1_binary']:.4f}  macro={res['f1_macro']:.4f}  weighted={res['f1_weighted']:.4f}  params={res['params']}")
