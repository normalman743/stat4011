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
ta = pd.read_csv('../train_acc.csv')

# Import the testing accounts
te = pd.read_csv('../test_acc_predict.csv')

# Import the transaction details
trans = pd.read_csv('../transactions.csv')

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