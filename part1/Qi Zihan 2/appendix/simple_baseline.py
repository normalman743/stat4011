import pandas as pd
import numpy as np
from collections import Counter

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

# Check flag distribution in training data
print("\nFlag distribution in training data:")
print(ta['flag'].value_counts())

# Calculate profit for transactions
print("\nCalculating transaction profits...")
trans['pprofit'] = (trans.value.astype('float') - trans.gas*trans.gas_price)/100000

# An indicator variable to show positive profit
trans['profit'] = np.where(trans['pprofit'] > 0, trans['pprofit'], 0)

print("Transactions with profit calculation:")
print(trans[['from_account', 'to_account', 'value', 'gas', 'gas_price', 'profit']].head())

# Replace zero flag by -1 flag in training data
ta_copy = ta.copy()
ta_copy.loc[ta_copy[ta_copy.flag==0].index,'flag'] = -1

print("\nUpdated flag distribution (0->-1):")
print(ta_copy['flag'].value_counts())