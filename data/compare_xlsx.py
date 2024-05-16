import numpy as np
import pandas as pd
import sys

def isnan(x):
    try:
        return np.isnan(x)
    except:
        return False
        
old = sys.argv[1]
old_data = pd.read_excel(old)

new = sys.argv[2]
new_data = pd.read_excel(new)

try:
    compare = old_data.compare(new_data, result_names=('Old', 'New'))
except ValueError:

    if len(old_data.columns) < len(new_data.columns):
        print("There are new columns:")
        print([c for c in new_data.columns if c not in old_data.columns])
        common_columns = [c for c in new_data.columns if c in old_data.columns]
        
    if len(old_data.columns) > len(new_data.columns):
        print("There are missing columns:")
        print([c for c in old_data.columns if c not in new_data.columns])
        common_columns = [c for c in old_data.columns if c in new_data.columns]

    old_data = old_data[common_columns]
    new_data_orig = new_data.copy()
    new_data = new_data[common_columns]

    compare = old_data.compare(new_data, result_names=('Old', 'New'))

changed_rows = set()
for row in compare.index:
    for c in compare.loc[row].index:
        if ~isnan(compare.loc[row][c]):
            print(f"{row} - {c} - {compare.loc[row][c]}")
            changed_rows = changed_rows.union({row})
    print()
        
