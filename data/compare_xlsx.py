import pandas as pd

old = "raw.xlsx"
old_data = pd.read_excel(old)

new = "raw_new.xlsx"
new_data = pd.read_excel(new)

print(old.compare(new))
