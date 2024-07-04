# Data set with ionic conductivity annotations

This data set has been constructed by Homin Shin and Rhiannon Hendley, using as sources:

1. [A database of experimentally measured lithium solid electrolyte conductivities evaluated with machine learning](https://www.nature.com/articles/s41524-022-00951-z) 
2. [Identification of potential solid-state Li-ion conductors with semi-supervised learning](https://pubs.rsc.org/en/content/articlelanding/2023/ee/d2ee03499a)

See `other/` for more details about these sources

## Files

- `raw.xlsx`: Raw dataset with DOIs
- `processed.csv`: processed file with paper info, output of `preprocessing.py`
- `unidentified_with_refs.csv`: List of 56 entries with paper info that are not in either database, sent by Homin (for automatic DOIs only)
- `checked_by_hand.csv`: A list of 94 entries for which the doi was confirmed manually (for automatic DOIs only)

## Scripts
- preprocessing.py: Adds paper information, process data to be compatible with DAVE
- compare_xlsx.py: Script to compare to excel files since the raw dataset is an excel file. Usage: `python compare_xlsx.py OLD_FILE NEW_FILE`
- datavis.ipynb: Visualize the raw data
- get_mp_entries.py: script to automatically download similar cifs files from the MP
- get_icsd_entries.py: script to automatically download similar cifs files from the ICSD