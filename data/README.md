# Data set with ionic conductivity annotations

This data set has been constructed by Homin Shin and Rhiannon Hendley, using as sources:

1. [A database of experimentally measured lithium solid electrolyte conductivities evaluated with machine learning](https://www.nature.com/articles/s41524-022-00951-z) 
2. [Identification of potential solid-state Li-ion conductors with semi-supervised learning](https://pubs.rsc.org/en/content/articlelanding/2023/ee/d2ee03499a)

See `other/` for more details about these sources

## Files

- `raw.xlsx`: Original Excel file provided by Homin, with corrections by Homin
- `unidentified_with_refs.csv`: List of 56 entries with paper info that are not in either database, sent by Homin
- `checked_by_hand.csv`: A list of 94 entries for which the doi was confirmed manually
- `processed.csv`: processed file with paper info, output of `preprocessing.py`

## Scripts
- preprocessing.py: Adds paper information, process data to be compatible with DAVE
- datavis.ipynb: Visualize the raw data