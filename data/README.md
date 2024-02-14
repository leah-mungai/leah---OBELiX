# Data set with ionic conductivity annotations

This data set has been constructed by Homin Shin and Rhiannon Hendley, using as sources:

1. [A database of experimentally measured lithium solid electrolyte conductivities evaluated with machine learning](https://www.nature.com/articles/s41524-022-00951-z) 
2. [Identification of potential solid-state Li-ion conductors with semi-supervised learning](https://pubs.rsc.org/en/content/articlelanding/2023/ee/d2ee03499a)

## Files

- `20231204.xlsx`: Original Excel file provided by Homin
- `20231204.csv`: Direct conversion to CSV
- `20231204v1.csv`: Post-processing by Alex
    - 2023.12.06: Replace "E-" by "e-"
    - 2023.12.06: Set all "<1e-10" to 1e-15
- `20240224.csv`: Post processing by Divya (removed all string target labels into sep_samples.csv)
- `20240224v1.csv`: Processed string composition into elements and counts
- `20240224v1.csv`: Log10 conversion of target column
- `20231204v1_with_paper_info.csv`: same as `20231204v1.csv` with paper information except for 56/608 entries
- `unidentified.txt`: List of 56 entries that can't be found in either papers 
- `laskowski_semi-fromatted.csv`: Data from the pdf SI of paper 2 above. Some of the pdf was not perfectly rendered in the csv
- `LiIonDatabase.csv`: Data from paper 1 above, take directly from [here](http://pcwww.liv.ac.uk/~msd30/lmds/LiIonDatabase.html)