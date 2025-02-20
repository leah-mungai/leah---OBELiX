# OBELiX: An Experimental Ionic Conductivity Dataset

OBELiX (<ins>O</ins>pen solid <ins>B</ins>attery <ins>E</ins>lectrolytes with <ins>Li</ins>: an e<ins>X</ins>perimental dataset) is an dataset of 599 synthesized solid electrolyte materials and their experimentally measured room temperature ionic conductivity along with descriptors of their space group, lattice parameters, and chemical composition. It contains full crystallographic description in the form of CIF files for 321 entries.

<h1 align="center">
<img src="https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/paper/figures/gathered.svg">


## Python API

### Installation

```
pip install obelix
```

### Usage

```
from obelix import OBELiX

ob = OBELiX()

print(f"The ionic conductivity of {ob[0]["Reduced Composition"]} is {ob[0]["Ionic conductivity (S cm-1)"]}")
```

## Files Download
| File               | Links   |
| --------           | ------- |
| Train Dataset       | [xlsx](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/train.xlsx), [csv](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/train.csv)|
| Train CIF files      | [zip](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/train_cifs.zip), [tar.gz](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/train_cifs.tar.gz)    |
| Test Dataset       | [xlsx](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/test.xlsx), [csv](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/test.csv)|
| Test CIF files      | [zip](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/test_cifs.zip), [tar.gz](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/test_cifs.tar.gz)    |
| Full Dataset       | [xlsx](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/all.xlsx), [csv](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/all.csv)|
| All CIF files      | [zip](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/all_cifs.zip), [tar.gz](https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/all_cifs.tar.gz)    |

