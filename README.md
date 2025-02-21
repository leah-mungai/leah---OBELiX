# OBELiX: An Experimental Ionic Conductivity Dataset

OBELiX (<ins>O</ins>pen solid <ins>B</ins>attery <ins>E</ins>lectrolytes with <ins>Li</ins>: an e<ins>X</ins>perimental dataset) is a dataset of 599 synthesized solid electrolyte materials and their **experimentally measured room temperature ionic conductivity** along with descriptors of their space group, lattice parameters, and chemical composition. It contains full crystallographic description in the form of CIF files for 321 entries. 

A full description an analysis can be found in [our paper](https://arxiv.org/abs/2502.14234)

<h1 align="center">
<img src="https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/paper/figures/gathered.svg">


## Python API

### Installation

```our 
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

## Labels and Features

| Name | Definition |
|------|------------|
| `ID`  | Entry identifier, a three symbols alphanumeric string|
| `Reduced Composition` | Chemical formula of the material |
| `Z` | Number of formula units in the unit cell |
| `True Composition` | Composition of the unit cell |
| `Ionic conductivity (S cm-1)` | Experimental ionic conductivity in Siemens per centimeter. It is the total ionic conductivity when both are reported |
| `IC (Total)` | Total ionic conductivity in S/cm| 
| `IC (Bulk)` | Bulk Ionic conductivity in S/cm, if both total and bulk are blank, it is assumed that the reported conductivity is the total IC|
| `Space group` | Space group Hermann–Mauguin notation short name|
| `Space group #` | Space group number |
| `a`, `b`, `c`, `alpha`, `beta`, `gamma` | Lattice parameters |
| `Family` | Crystal family | 
| `DOI` | Digital object identifier of the original experimental publication from which this IC measurement was taken|
| `Checked` | (*OUTDATED*) Weather entries was manually checked |
| `Ref` | (*OUTDATED*, see `Laskowski ID` and `Liion ID` instead)) Reference from which data was taken from D1=Liion, D2=Laskowski |
| `Cif ID` | Whether the entry has a cif or not. This field will read "done" when it does and will be empty otherwise|
| `Cif ref_1`, `Cif ref_2` | (*OUTDATED*, see `ICSD ID`) Notes of where to find the CIF information if available|
| `note` | Notes and comments about the entry |
| `close match` | Whether the cif file comes from a closely matching structure or from the actual publication (DOI). If a close match this field will read "Yes"
| `close match DOI` | Digital object identifier of the publication from which the CIF was taken|
| `ICSD ID` | Inorganic Crystal Structure Database ID of the structure if it can be found in that database |
| `Laskowski ID` | Entry number (in order of appearance) in the supplementary information of [Forrest A. L. Laskowski et al., Energy Environ. Sci., 16, 1264 (2023)](https://pubs.rsc.org/en/content/articlelanding/2023/ee/d2ee03499a#!) if the entry is also in that database|
| `Liion ID` | Entry number in the [The Liverpool Ionics Dataset](http://pcwww.liv.ac.uk/~msd30/lmds/LiIonDatabase.html) if the entry is also in that database|

## Citation

If you use OBELiX, please cite [our paper](https://arxiv.org/abs/2502.14234)

```
@article{therrien2025obelix,
  title   = {OBELiX: A Curated Dataset of Crystal Structures and Experimentally Measured Ionic Conductivities for Lithium Solid-State Electrolytes},
  author  = {Félix Therrien and Jamal Abou Haibeh and Divya Sharma and Rhiannon Hendley and Alex Hernández-García and Sun Sun and Alain Tchagang and Jiang Su and Samuel Huberman and Yoshua Bengio and Hongyu Guo and Homin Shin},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2502.14234}
}

```


