import pandas as pd
from pymatgen.core import Lattice, Structure, Molecule

data = pd.read_excel("raw.xlsx", index_col="ID")

for i, row in data.iterrows():

    if row["Cif ID"] != "done":
        continue
        
    structure = Structure.from_file("cifs/" + i + ".cif")

    structure.to("anon_cifs/" + i + ".cif")
