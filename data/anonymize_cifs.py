import pandas as pd
from pymatgen.core import Structure, Composition
from ase.io import read, write
import re
import pathlib
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np
import warnings

def get_occ(atoms):
    symbols = list(atoms.symbols)
    coords = list(atoms.get_positions())
    occ_info = atoms.info.get('occupancy')
    kinds = atoms.arrays.get('spacegroup_kinds')
    occupancies = []
    if occ_info is not None and kinds is not None:
        occupancies = [occ_info[str(k)] for k in kinds]
    else:
        occupancies = [{s: 1.0} for s in symbols]
    return occupancies

def atoms_to_structure(atoms):
    structure = AseAtomsAdaptor().get_structure(atoms)
    to_remove = []
    assert len(structure) == len(atoms)
    for i, occ in enumerate(get_occ(atoms)):
        if all(np.array(list(occ.values())) == 0):
            to_remove.append(i)
        else:
            structure[i]._species = structure[i].species.from_dict(occ)
    structure.remove_sites(to_remove)
    return structure
    
def ase_read(filename, **kwargs):
    filename = pathlib.Path(filename)
    with open(filename,"r") as f:
        s = f.read()
    #s = re.sub("\([0-9]+\)", "", s)
    s = re.sub("#.*", "", s)
    s = s[max(s.find("data_"), 0):]
    tmpname = filename.with_stem(filename.stem + "_tmp")
    with open(tmpname,"w") as f:
        f.write(s)
    atoms = read(tmpname, **kwargs)
    pathlib.Path(tmpname).unlink()
    return atoms

data = pd.read_excel("raw.xlsx", index_col="ID")

def is_fractional(formula):
    broken_down_formula = re.findall("([A-Za-z]{1,2})([\.0-9]*)", formula)
    for elem, number in broken_down_formula:
        if number == "":
            number = "1"
        if float(number) != float(int(number)):
            return True
    return False

folder = "new_cifs/cifs/"
problems = dict()
for i, row in data.iterrows():

    if row["Cif ID"] != "done":
        continue

    print("Processing", i)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="crystal system .*")
            atoms = ase_read(folder + i + ".cif", fractional_occupancies=True, format="cif")
    except:
        print(i, "WARNING: ASE could not read the cif")
        problems[i] = "ASE could not read the cif"
        #structure = Structure.from_file(folder + i + ".cif")
        #atoms = AseAtomsAdaptor().get_atoms(structure)
        continue
    
    structure = atoms_to_structure(atoms)

    try:
        structure.to("anon_cifs/" + i + ".cif", symprec=2e-3)
    except:
        problems[i] = "Could not write anonymized cif"
    
    #write("anon_cifs/" + i + ".cif", atoms, format="cif")

    new_atoms = read("anon_cifs/" + i + ".cif", fractional_occupancies=True)

    occ_info = atoms.info.get('occupancy')
    kinds = atoms.arrays.get('spacegroup_kinds')

    new_occ_info = new_atoms.info.get('occupancy')
    new_kinds = new_atoms.arrays.get('spacegroup_kinds')

    #assert SpacegroupAnalyzer(structure, symprec=2e-3).get_space_group_number() == row["Space group #"]

    n_zero_occ = len([1 for occ in get_occ(atoms) if sum(list(occ.values())) == 0])
    
    if len(atoms) - n_zero_occ != len(new_atoms):
        problems[i] = "Different number of atoms"
        continue
        
    if new_occ_info is None and occ_info is not None:
        problems[i] = "Fractional occupancies, but none found in anonymized cif"
        continue
    elif occ_info is None:
        if is_fractional(row["Reduced Composition"]):
            problems[i] = "Fractional occupancies, but none found in original cif"
        continue

    
    for site, atom in occ_info.items():
        for a in new_occ_info.values():
            for k, v in atom.copy().items():
                if v == 0.0:
                    atom.pop(k)
            if a == atom or atom == {}:
                break
        else:
            problems[i] = "Different occupancies"
            continue

print(len(problems), "problems found")
print(problems)

        
