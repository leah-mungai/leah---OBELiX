import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifFile
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

noise_level = 0.01
neighbor_cutoff = 1

def symmetrize(new_structure, neighbor_cutoff=neighbor_cutoff):
    indices = []
    for j, neighbors in enumerate(new_structure.get_all_neighbors(neighbor_cutoff)):
        if j in indices:
            continue
        coords = [new_structure[j].coords]
        for n in neighbors:
            if n.label == new_structure[j].label:
                indices.append(n.index)
                coords.append(n.coords)
        new_coords = np.array(coords).mean(axis=0)
        new_structure[j].coords = new_coords

    new_structure.remove_sites(indices)

    # indices = []
    # for j, neighbors in enumerate(new_structure.get_all_neighbors(1)):
    #     if j in indices:
    #         continue
    #     species_dict = new_structure[j].species.as_dict()
    #     for n in neighbors:
    #         indices.append(n.index)
    #         species_dict.update(n.species.as_dict())

    #      if sum(species_dict.values()) > 1:
    #          print("WARNING: sum of species > 1", species_dict)
             
    # new_structure.remove_sites(indices)
            
    return new_structure

def add_random_noise(filename, rng):
    
    with open(filename,"r") as f:
        s = f.read()

    cif = CifFile.from_str(s)

    assert len(cif.data) == 1

    k = next(iter(cif.data.keys()))

    vals = []
    for dim in ["x","y","z"]:
        vals.append(np.array(cif.data[k].data[f"_atom_site_fract_{dim}"], dtype=float))

    vals = np.array(vals)
    mask_idx = np.arange(vals.shape[1])
    for i, pos in enumerate(vals.T):
        for j in range(i+1, vals.shape[1]):
            if np.allclose(np.mod(pos,1), np.mod(vals[:,j],1), atol=1e-5):
                mask_idx[j] = mask_idx[i]
    
    noise = rng.normal(0, noise_level, vals.shape)
    new_vals = vals + noise[:,mask_idx]

    for i, dim in enumerate(["x","y","z"]):
        cif.data[k].data[f"_atom_site_fract_{dim}"] = new_vals[i,:].astype(str).tolist()

    new_structure =  Structure.from_str(str(cif), fmt="cif")
    
    if len(new_structure) > len(structure):
        new_structure = symmetrize(new_structure)
    elif len(new_structure) < len(structure):
        print("WARNING: number of atoms decreased!")

    return new_structure
            
if __name__ == "__main__":

    rng = np.random.default_rng(seed=59603)
    
    data = pd.read_excel("raw_labeled.xlsx", index_col="ID")
    folder = "anon_cifs/"
    problems = []
    for i, row in data.iterrows():
    
        if row["Cif ID"] != "done" and row["ICSD ID"] is not None:
            continue

        filename = folder + i + ".cif"

        print("Processing", i)

        structure = Structure.from_file(filename)

        sym_structure = symmetrize(structure.copy())

        if len(sym_structure) < len(structure):
            breakpoint()
        
        for attempt in range(10):
        
            new_structure = add_random_noise(filename, rng)

            if (len(structure) == len(new_structure) and 
                structure.composition == new_structure.composition and
                SpacegroupAnalyzer(structure, symprec=2e-3).get_space_group_number() == SpacegroupAnalyzer(new_structure, symprec=2e-3).get_space_group_number()):
                new_structure.to("randomized_cifs/" + i + ".cif", symprec=2e-3)
                break
            
        else:
            new_structure.to("randomized_cifs/" + i + ".cif", symprec=2e-3)
            print(f"PROBLEM: {i}")
            problems.append(i)
            continue

    print(f"There were {len(problems)} problems.")
    print(problems)
        # TODO: 1. new_structure can be larger if two species are on the same site, maybe move identical coords by the same amount?
        # 2. Large structure may have atoms labeled as different but are acutually in the same orbit? 
