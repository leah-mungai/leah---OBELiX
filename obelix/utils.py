import numpy as np
from pymatgen.core import Composition

def round_partial_occ(structure):

    structure = structure.copy()
    to_remove = []
    for i, site in enumerate(structure):   
        for k,v in site.species.as_dict().items():
            v = int(round(v))
            if v == 1:
                new_occ = {k: 1}
                structure[i]._species = Composition(new_occ) 
                break
        else:
                to_remove.append(i)
    structure.remove_sites(to_remove)
    return structure
