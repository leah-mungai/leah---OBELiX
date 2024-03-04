import numpy as np
from mendeleev.fetch import fetch_table
import re
import pandas as pd

def is_same_formula(formula_string1, formula_string2):
    f1 = re.findall("([A-Za-z]{1,2})([0-9\.]*)\s*", formula_string1)
    f2 = re.findall("([A-Za-z]{1,2})([0-9\.]*)\s*", formula_string2)
    if len(f1) != len(f2):
        return False
    for elem, count in f1:
        if (elem, count) not in f2:
            return False
    return True

def get_paper_from_laskowski(material, laskowski_data):
    for j, paper_material in enumerate(laskowski_data[:,0]):
        if (material == paper_material or
            is_same_formula(material, paper_material) or
            paper_material + laskowski_data[min(j+1, laskowski_data.shape[0]-1),0] == material or
            paper_material + laskowski_data[min(j+2, laskowski_data.shape[0]-1),0] == material):
            paper_number = laskowski_data[j,-1]
            if paper_number == "":
                if paper_material + laskowski_data[min(j+1, laskowski_data.shape[0]-1),0] == material and laskowski_data[j+1,-1] != "":
                    paper_number = laskowski_data[j+1,-1]
                elif paper_material + laskowski_data[min(j+2, laskowski_data.shape[0]-1),0] == material and (laskowski_data[j+1,-1] != "" or laskowski_data[j+2,-1] != ""):
                    if laskowski_data[j+1,-1] == "":
                        paper_number = laskowski_data[j+2,-1]
                    else:
                        paper_number = laskowski_data[j+1,-1]
                else:
                    print(material, "Missing paper number")
                    paper_number = "MP"
            return "L-" + paper_number
    return None

def get_paper_from_liion(material, liion_data):
    for j, liion_material in enumerate(liion_data[:,1]):
        if material == liion_material or is_same_formula(material, liion_material):
            return liion_data[j, 2]
    return None

def get_paper_from_unidentified(material, uni_data):
    for j, uni_material in enumerate(uni_data[:,0]):
        if material == uni_material or is_same_formula(material, uni_material):
            return uni_data[j, 1]
    return None

def add_paper_info(database):

    laskowski_database = "laskowski_semi-fromatted.csv"
    liion_database = "LiIonDatabase.csv"
    uni_datatbase = "unidentified_with_refs.csv"

    homin_data = database.to_numpy()
    laskowski_data = np.genfromtxt(laskowski_database, delimiter=',', dtype=str)
    liion_data = np.genfromtxt(liion_database, delimiter=',', dtype=str, skip_header=1)
    uni_data = np.genfromtxt(uni_datatbase, delimiter=',', dtype=str)
    
    new_homin_data = np.concatenate((homin_data,np.ones((homin_data.shape[0],1))), axis=1)
    
    not_found_count = 0
    not_found = open("unidentified.txt", "w")
    
    for i, material in enumerate(homin_data[:,0]):
        paper_info = get_paper_from_laskowski(material, laskowski_data)
        if paper_info is None:
            paper_info = get_paper_from_liion(material, liion_data)
            if paper_info is None:
                paper_info = get_paper_from_unidentified(material, uni_data)
                if paper_info is None:
                    not_found_count += 1
        new_homin_data[i,-1] = paper_info
        
    not_found.close()

    new_homin_data = pd.DataFrame(new_homin_data, columns=database.columns.tolist() + ["paper"])
    
    return new_homin_data


if __name__ == "__main__":
    
    homin_database = "20231204v1.csv"

    homin_data = pd.read_csv(homin_database, dtype=str)

    new_homin_data = add_paper_info(homin_data)
    
    # thresh = 1e-8
    # final.loc[final["IC"] < thresh, "IC"] = thresh
    final = new_homin_data.drop(["Reduced Composition", "Z"], axis=1)
    final.rename({"True Composition": "Composition"}, axis=1, inplace=True)
    cols = list(final.columns)
    cols = [cols[0]] + cols[2:-1] + [cols[1]] + [cols[-1]]
    final = final[cols]

    final.to_csv("20231204v1_preproc.csv")
    
    
