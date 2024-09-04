import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pymatgen.core.periodic_table import Element

pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)


def find_duplicates(df, subset):
    # Check duplicates
    n_data = len(df)
    all_dup_rows = df.duplicated(subset=subset, keep=False)
    dup_rows = df.duplicated(subset=subset, keep="first")
    duplicated_df = df[all_dup_rows]
    unique_comp_df = df[~dup_rows]

    print("Number of unique data points: ", len(unique_comp_df), "/", n_data)

    repeated_comps = duplicated_df.drop_duplicates(subset=subset)

    print("Number of repeated compositions: ", len(repeated_comps), "/", len(duplicated_df))

    df_list = []
    conds = []
    scatter_points = []
    ymins = []
    ymaxs = []
    stds = []
    for i, dat in enumerate(repeated_comps.iterrows()):
        index, row = dat
        condition = np.all((duplicated_df[subset] == row[subset]).to_numpy(), axis=1)
        same_comp = duplicated_df[condition]
        df_list.append(same_comp)
        diff = np.log10(same_comp["Ionic conductivity (S cm-1)"]).max() - np.log10(same_comp["Ionic conductivity (S cm-1)"]).min()
        if diff == 0:
            print(index, len(same_comp), diff)
            print(same_comp)
            print()
        conds.append(same_comp["Ionic conductivity (S cm-1)"].median())
        scatter_points.append([[i]*len(df_list[i]),same_comp["Ionic conductivity (S cm-1)"].to_list()])
        ymins.append(same_comp["Ionic conductivity (S cm-1)"].min())
        ymaxs.append(same_comp["Ionic conductivity (S cm-1)"].max())
        stds.append(np.log10(same_comp["Ionic conductivity (S cm-1)"]).std())

    print("Average log standard deviation: ", np.mean(stds))

    idx = np.argsort(conds)

    revidx = np.argsort(idx)
    
    scatter_points = np.concatenate([scatter_points[i] for i in idx], axis=1)
    ymins = np.array(ymins)[idx]
    ymaxs = np.array(ymaxs)[idx]
    names = repeated_comps.index.to_list()
    names = [names[i] for i in idx]
    
    scatter_points[0, :] = revidx[scatter_points[0,:].astype(int)]
    
    fig, ax = plt.subplots()
    ax.scatter(scatter_points[0,:], scatter_points[1,:], s=10)
    ax.vlines(np.arange(len(repeated_comps)), ymins, ymaxs, alpha=0.5, color='c')
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(repeated_comps)), names, rotation=90)
    ax.grid()

def get_atomic_volume(formula):
    broken_down_formula = re.findall("([A-Za-z]{1,2})([\.0-9]*)", formula)
    vol = 0
    for elem, number in broken_down_formula:
        if elem not in ['Li', 'Na', 'K', 'Rb', 'Cs']:
            vol += 4*np.pi*Element(elem).atomic_radius**3/3 * float(number)
    return vol

def get_total_mass(formula):
    broken_down_formula = re.findall("([A-Za-z]{1,2})([\.0-9]*)", formula)
    mass = 0
    for elem, number in broken_down_formula:
        mass += Element(elem).atomic_mass * float(number)
    return mass

def get_cell_volume(a,b,c,alpha,beta,gamma):
    return a * b * c * (1 - np.cos(np.deg2rad(alpha))**2 - np.cos(np.deg2rad(beta))**2 - np.cos(np.deg2rad(gamma))**2 + 2 * np.cos(np.deg2rad(alpha)) * np.cos(np.deg2rad(beta)) * np.cos(np.deg2rad(gamma)))**(1/2)
        
def get_number_of_Li(formula):
    broken_down_formula = re.findall("([A-Za-z]{1,2})([\.0-9]*)", formula)
    for elem in broken_down_formula:
        if elem[0] == 'Li':
            return float(elem[1])

def get_density_of_carriers(data):
    volumes = get_cell_volume(data['a'], data['b'], data['c'], data['alpha'], data['beta'], data['gamma'])
    n_Li = data["Composition"].apply(get_number_of_Li)    

    return n_Li / volumes

def get_density(data):
    volumes = get_cell_volume(data['a'], data['b'], data['c'], data['alpha'], data['beta'], data['gamma'])
    masses = data["Composition"].apply(get_total_mass)
    
    return masses / volumes

def density_of_carriers(data):

    Lidensity = get_density_of_carriers(data)

    density = get_density(data)

    atomic_volumes = data["Composition"].apply(get_atomic_volume)
    volumes = get_cell_volume(data['a'], data['b'], data['c'], data['alpha'], data['beta'], data['gamma'])
    
    space = (volumes-atomic_volumes)/volumes

    plt.figure()
    plt.hist(np.log10(Lidensity), bins=50)
    plt.title("Density of carriers")

    plt.figure()
    plt.hist(np.log10(density), bins=50)
    plt.title("Density")

    plt.figure()
    plt.hist(space, bins=50)
    plt.title("Space in unit cell")
    
    a, b = np.linalg.lstsq(np.vstack([np.log10(Lidensity), np.ones(len(Lidensity))]).T, np.log10(data["Ionic conductivity (S cm-1)"]))[0]

    plt.figure()
    plt.plot(np.log10(Lidensity), np.log10(data["Ionic conductivity (S cm-1)"]), 'k.')
    plt.plot(np.log10(Lidensity), a*np.log10(Lidensity)+b, 'r-')
    plt.xlabel("Log(density of Li carriers)")
    plt.ylabel("Log(Ionic conductivity)")
    
    plt.figure()
    plt.plot(np.log10(space), np.log10(data["Ionic conductivity (S cm-1)"]), 'o')

    plt.figure()
    plt.plot(np.log10(density), np.log10(data["Ionic conductivity (S cm-1)"]), 'o')
    plt.xlabel("Log(density)")
    plt.ylabel("Log(Ionic conductivity)")
    
    
if __name__ == "__main__":

    data = pd.read_csv('processed.csv', index_col="ID")
    
    find_duplicates(data, ['Composition', 'Space group number'])

    density_of_carriers(data)

    plt.figure()
    plt.plot(data["Space group number"], np.log10(data["Ionic conductivity (S cm-1)"]), 'o')
 
    
    plt.show()
