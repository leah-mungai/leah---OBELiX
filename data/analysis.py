import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from pymatgen.core.periodic_table import Element
import seaborn as sns
from matplotlib.colors import ListedColormap


plt.rcParams['text.usetex'] = True

pd.set_option('display.max_columns', 12)
pd.set_option('display.width', 1000)

palette = ["#e60049", "#0bb4ff", "#50e991", "#e6d800", "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", "#00bfa0"]

spg_colors = np.empty(230, dtype="<U10")
spg_colors[:2] = palette[0]
spg_colors[2:15] = palette[1]
spg_colors[15:74] = palette[2]
spg_colors[74:142] = palette[3]
spg_colors[142:167] = palette[4]
spg_colors[167:] = palette[5]
spg_colors = np.array(spg_colors)

spg_cmp = ListedColormap(spg_colors)

def distance(df):
    #averages = df[["a", "b", "c", "alpha", "beta", "gamma"]].mean(axis=0)
    #diffs = abs(df[["a", "b", "c", "alpha", "beta", "gamma"]] - averages[["a", "b", "c", "alpha", "beta", "gamma"]])/averages[["a", "b", "c", "alpha", "beta", "gamma"]]
    vals = df[["a", "b", "c", "alpha", "beta", "gamma"]].to_numpy()
    diffs = abs(vals[:, None, :] - vals)/vals
    return diffs.max(axis=(1,2))
    
def best_permutation_dist(df):
    for i, cid_row in enumerate(df.iterrows()):
        cid, row = cid_row
        if i == 0:
            continue
        best_distance = distance(df.iloc[:i+1])
        best_permutation = [0, 1, 2]
        permutations = [[0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        df_cp = df.copy().loc[cid]
        for j in permutations:
            df.loc[cid, ["a", "b", "c"]] = df_cp[["a", "b", "c"]].to_numpy()[j]
            df.loc[cid, ["alpha", "beta", "gamma"]] = df_cp[["alpha", "beta", "gamma"]].to_numpy()[j]
            if distance(df.iloc[:i+1]).max() < best_distance.max():
                best_distance = distance(df.iloc[:i+1]).max()
                best_permutation = j
        df.loc[cid, ["a", "b", "c"]] = df_cp[["a", "b", "c"]].to_numpy()[best_permutation]
        df.loc[cid, ["alpha", "beta", "gamma"]] = df_cp[["alpha", "beta", "gamma"]].to_numpy()[best_permutation]
    return distance(df)
                
def plot_similar(df, subset, palette = ["red", "blue"]):
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
    stds_alt = []
    diffs = []
    same_comps = []
    dists = []
    for i, dat in enumerate(repeated_comps.iterrows()):
        index, row = dat
        condition = np.all((duplicated_df[subset] == row[subset]).to_numpy(), axis=1)
        same_comp = duplicated_df[condition]
        df_list.append(same_comp)
        diff = np.log10(same_comp["Ionic conductivity (S cm-1)"]).max() - np.log10(same_comp["Ionic conductivity (S cm-1)"]).min()
        dists.append(best_permutation_dist(same_comp))
        same_comps.append(same_comp.index)
        diffs.append(diff)
        conds.append(same_comp["Ionic conductivity (S cm-1)"].median())
        scatter_points.append([[i]*len(df_list[i]),same_comp["Ionic conductivity (S cm-1)"].to_list()])
        ymins.append(same_comp["Ionic conductivity (S cm-1)"].min())
        ymaxs.append(same_comp["Ionic conductivity (S cm-1)"].max())
        stds.append(np.log10(same_comp["Ionic conductivity (S cm-1)"]).std())
        stds_alt.append(np.log10(same_comp["Ionic conductivity (S cm-1)"]) - np.log10(same_comp["Ionic conductivity (S cm-1)"]).mean())

    print("Average log standard deviation: ", np.mean(stds))
    
    print("Average log standard deviation (RMSE like): ", np.sqrt(np.mean(np.concatenate(stds_alt)**2)))
    print("Average log difference with mean (MAE like): ", np.mean(abs(np.concatenate(stds_alt))))


    plt.figure()
    #plt.boxplot(np.concatenate(stds_alt))
    diff_df = pd.DataFrame(np.concatenate(stds_alt), columns=["Difference with mean log$_{10}$(Ionic Conductivity)"])
    sns.violinplot(data=diff_df["Difference with mean log$_{10}$(Ionic Conductivity)"], inner_kws=dict(box_width=15, whis_width=2, color=".8"))

    plt.savefig("violon.png")
    
    sorted_idx = [same_comps[i] for i in np.argsort(diffs)]

    print("Closest compounds:")
    print(duplicated_df.loc[sorted_idx[0]])
    print()
    print(duplicated_df.loc[sorted_idx[1]])
    print("Furthest compounds:")
    print(duplicated_df.loc[sorted_idx[-1]])
    print()
    print(duplicated_df.loc[sorted_idx[-2]])
    
    idx = np.argsort(conds)

    revidx = np.argsort(idx)
    
    scatter_points = np.concatenate([scatter_points[i] for i in idx], axis=1)
    ymins = np.array(ymins)[idx]
    ymaxs = np.array(ymaxs)[idx]
    names = repeated_comps.index.to_list()
    names = [", ".join(df_list[i].index) for i in idx]
    test_mask = np.array([row["in_test"] for i in idx for _, row in df_list[i].iterrows()])
    colors = np.array([100*dist for i in idx for dist in dists[i]])
    
    scatter_points[0, :] = revidx[scatter_points[0,:].astype(int)]

    col_max = np.max(colors)
    col_min = np.min(colors)
    
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    cm = plt.cm.get_cmap('winter')
    ax.vlines(np.arange(len(repeated_comps)), ymins, ymaxs, alpha=0.3, color='k')
    s = ax.scatter(scatter_points[0,:][~test_mask], scatter_points[1,:][~test_mask], s=30, c=colors[~test_mask], cmap=cm, alpha=0.8, vmin=col_min, vmax=col_max, label="Training data")
    s = ax.scatter(scatter_points[0,:][test_mask], scatter_points[1,:][test_mask], s=30, c=colors[test_mask], cmap=cm, alpha=0.8, marker='x', vmin=col_min, vmax=col_max, label="Test data")
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(repeated_comps)), names, rotation=45, ha='left')
    ax.grid()
    ax.xaxis.tick_top()
    ax.set_ylabel(r"Log$_10$(Ionic conductivity)")
    cbar = plt.colorbar(s)
    cbar.set_label('Max \% difference between lattice parameters', rotation=270, labelpad=15)
    fig.tight_layout()

    plt.savefig("differences_same_struct.png")
    
    diffs = [abs(d) for i in idx for d in np.log10(df_list[i]["Ionic conductivity (S cm-1)"]) - np.log10(df_list[i]["Ionic conductivity (S cm-1)"]).mean()]
    dists = [d for i in idx for d in dists[i]]
    
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.scatter(dists, diffs, s=10, c=scatter_points[1,:], cmap=cm, alpha=0.8)
    ax.set_xlabel("Cell parameters distance with set mean")
    ax.set_ylabel("Log(Ionic conductivity) - mean")
    

def get_atomic_volume(formula):
    broken_down_formula = re.findall("([A-Za-z]{1,2})([\.0-9]*)", formula)
    vol = 0
    for elem, number in broken_down_formula:
        if elem not in ['Li']: #['Li', 'Na', 'K', 'Rb', 'Cs']:
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
    #s = plt.scatter(np.log10(Lidensity), np.log10(space), c=np.log10(data["Ionic conductivity (S cm-1)"]), alpha=0.8, s=10, cmap='winter')
    s = plt.scatter(np.log10(Lidensity), np.log10(space), c=data["Space group number"].to_numpy()-1, alpha=0.8, s=10, cmap=spg_cmp, vmin=0, vmax=229)
    cbar = plt.colorbar(s)

    # cbar.set_label("Ionic conductivity (S cm-1)", rotation=270, labelpad=15)
    cbar.set_label("Space Group", rotation=270, labelpad=15)
    

    plt.xlabel("Log$_{10}$(density of Li carriers)")
    plt.ylabel("Log$_{10}$(space in unit cell)")

    plt.savefig("Scatter_Li_density_log.png")
    
    plt.figure()
    #s = plt.scatter(Lidensity, space, c=np.log10(data["Ionic conductivity (S cm-1)"]), alpha=0.8, s=10, cmap='winter')
    s = plt.scatter(Lidensity, space, c=data["Space group number"].to_numpy()-1, alpha=0.8, s=10, cmap=spg_cmp, vmin=0, vmax=229)
    cbar = plt.colorbar(s)
    
    #cbar.set_label("Ionic conductivity (S cm-1)", rotation=270, labelpad=15)
    cbar.set_label("Space Group", rotation=270, labelpad=15)

    plt.xlabel("Density of Li carriers")
    plt.ylabel("Space in unit cell")

    plt.savefig("Scatter_Li_density.png")

def distributions(data):

    plt.figure()
    ax=sns.displot(data=data, x="Ionic conductivity (S cm-1)", hue="CIF", kde=True, multiple="stack", row="in_test", log_scale=True, hue_order=["No Match", "Close Match", "Match"], facet_kws={'sharey': False}, palette=palette)

    plt.savefig("IC_distribution.png")

    plt.figure()
    ax=sns.displot(data=data, x="Space group number", hue="CIF", kde=True, multiple="stack", row="in_test", hue_order=["No Match", "Close Match", "Match"], facet_kws={'sharey': False}, palette=palette)
    sns.move_legend(ax, "upper left")
    #plt.tight_layout()

    
    plt.figure()
    #plt.plot(data["Space group number"], np.log10(data["Ionic conductivity (S cm-1)"]), 'o')
    plt.hist(data["Space group number"], bins=50)
    plt.axvline(x=2, color='r', linestyle='--')
    plt.axvline(x=15, color='r', linestyle='--')
    plt.axvline(x=74, color='r', linestyle='--')
    plt.axvline(x=142, color='r', linestyle='--')
    plt.axvline(x=167, color='r', linestyle='--')

def get_number_of_elements(formula):
    broken_down_formula = re.findall("([A-Za-z]{1,2})([\.0-9]*)", formula)
    return pd.Series({elem[0]: 1 for elem in broken_down_formula})
    
def periodic_table_map(data):
    from pymatviz import ptable_heatmap
    import matplotlib.colors as colors
    occurences = data["Composition"].apply(get_number_of_elements).sum()
    #ptable_heatmap(occurences, colormap="winter", cbar_kwargs={"norm": colors.LogNorm(vmin=1, vmax=occurences.max())})
    plt.figure()
    ptable_heatmap(occurences, colormap="Spectral_r")

    plt.savefig("prediodic_table.png")
    
def spg_charts(data):
    from pymatviz import spacegroup_sunburst
    # plotly
    fig = spacegroup_sunburst(data["Space group number"][data["in_test"]], show_counts='value', color_discrete_sequence=palette)
    fig.show()

    fig = spacegroup_sunburst(data["Space group number"][~data["in_test"]], show_counts='value', color_discrete_sequence=palette)
    fig.show()
    
if __name__ == "__main__":

    data = pd.read_csv('processed.csv', index_col="ID")
    test =  pd.read_csv('../test_idx.csv', index_col="ID")
    data["in_test"] = data.index.isin(test.index)

    # Similar entries plots
    plot_similar(data, ['Composition', 'Space group number'])

    # Relevant quantities plots
    density_of_carriers(data)

    # Distributions
    distributions(data)

    # Periodec table heatmap
    periodic_table_map(data)
    
    spg_charts(data)
    
    plt.show()
