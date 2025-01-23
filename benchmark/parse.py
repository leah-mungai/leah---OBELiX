import pandas as pd
from mendeleev import get_all_elements
from pymatgen.core import Composition
from pathlib import Path


def flatten_comp(true_comp, compdf):
    for i, item in true_comp.iteritems():
        comp = Composition(item).get_el_amt_dict()
        for k, v in comp.items():
            compdf[k][i] = v
    return compdf


def read_xy(csv_fptr, cif=True, partial=True):
    data = pd.read_csv(
        csv_fptr,
        index_col="ID",
        usecols=[
            "ID",
            "Composition",
            "Space group number",
            "a",
            "b",
            "c",
            "alpha",
            "beta",
            "gamma",
            "CIF",
            "Ionic conductivity (S cm-1)",
        ],
    )

    comp_df = {}
    for e in get_all_elements():
        comp_df[e.symbol] = [0 for _ in range(len(data))]
    comp_df = pd.DataFrame(comp_df)
    comp_df = comp_df.set_index(data.index)
    comp_df = flatten_comp(data["Composition"], comp_df)
    comp_df = comp_df.loc[:, (comp_df != 0).any(axis=0)]
    if not partial:
        comp_df = comp_df.map(round, axis=1)
    data = data.drop("Composition", axis=1)
    data = pd.concat([comp_df, data], axis=1)
    return data


if __name__ == "__main__":
    read_xy("/home/mila/d/divya.sharma/ionic-conductivity/data/processed.csv", False)
    ### Latest
    # mean
    # a                               9.307045
    # b                               9.213725
    # c                              12.418862
    # alpha                          90.260857
    # beta                           91.176803
    # gamma                          97.260356
    # Ionic conductivity (S cm-1)     0.000907
    # Std
    # a                               3.117529
    # b                               3.488247
    # c                               5.601230
    # alpha                           2.425697
    # beta                            5.300615
    # gamma                          13.154874
    # Ionic conductivity (S cm-1)     0.002588
