from ICSDClient import ICSDClient
import pandas as pd
import numpy as np
import re
from habanero import Crossref

def get_bib_info(doi):

    cr = Crossref()

    result = cr.works(ids=doi)

    list_authors = result["message"]["author"]

    family_names = [author["family"] for author in list_authors]
    
    year = result["message"]["created"]["date-parts"][0][0]

    title = result["message"]["title"][0]

    return " ".join(family_names), year, title

def get_bib_info(doi):

    cr = Crossref()

    result = cr.works(ids=doi)

    list_authors = result["message"]["author"]

    year = result["message"]["created"]["date-parts"][0][0]

    title = result["message"]["title"][0]

    return list_authors, year, title

data = pd.read_excel("raw_from_google.xlsx")

number_of_matches = 0
number_of_multi_matches = 0
number_of_partial_compositions = 0
client =  ICSDClient("ICSD_login", "ICSD_password")

for i, row in data.iterrows():

    if row["Cif ID"] == "done":
        continue

    formula = re.findall("([A-Za-z]{1,2})([0-9\.]*)\s*", row["True Composition"])

    composition = ""
    for element, amount in formula:
        reduced_amount = amount/row["Z"]
        composition += f"{element}:{reduced_amount}:{reduced_amount} "

    cellparamstring = f"{row['a']} {row['b']} {row['c']} {row['alpha']} {row['beta']} {row['gamma']}"   

    authors, article, publicationyear = get_bib_info(row["DOI"])
    
    matches = client.advanced_search({"composition": composition, "cellparameters": cellparamstring, "authors":authors, "article":article, "publicationyear":publicationyear}, search_type="and")

    #client.fetch_cifs(matches, cif_dir="cifs")
    
    print(matches)
        
