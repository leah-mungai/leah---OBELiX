import pandas as pd
import urllib.request
import tarfile
from pathlib import Path
from pymatgen.core import Structure
import warnings
from tqdm import tqdm

class OBELiX():
    def __init__(self, data_path="./data", no_cifs=False):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            self.download_data(self.data_path)

        self.dataframe = self.read_data(self.data_path, no_cifs)

        self.index = self.dataframe.index

    def download_data(self, output_path, commit_id=None):
        print("Downloading data...", end="")
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)

        if commit_id is None:
            commit_id = "main"

        tar_url = f"https://raw.githubusercontent.com/NRC-Mila/OBELiX/{commit_id}/data/downloads/all_cifs.tar.gz"

        fileobj = urllib.request.urlopen(tar_url)
        tar = tarfile.open(fileobj=fileobj, mode="r|gz")
        tar.extractall(output_path)

        csv_url = f"https://raw.githubusercontent.com/NRC-Mila/OBELiX/{commit_id}/data/downloads/all.csv"
        df = pd.read_csv(csv_url, index_col="ID")
        df.to_csv(output_path / "all.csv")

        test_csv_url = f"https://raw.githubusercontent.com/NRC-Mila/OBELiX/{commit_id}/data/downloads/test.csv"
        df = pd.read_csv(test_csv_url, index_col="ID")
        df.to_csv(output_path / "test.csv")
        
        print("Done.")
        
    def read_data(self, data_path, no_cifs=False):
        
        data = pd.read_csv(self.data_path / "all.csv", index_col="ID")

        if no_cifs:
            return data
        
        cif_path = Path(data_path) / "all_randomized_cifs"
        
        struc_dict = {}
            
        print("Reading CIFs...")
        for i, row in tqdm(data.iterrows(), total=len(data)):

            filename = (cif_path / i).with_suffix(".cif")
        
            if row["Cif ID"] == "done":
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="We strongly encourage explicit .*")
                    warnings.filterwarnings("ignore", message="Issues encountered .*")
                    structure = Structure.from_file(filename)
            else:
                structure = None
                    
            struc_dict[i] = structure

        data["structure"] = pd.Series(struc_dict)
        
        return data

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.dataframe.loc[idx]

    def to_numpy(self):
        return self.dataframe.to_numpy()

    def to_dict(self):
        return self.dataframe.to_dict()

    def keys(self):
        return self.dataframe.keys()

    def items(self):
        return self.dataframe.items()

    def iterrows(self):
        return self.dataframe.iterrows()
        
           
