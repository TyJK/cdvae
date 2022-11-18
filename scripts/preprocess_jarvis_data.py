""" Converts Jarvis data to pymatgen molecule objects and splits to train/val/test """


import pandas as pd

from pymatgen.core.structure import Molecule
row_to_mol = lambda row: str(Molecule(species=row["elements"], coords=row["coords"]))

def main():
    print("reading csv...")
    jarvis = pd.read_csv("~/Projects/mila/molecule-representation-tda/data/raw/jarvis-oct-18.csv", low_memory=False)
    jarvis = jarvis.groupby("dataset_id").first()  # removes duplicates

    jarvis = jarvis[jarvis["dataset_name"]!="QM9"]

    print("processing cols...")
    for col in ["coords", "elements"]:
        try:
            jarvis[col] = jarvis[col].apply(eval)  # string-to-list op
        except Exception as e:
            print(f"skipping {col} after error, received: {e}")

    jarvis = jarvis.round(8)  # smooths out compute tails
    # import pdb; pdb.set_trace()

    print("building mols...")
    mols = jarvis.apply(row_to_mol, axis=1)
    del jarvis["coords"], jarvis["elements"]
    jarvis["molecule"] = mols

    # stratified random sample 60/20/20
    print("splitting train/val/test...")
    train = jarvis.groupby("dataset_name").sample(frac=.6)
    not_train_idx = list(set(jarvis.index).difference(train.index))

    val = jarvis.loc[not_train_idx].groupby("dataset_name").sample(frac=.5)
    test_idx = list(set(jarvis.index).difference(train.index).difference(val.index))

    test = jarvis.loc[test_idx]

    print("writing to disk...")
    train.to_csv("~/Projects/mila/cdvae/data/jarvis/train.csv")
    val.to_csv("~/Projects/mila/cdvae/data/jarvis/val.csv")
    test.to_csv("~/Projects/mila/cdvae/data/jarvis/test.csv")


if __name__ == "__main__":
    main()
