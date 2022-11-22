""" Converts Jarvis data to pymatgen molecule objects and splits to train/val/test """

import pandas as pd

from cdvae.common.constants import ATOMIC_SYMBOL_TO_NUMBER_MAP

def main():
    print("reading csv...")
    jarvis = pd.read_csv("~/Projects/mila/molecule-representation-tda/data/raw/jarvis-oct-18.csv", low_memory=False)
    jarvis = jarvis.groupby("dataset_id").first()  # removes duplicates

    jarvis = jarvis[jarvis["dataset_name"]!="QM9"]

    # print("processing cols...")
    jarvis["elements"] = jarvis["elements"].apply(
        lambda elem_list: str(list(map(lambda sym: ATOMIC_SYMBOL_TO_NUMBER_MAP[sym], eval(elem_list)
    ))))

    jarvis["num_atoms"] = jarvis["elements"].apply(len)

    jarvis = jarvis.round(8)  # smooths out compute tails

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
